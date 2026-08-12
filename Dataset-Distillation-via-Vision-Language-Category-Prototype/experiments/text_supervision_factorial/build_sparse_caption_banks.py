#!/usr/bin/env python3
"""Build deterministic nested class-caption banks from existing caption metadata."""

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from common import atomic_write_json, sha256_file


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", required=True)
    parser.add_argument("--caption-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--budgets", type=int, nargs="+", default=(4, 8, 16, 32))
    parser.add_argument("--bank-seeds", type=int, nargs="+", default=(0, 1))
    return parser.parse_args()


def class_seed(seed, synset):
    digest = hashlib.sha256(f"{int(seed)}:{synset}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


def read_caption_records(train_root, caption_file):
    train_root = Path(train_root).resolve()
    images = [
        path for path in train_root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    ]
    by_basename = defaultdict(list)
    for path in images:
        by_basename[path.name].append(path)
    rows, seen = [], set()
    with Path(caption_file).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            relative = str(item.get("file_name", "")).replace("\\", "/")
            caption = str(item.get("text", "")).strip()
            path = train_root / relative
            if relative and not path.is_file() and len(Path(relative).parts) == 1:
                matches = by_basename.get(relative, [])
                if len(matches) == 1:
                    path = matches[0]
                    relative = path.relative_to(train_root).as_posix()
            if not relative or not caption or not path.is_file():
                raise ValueError(f"Invalid caption row {line_number}: {relative}")
            if relative in seen:
                raise ValueError(f"Duplicate caption metadata path: {relative}")
            seen.add(relative)
            rows.append({
                "relative": relative,
                "synset": Path(relative).parts[0],
                "caption": caption,
            })
    expected = {path.relative_to(train_root).as_posix() for path in images}
    if expected != seen:
        raise RuntimeError(
            f"Caption/image mismatch: {len(expected - seen)} missing, {len(seen - expected)} unknown"
        )
    return rows


def build_banks(train_root, caption_file, output_dir, budgets, bank_seeds):
    budgets = sorted(set(map(int, budgets)))
    if not budgets or budgets[0] < 2:
        raise ValueError("Sparse-bank budgets must be unique integers >= 2")
    rows = read_caption_records(train_root, caption_file)
    by_class = defaultdict(list)
    for row in rows:
        by_class[row["synset"]].append(row)
    if min(map(len, by_class.values())) < budgets[-1]:
        raise ValueError(f"At least one class has fewer than {budgets[-1]} images")
    output_dir = Path(output_dir).resolve()
    outputs = []
    for seed in sorted(set(map(int, bank_seeds))):
        ranked = {}
        for synset in sorted(by_class):
            candidates = sorted(by_class[synset], key=lambda row: row["relative"])
            random.Random(class_seed(seed, synset)).shuffle(candidates)
            ranked[synset] = candidates
        for budget in budgets:
            payload = {
                "format_version": 1,
                "selection": "uniform_without_replacement_nested_prefix",
                "bank_seed": seed,
                "budget_per_class": budget,
                "maximum_nested_budget": budgets[-1],
                "caption_file": str(Path(caption_file).resolve()),
                "caption_file_sha256": sha256_file(caption_file),
                "classes": {
                    synset: [
                        {
                            "relative": row["relative"],
                            "caption": row["caption"],
                            "nested_rank": rank,
                        }
                        for rank, row in enumerate(ranked[synset][:budget])
                    ]
                    for synset in sorted(ranked)
                },
            }
            path = output_dir / f"bank_seed_{seed}" / f"m_{budget}.json"
            if path.is_file():
                existing = json.loads(path.read_text(encoding="utf-8"))
                if existing != payload:
                    raise RuntimeError(f"Sparse bank differs from existing artifact: {path}")
            else:
                atomic_write_json(path, payload)
            outputs.append(path)
    atomic_write_json(output_dir / "bank_index.json", {
        "format_version": 1,
        "budgets": budgets,
        "bank_seeds": sorted(set(map(int, bank_seeds))),
        "banks": [str(path) for path in outputs],
    })
    return outputs


def main():
    args = parse_args()
    outputs = build_banks(
        args.train_root, args.caption_file, args.output_dir, args.budgets, args.bank_seeds
    )
    print(f"Prepared {len(outputs)} deterministic nested sparse caption banks")


if __name__ == "__main__":
    main()
