#!/usr/bin/env python3
"""Audit why the new a_pad77 result differs from the historical strength table."""

import argparse
import ast
import csv
import hashlib
import json
import re
import statistics
from pathlib import Path


RESULT = re.compile(r"Best, last acc:----(\[[^\]]+\])")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current-index", required=True)
    parser.add_argument("--historical-index", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def scores(path):
    matches = RESULT.findall(Path(path).read_text(encoding="utf-8", errors="replace"))
    if not matches:
        raise ValueError(f"No classifier result: {path}")
    return [float(value) for value in ast.literal_eval(matches[-1])]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def is_current(row):
    return row.get("prompt") == "label" and row.get("checkpoint_family") == "matched_ft"


def is_historical(row):
    return (
        row.get("prompt") == "label"
        and row.get("supervision") == "matched_ft"
        and row.get("spec") == "nette"
        and int(row.get("ipc", -1)) == 50
        and abs(float(row.get("strength", -1)) - 0.8) < 1e-9
    )


def synthetic_dir(row):
    path = row.get("synthetic_dir")
    if not path:
        raise ValueError(f"Index row lacks synthetic_dir: {row}")
    return Path(path).resolve()


def record_map(root):
    records = json.loads((root / "prompt_records.json").read_text(encoding="utf-8"))
    return {(row["synset"], int(row["image_index"])): row for row in records}


def comparable_manifest(root):
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    keys = (
        "checkpoint", "prototype_sha256", "dcs_sha256", "generation_seed", "ipc",
        "strength", "visual_mode", "guidance_scale", "num_inference_steps",
        "negative_prompt", "supervision_mode", "prompt_mode",
    )
    return {key: manifest.get(key) for key in keys}


def compare_generation(current_row, historical_row):
    current_root = synthetic_dir(current_row)
    historical_root = synthetic_dir(historical_row)
    current_records = record_map(current_root)
    historical_records = record_map(historical_root)
    keys = sorted(set(current_records) & set(historical_records))
    png_equal = 0
    record_equal = 0
    record_fields = (
        "prompt", "embedding_policy", "conditioning_chunks",
        "conditioning_sequence_length", "image_seed", "prototype_index",
    )
    for synset, image_index in keys:
        current_record = current_records[(synset, image_index)]
        historical_record = historical_records[(synset, image_index)]
        record_equal += all(
            current_record.get(field) == historical_record.get(field)
            for field in record_fields
        )
        relative = Path(synset) / f"image_{image_index:05d}.png"
        png_equal += sha256(current_root / relative) == sha256(historical_root / relative)
    current_manifest = comparable_manifest(current_root)
    historical_manifest = comparable_manifest(historical_root)
    differing_manifest_fields = sorted(
        key for key in current_manifest if current_manifest[key] != historical_manifest[key]
    )
    current_scores = scores(current_row["evaluation_log"])
    historical_scores = scores(historical_row["evaluation_log"])
    return {
        "generation_seed": int(current_row["generation_seed"]),
        "shared_images": len(keys),
        "prompt_record_equal_fraction": record_equal / len(keys),
        "png_sha256_equal_fraction": png_equal / len(keys),
        "manifest_exact_match": not differing_manifest_fields,
        "manifest_differences": ",".join(differing_manifest_fields),
        "current_mean_accuracy": statistics.fmean(current_scores),
        "historical_mean_accuracy": statistics.fmean(historical_scores),
        "current_minus_historical": statistics.fmean(current_scores) - statistics.fmean(historical_scores),
    }


def aggregate(name, rows):
    values = [value for row in rows for value in scores(row["evaluation_log"])]
    return {
        "scope": name,
        "training_seeds": ",".join(str(value) for value in sorted({int(row.get("training_seed", 0)) for row in rows})),
        "generation_seeds": ",".join(str(value) for value in sorted({int(row["generation_seed"]) for row in rows})),
        "cells": len(rows),
        "classifier_observations": len(values),
        "mean_accuracy": statistics.fmean(values),
    }


def main():
    args = parse_args()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    current = [row for row in json.loads(Path(args.current_index).read_text(encoding="utf-8")) if is_current(row)]
    historical = [row for row in json.loads(Path(args.historical_index).read_text(encoding="utf-8")) if is_historical(row)]
    if not current or not historical:
        raise RuntimeError(f"Missing comparison rows: current={len(current)}, historical={len(historical)}")

    overlap_seeds = sorted(
        {int(row["generation_seed"]) for row in current}
        & {int(row["generation_seed"]) for row in historical if int(row.get("training_seed", 0)) == 0}
    )
    current_overlap = [row for row in current if int(row["generation_seed"]) in overlap_seeds]
    historical_t0_overlap = [
        row for row in historical
        if int(row.get("training_seed", 0)) == 0 and int(row["generation_seed"]) in overlap_seeds
    ]
    aggregates = [
        aggregate("current_ft0_all_generation_seeds", current),
        aggregate("current_ft0_overlap_generation_seeds", current_overlap),
        aggregate("historical_ft0_overlap_generation_seeds", historical_t0_overlap),
        aggregate("historical_all_ft_and_generation_seeds", historical),
    ]

    current_by_seed = {int(row["generation_seed"]): row for row in current_overlap}
    historical_by_seed = {int(row["generation_seed"]): row for row in historical_t0_overlap}
    generation_audit = [
        compare_generation(current_by_seed[seed], historical_by_seed[seed])
        for seed in overlap_seeds
    ]
    write_csv(output / "aggregate_estimands.csv", aggregates)
    write_csv(output / "overlap_generation_audit.csv", generation_audit)

    all_png_equal = all(row["png_sha256_equal_fraction"] == 1.0 for row in generation_audit)
    all_records_equal = all(row["prompt_record_equal_fraction"] == 1.0 for row in generation_audit)
    if all_png_equal:
        decision = "generation_reproduced_exactly; aggregate discrepancy comes from estimand composition and/or classifier evaluation"
    elif all_records_equal:
        decision = "conditioning records match but PNGs differ; inspect diffusers/runtime provenance or model/artifact hashes"
    else:
        decision = "generation protocol differs in the overlapping cells; inspect manifest_differences and prompt records"
    payload = {
        "format_version": 1,
        "decision": decision,
        "critical_estimand_difference": (
            "Historical 78.75 pools training seeds 0/1 and generation seeds 0/1. "
            "Current 77.87 uses only training seed 0 and generation seeds 0/1/2. "
            "Their marginal confidence intervals are therefore not directly comparable."
        ),
        "aggregates": aggregates,
        "overlap_generation_audit": generation_audit,
        "runtime_provenance_boundary": (
            "Legacy manifests do not guarantee that the historical run recorded its imported "
            "diffusers module path. A prompt-record match with a PNG mismatch points to this "
            "unrecorded runtime/code dimension."
        ),
    }
    (output / "reproducibility_audit.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# a_pad77 reproducibility audit", "", f"Decision: **{decision}**", "",
        "The historical 78.75 and current 77.87 are different estimands: the former pools two "
        "fine-tuning seeds, while the latter contains only fine-tuning seed 0.", "",
        "## Aggregate decomposition", "", "| Scope | FT seeds | Gen seeds | Mean |", "|---|---|---|---:|",
    ]
    for row in aggregates:
        lines.append(f"| {row['scope']} | {row['training_seeds']} | {row['generation_seeds']} | {row['mean_accuracy']:.3f} |")
    lines.extend(["", "## Exact overlapping-cell audit", "", "| Gen seed | Record equality | PNG equality | Manifest differences |", "|---:|---:|---:|---|"])
    for row in generation_audit:
        lines.append(
            f"| {row['generation_seed']} | {row['prompt_record_equal_fraction']:.3f} | "
            f"{row['png_sha256_equal_fraction']:.3f} | {row['manifest_differences'] or 'none'} |"
        )
    (output / "REPRODUCIBILITY_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
