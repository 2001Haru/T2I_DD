import hashlib
import json
import os
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def parse_shift_runs(items):
    runs = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected SHIFT=RUN_ROOT, got {item!r}")
        shift_text, root_text = item.split("=", 1)
        shift = int(shift_text)
        if not 1 <= shift <= 9:
            raise ValueError(f"Shuffle shift must be in [1, 9], got {shift}")
        if shift in runs:
            raise ValueError(f"Duplicate shuffle shift: {shift}")
        runs[shift] = Path(root_text).resolve()
    return runs


def seed_directories(run_root):
    synthetic_root = Path(run_root) / "synthetic"
    directories = sorted(
        synthetic_root.glob("seed_*"), key=lambda path: int(path.name.split("_")[-1])
    )
    if not directories:
        raise FileNotFoundError(f"No seed directories under {synthetic_root}")
    return directories


def load_consistent_manifest(run_root, condition):
    manifests = []
    for seed_dir in seed_directories(run_root):
        path = seed_dir / condition / "manifest.json"
        completion = seed_dir / condition / "complete.json"
        if not path.is_file() or not completion.is_file():
            raise FileNotFoundError(f"Incomplete condition: {seed_dir / condition}")
        manifests.append((int(seed_dir.name.split("_")[-1]), load_json(path), path))

    reference = [
        (
            record["synset"],
            int(record["image_index"]),
            int(record["prototype_index"]),
            record["prompt_source_index"],
            record["prompt"],
        )
        for record in manifests[0][1]["prompt_records"]
    ]
    for generation_seed, manifest, path in manifests[1:]:
        current = [
            (
                record["synset"],
                int(record["image_index"]),
                int(record["prototype_index"]),
                record["prompt_source_index"],
                record["prompt"],
            )
            for record in manifest["prompt_records"]
        ]
        if current != reference:
            raise RuntimeError(
                f"Prompt assignment changes across generation seeds in {path} "
                f"(seed={generation_seed})"
            )
    return manifests[0][1], [item[0] for item in manifests]


def image_paths(root):
    paths = sorted(
        path
        for path in Path(root).rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not paths:
        raise FileNotFoundError(f"No images under {root}")
    return paths


def file_inventory_signature(paths, extra=None):
    digest = hashlib.sha256()
    if extra is not None:
        digest.update(json.dumps(extra, sort_keys=True).encode("utf-8"))
    for path in paths:
        path = Path(path)
        stat = path.stat()
        digest.update(str(path.resolve()).encode("utf-8"))
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
    return digest.hexdigest()
