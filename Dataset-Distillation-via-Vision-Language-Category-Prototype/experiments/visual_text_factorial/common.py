import hashlib
import json
import os
from pathlib import Path


VISUAL_MODES = ("no_visual", "prototype")
PROMPT_MODES = ("label", "dcs", "dcs_shuffled")


def condition_name(visual_mode, prompt_mode):
    if visual_mode not in VISUAL_MODES:
        raise ValueError(f"Unknown visual mode: {visual_mode}")
    if prompt_mode not in PROMPT_MODES:
        raise ValueError(f"Unknown prompt mode: {prompt_mode}")
    return f"{visual_mode}_{prompt_mode}"


def condition_matrix():
    return [
        {
            "visual_mode": visual_mode,
            "prompt_mode": prompt_mode,
            "condition": condition_name(visual_mode, prompt_mode),
        }
        for visual_mode in VISUAL_MODES
        for prompt_mode in PROMPT_MODES
    ]


def shuffled_prompt_index(prototype_index, count, shift=1):
    if count < 2:
        raise ValueError("A shuffled DCS control requires at least two prototypes")
    if shift <= 0 or shift >= count:
        raise ValueError(f"Shuffle shift must be in [1, {count - 1}], got {shift}")
    return (prototype_index + shift) % count


def stable_image_seed(generation_seed, class_index, image_index):
    if generation_seed < 0 or class_index < 0 or image_index < 0:
        raise ValueError("Seed inputs must be non-negative")
    return int(generation_seed) * 1_000_000 + int(class_index) * 10_000 + int(image_index)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_manifest(payload):
    return json.loads(json.dumps(payload, sort_keys=True))


def ensure_manifest(output_dir, payload, resume=False):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    payload = canonical_manifest(payload)

    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as handle:
            existing = canonical_manifest(json.load(handle))
        if existing != payload:
            raise RuntimeError(
                f"Refusing to mix incompatible runs in {output_dir}. "
                "Use a new run id or move the incompatible directory."
            )
        if not resume:
            raise RuntimeError(
                f"Output already exists at {output_dir}. Pass --resume to reuse matching files."
            )
        return manifest_path

    if any(output_dir.iterdir()):
        raise RuntimeError(
            f"Non-empty output has no manifest: {output_dir}. "
            "Refusing to guess whether it belongs to this run."
        )

    temporary_path = manifest_path.with_suffix(".json.tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary_path, manifest_path)
    return manifest_path
