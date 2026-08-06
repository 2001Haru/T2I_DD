import hashlib
import json
import os
import random
from pathlib import Path


SUPERVISION_MODES = (
    "frozen",
    "empty_ft",
    "constant_ft",
    "label_ft",
    "unpaired_ft",
    "matched_ft",
)
PROMPT_MODES = ("label", "correct", "shuffled")


def condition_name(supervision_mode, prompt_mode):
    if supervision_mode not in SUPERVISION_MODES:
        raise ValueError(f"Unknown supervision mode: {supervision_mode}")
    if prompt_mode not in PROMPT_MODES:
        raise ValueError(f"Unknown prompt mode: {prompt_mode}")
    return f"{supervision_mode}_{prompt_mode}"


def condition_matrix():
    return [
        {
            "supervision_mode": supervision,
            "prompt_mode": prompt,
            "condition": condition_name(supervision, prompt),
        }
        for supervision in SUPERVISION_MODES
        for prompt in PROMPT_MODES
    ]


def shuffled_prompt_index(index, count, shift=1):
    if count < 2:
        raise ValueError("Shuffled DCS requires at least two prompts")
    if not 0 < shift < count:
        raise ValueError(f"shuffle shift must be in [1, {count - 1}], got {shift}")
    return (index + shift) % count


def stable_image_seed(generation_seed, class_index, image_index):
    return int(generation_seed) * 1_000_000 + int(class_index) * 10_000 + int(image_index)


def build_unpaired_donors(class_indices, seed, epoch):
    """Build a deterministic within-class derangement preserving caption marginals."""
    total = sum(len(indices) for indices in class_indices.values())
    donors = list(range(total))
    flattened = sorted(index for indices in class_indices.values() for index in indices)
    if flattened != list(range(total)):
        raise ValueError("class_indices must partition contiguous dataset indices")
    for class_offset, class_key in enumerate(sorted(class_indices)):
        indices = list(class_indices[class_key])
        if len(indices) < 2:
            raise ValueError(f"Unpaired supervision needs at least two images in {class_key}")
        rng = random.Random(int(seed) + int(epoch) * 1_000_003 + class_offset * 10_007)
        rng.shuffle(indices)
        shift = rng.randrange(1, len(indices))
        for position, image_index in enumerate(indices):
            donors[image_index] = indices[(position + shift) % len(indices)]
    if any(index == donor for index, donor in enumerate(donors)):
        raise AssertionError("Unpaired assignment contains a self-pair")
    return donors


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def ensure_manifest(output_dir, payload, resume=False):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    canonical = json.loads(json.dumps(payload, sort_keys=True))
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != canonical:
            raise RuntimeError(f"Resume configuration differs from {manifest_path}")
        if not resume:
            raise RuntimeError(f"Output already exists: {output_dir}; pass --resume")
        return manifest_path
    if any(output_dir.iterdir()):
        raise RuntimeError(f"Non-empty output has no manifest: {output_dir}")
    atomic_write_json(manifest_path, canonical)
    return manifest_path
