"""Assemble complete IPC datasets from P4/P5 subsets and neutral fillers."""

import argparse
import json
import os
import shutil
from pathlib import Path


REGIMES = ("i0g0", "i1g0", "i0g1", "i1g1")
PROMPTS = ("label", "correct", "shuffled")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--filler-manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--specs", nargs="+", required=True)
    parser.add_argument("--generation-seeds", nargs="+", type=int, required=True)
    parser.add_argument("--ipc", type=int, default=10)
    return parser.parse_args()


def load_manifest(path, key_fields):
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    result = {}
    for row in rows:
        key = tuple(row[field] for field in key_fields)
        if key in result:
            raise ValueError(f"Duplicate manifest key {key} in {path}")
        result[key] = Path(row["dataset_dir"])
    return result


def link_or_copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def class_images(root):
    return {
        class_dir.name: {
            int(path.stem): path
            for path in class_dir.glob("*.png")
            if path.stem.isdigit()
        }
        for class_dir in sorted(root.glob("n????????"))
        if class_dir.is_dir()
    }


def assemble(source, filler, destination, ipc):
    source_images = class_images(source)
    filler_images = class_images(filler)
    if not source_images:
        raise ValueError(f"No ImageNet class directories in {source}")
    expected = set(range(ipc))
    details = {}
    for class_id, images in source_images.items():
        unexpected = set(images) - expected
        if unexpected:
            raise ValueError(f"Out-of-range images in {source}/{class_id}: {unexpected}")
        missing = sorted(expected - set(images))
        available_fillers = filler_images.get(class_id, {})
        unavailable = [index for index in missing if index not in available_fillers]
        if unavailable:
            raise ValueError(
                f"Missing neutral fillers for {class_id} indices {unavailable}: {filler}"
            )
        for index in sorted(images):
            link_or_copy(images[index], destination / class_id / f"{index}.png")
        for index in missing:
            link_or_copy(available_fillers[index], destination / class_id / f"{index}.png")
        details[class_id] = {
            "source_images": len(images),
            "neutral_filler_indices": missing,
            "final_images": ipc,
        }
    extra_filler_classes = set(filler_images) - set(source_images)
    if extra_filler_classes:
        raise ValueError(f"Filler has unexpected classes: {sorted(extra_filler_classes)}")
    return details


def main():
    args = parse_args()
    source = load_manifest(
        args.source_manifest,
        ("spec", "generation_seed", "visual_mode", "prompt_condition"),
    )
    filler = load_manifest(
        args.filler_manifest, ("spec", "generation_seed", "visual_mode")
    )
    output_root = Path(args.output_root)
    expected_source = {
        (spec, seed, regime, prompt)
        for spec in args.specs
        for seed in args.generation_seeds
        for regime in REGIMES
        for prompt in PROMPTS
    }
    missing_source = expected_source - set(source)
    if missing_source:
        raise ValueError(f"Source manifest lacks cells: {sorted(missing_source)}")

    output_manifest = []
    audit = []
    for spec, seed, regime, prompt in sorted(expected_source):
        filler_key = (spec, seed, regime)
        if filler_key not in filler:
            raise ValueError(f"Filler manifest lacks {filler_key}")
        destination = output_root / spec / f"seed_{seed}" / f"{regime}_{prompt}"
        if destination.exists():
            raise FileExistsError(
                f"Refusing to merge into existing P6 dataset: {destination}"
            )
        details = assemble(source[(spec, seed, regime, prompt)], filler[filler_key], destination, args.ipc)
        output_manifest.append(
            {
                "spec": spec,
                "generation_seed": seed,
                "visual_mode": regime,
                "prompt_condition": prompt,
                "dataset_dir": str(destination),
            }
        )
        audit.append(
            {
                "spec": spec,
                "generation_seed": seed,
                "visual_mode": regime,
                "prompt_condition": prompt,
                "source_dataset_dir": str(source[(spec, seed, regime, prompt)]),
                "filler_dataset_dir": str(filler[filler_key]),
                "classes": details,
            }
        )

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "dataset_manifest.json").write_text(
        json.dumps(output_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_root / "assembly_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Assembled {len(output_manifest)} complete P6 datasets in {output_root}")


if __name__ == "__main__":
    main()
