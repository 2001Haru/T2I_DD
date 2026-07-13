"""Validate a fixed ImageNet subset before running a distillation experiment."""

import argparse
from collections import Counter
import json
import os

from data import create_imagenet_dataset


def _class_name_map():
    root = os.path.join(os.path.dirname(__file__), "misc")
    with open(os.path.join(root, "class_indices.txt"), "r", encoding="utf-8") as file:
        class_ids = [line.strip() for line in file if line.strip()]
    with open(os.path.join(root, "class_names.txt"), "r", encoding="utf-8") as file:
        class_names = [line.strip() for line in file]
    return dict(zip(class_ids, class_names))


def _missing_paths(dataset):
    return [path for path, _ in dataset.samples if not os.path.isfile(path)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_dir", required=True)
    parser.add_argument("--val_dir", required=True)
    parser.add_argument("--spec", default="imageB")
    parser.add_argument("--nclass", type=int, default=10)
    parser.add_argument("--min_train_per_class", type=int, default=10)
    parser.add_argument("--expected_val_per_class", type=int, default=50)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    train = create_imagenet_dataset(
        args.train_dir, nclass=args.nclass, spec=args.spec, seed=0, load_memory=False
    )
    val = create_imagenet_dataset(
        args.val_dir, nclass=args.nclass, spec=args.spec, seed=0, load_memory=False
    )
    if train.classes != val.classes:
        raise ValueError("Train and validation subsets use different class orderings.")

    train_counts = Counter(train.targets)
    val_counts = Counter(val.targets)
    missing_train = _missing_paths(train)
    missing_val = _missing_paths(val)
    failures = []
    name_by_id = _class_name_map()
    rows = []
    for index, class_id in enumerate(train.classes):
        train_count = train_counts[index]
        val_count = val_counts[index]
        if train_count < args.min_train_per_class:
            failures.append(f"{class_id}: only {train_count} training images")
        if val_count != args.expected_val_per_class:
            failures.append(
                f"{class_id}: expected {args.expected_val_per_class} validation images, found {val_count}"
            )
        rows.append({
            "local_label": index,
            "class_id": class_id,
            "class_name": name_by_id.get(class_id, class_id),
            "train_count": train_count,
            "validation_count": val_count,
        })
    if missing_train:
        failures.append(f"{len(missing_train)} training paths are missing")
    if missing_val:
        failures.append(f"{len(missing_val)} validation paths are missing")

    payload = {
        "spec": args.spec,
        "class_selection_seed": 0,
        "train_dir": os.path.abspath(args.train_dir),
        "val_dir": os.path.abspath(args.val_dir),
        "status": "failed" if failures else "passed",
        "classes": rows,
        "missing_train_paths": missing_train[:20],
        "missing_validation_paths": missing_val[:20],
        "failures": failures,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")

    for row in rows:
        print(
            f"{row['local_label']:2d} {row['class_id']} "
            f"train={row['train_count']:4d} val={row['validation_count']:2d} "
            f"{row['class_name']}"
        )
    if failures:
        raise RuntimeError("ImageNet subset validation failed: " + "; ".join(failures))
    print(f"ImageNet subset validation passed; report: {args.output}")


if __name__ == "__main__":
    main()
