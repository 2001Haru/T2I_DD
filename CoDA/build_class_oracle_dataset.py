"""Build a diagnostic class-wise oracle from real and Diffusion endpoint results."""

import argparse
import csv
import json
import os
import shutil
from datetime import datetime, timezone

import numpy as np


def _load_result(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Classifier result was not found: {path}")
    with open(path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    if "class_summary" not in payload or "overall_top1" not in payload:
        raise ValueError(f"Invalid per-class classifier result: {path}")
    return payload


def _class_rows(payload):
    return {row["class_id"]: row for row in payload["class_summary"]}


def _indexed_images(root, class_id, ipc):
    paths = []
    for index in range(ipc):
        path = os.path.join(root, class_id, f"{index}.png")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Oracle candidate image was not found: {path}")
        paths.append(os.path.abspath(path))
    return paths


def _write_json(path, payload):
    temporary = f"{path}.tmp"
    with open(temporary, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")
    os.replace(temporary, path)


def _write_csv(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_oracle(args):
    if os.path.exists(args.output_dir):
        raise FileExistsError(f"Refusing to overwrite class-oracle dataset: {args.output_dir}")
    real_result = _load_result(args.real_result)
    diffusion_result = _load_result(args.diffusion_result)
    real_rows = _class_rows(real_result)
    diffusion_rows = _class_rows(diffusion_result)
    if set(real_rows) != set(diffusion_rows):
        raise ValueError("Real and Diffusion results contain different class IDs.")

    os.makedirs(args.output_dir)
    selection_rows = []
    selected_counts = {"real": 0, "diffusion": 0}
    for local_label, real_info in enumerate(real_result["class_summary"]):
        class_id = real_info["class_id"]
        diffusion_info = diffusion_rows[class_id]
        if int(real_info["local_label"]) != int(diffusion_info["local_label"]):
            raise ValueError(f"Local-label mismatch for class {class_id}.")
        real_mean = float(real_info["mean"])
        diffusion_mean = float(diffusion_info["mean"])
        if diffusion_mean > real_mean:
            selected_source = "diffusion"
        elif real_mean > diffusion_mean:
            selected_source = "real"
        else:
            selected_source = args.tie_policy
        selected_counts[selected_source] += 1
        source_root = args.real_dir if selected_source == "real" else args.diffusion_dir
        source_paths = _indexed_images(source_root, class_id, args.ipc)
        destination_dir = os.path.join(args.output_dir, class_id)
        os.makedirs(destination_dir)
        for index, source_path in enumerate(source_paths):
            shutil.copy2(source_path, os.path.join(destination_dir, f"{index}.png"))
        selection_rows.append({
            "spec": args.spec,
            "local_label": local_label,
            "class_id": class_id,
            "class_name": real_info.get("class_name", class_id),
            "real_mean": real_mean,
            "diffusion_mean": diffusion_mean,
            "diffusion_minus_real": diffusion_mean - real_mean,
            "selected_source": selected_source,
            "selected_endpoint_mean": max(real_mean, diffusion_mean),
            "source_dir": os.path.abspath(source_root),
            "destination_dir": os.path.abspath(destination_dir),
        })

    expected_oracle = float(np.mean([
        row["selected_endpoint_mean"] for row in selection_rows
    ]))
    metadata = {
        "format_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "diagnostic_only": True,
        "warning": (
            "This class-wise oracle uses downstream validation accuracy to choose candidates "
            "and is not a deployable selection method."
        ),
        "spec": args.spec,
        "ipc": args.ipc,
        "tie_policy": args.tie_policy,
        "real_dir": os.path.abspath(args.real_dir),
        "diffusion_dir": os.path.abspath(args.diffusion_dir),
        "real_result": os.path.abspath(args.real_result),
        "diffusion_result": os.path.abspath(args.diffusion_result),
        "selected_class_counts": selected_counts,
        "expected_independent_oracle_accuracy": expected_oracle,
        "classes": selection_rows,
    }
    _write_csv(os.path.join(args.output_dir, "oracle_selection.csv"), selection_rows)
    _write_json(os.path.join(args.output_dir, "oracle_manifest.json"), metadata)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--real_dir", required=True)
    parser.add_argument("--diffusion_dir", required=True)
    parser.add_argument("--real_result", required=True)
    parser.add_argument("--diffusion_result", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--ipc", type=int, default=10)
    parser.add_argument("--tie_policy", choices=("real", "diffusion"), default="real")
    args = parser.parse_args()
    if args.ipc < 1:
        parser.error("--ipc must be positive.")
    metadata = build_oracle(args)
    print(
        f"Built {args.spec} class oracle at {args.output_dir}: "
        f"real classes={metadata['selected_class_counts']['real']}, "
        f"diffusion classes={metadata['selected_class_counts']['diffusion']}, "
        f"expected independent accuracy="
        f"{metadata['expected_independent_oracle_accuracy']:.2f}"
    )


if __name__ == "__main__":
    main()
