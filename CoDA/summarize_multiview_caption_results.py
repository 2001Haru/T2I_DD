"""Summarize baseline, single-image caption, and neighbor-montage caption runs."""

import argparse
import csv
import json
import os

import numpy as np


METHODS = ("coda_baseline", "single_focused", "montage_common_mode")


def _load_result(root, generation_seed, method):
    path = os.path.join(
        root, f"seed_{generation_seed}", f"{method}-resnet_ap",
        "per_class_accuracy_all_seeds.json",
    )
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing completed classifier result: {path}")
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--trained_root", required=True)
    parser.add_argument("--generation_seeds", nargs="+", type=int, default=[0, 1])
    args = parser.parse_args()

    results = {
        method: {
            str(seed): _load_result(args.trained_root, seed, method)
            for seed in args.generation_seeds
        }
        for method in METHODS
    }
    summary = {"generation_seeds": args.generation_seeds, "methods": {}, "paired_differences": {}}
    for method in METHODS:
        by_seed = {}
        all_scores = []
        for seed in args.generation_seeds:
            scores = results[method][str(seed)]["overall_top1"]
            by_seed[str(seed)] = {"classifier_accuracies": scores, "mean": float(np.mean(scores))}
            all_scores.extend(scores)
        summary["methods"][method] = {
            "by_generation_seed": by_seed,
            "mean": float(np.mean(all_scores)),
            "std": float(np.std(all_scores, ddof=1)) if len(all_scores) > 1 else 0.0,
        }

    for label, left, right in (
        ("single_minus_baseline", "single_focused", "coda_baseline"),
        ("montage_minus_single", "montage_common_mode", "single_focused"),
        ("montage_minus_baseline", "montage_common_mode", "coda_baseline"),
    ):
        differences = {
            str(seed): (
                summary["methods"][left]["by_generation_seed"][str(seed)]["mean"]
                - summary["methods"][right]["by_generation_seed"][str(seed)]["mean"]
            )
            for seed in args.generation_seeds
        }
        summary["paired_differences"][label] = {
            "by_generation_seed": differences,
            "mean": float(np.mean(list(differences.values()))),
        }

    first = results[METHODS[0]][str(args.generation_seeds[0])]
    rows = []
    for class_index, class_info in enumerate(first["class_summary"]):
        row = {
            "local_label": class_info["local_label"],
            "class_id": class_info["class_id"],
            "class_name": class_info["class_name"],
        }
        for method in METHODS:
            scores = [
                run["classes"][class_index]["accuracy"]
                for seed in args.generation_seeds
                for run in results[method][str(seed)]["runs"]
            ]
            row[f"{method}_mean"] = float(np.mean(scores))
        row["montage_minus_single"] = row["montage_common_mode_mean"] - row["single_focused_mean"]
        row["montage_minus_baseline"] = row["montage_common_mode_mean"] - row["coda_baseline_mean"]
        rows.append(row)

    output_dir = os.path.join(args.run_dir, "summary")
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "experiment_summary.json"), "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
        file.write("\n")
    with open(os.path.join(output_dir, "per_class_comparison.csv"), "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved multiview caption experiment summary to: {output_dir}")


if __name__ == "__main__":
    main()
