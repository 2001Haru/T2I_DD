"""Summarize the fixed ImageB baseline, focused, and projection matrix."""

import argparse
import csv
import json
import os

import numpy as np


METHODS = ("coda_baseline", "v1_focused_alpha_0", "v1_focused_alpha_0p5")


def _load_result(trained_root, generation_seed, method):
    path = os.path.join(
        trained_root,
        f"seed_{generation_seed}",
        f"{method}-resnet_ap",
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
    method_summary = {}
    for method in METHODS:
        by_seed = {}
        all_accuracies = []
        for seed in args.generation_seeds:
            accuracies = results[method][str(seed)]["overall_top1"]
            by_seed[str(seed)] = {
                "classifier_accuracies": accuracies,
                "mean": float(np.mean(accuracies)),
                "std": float(np.std(accuracies, ddof=1)) if len(accuracies) > 1 else 0.0,
            }
            all_accuracies.extend(accuracies)
        method_summary[method] = {
            "by_generation_seed": by_seed,
            "all_classifier_runs_mean": float(np.mean(all_accuracies)),
            "all_classifier_runs_std": (
                float(np.std(all_accuracies, ddof=1)) if len(all_accuracies) > 1 else 0.0
            ),
        }

    paired = {}
    for label, left, right in (
        ("focused_minus_baseline", "v1_focused_alpha_0", "coda_baseline"),
        ("alpha_0p5_minus_alpha_0", "v1_focused_alpha_0p5", "v1_focused_alpha_0"),
        ("alpha_0p5_minus_baseline", "v1_focused_alpha_0p5", "coda_baseline"),
    ):
        differences = {
            str(seed): (
                method_summary[left]["by_generation_seed"][str(seed)]["mean"]
                - method_summary[right]["by_generation_seed"][str(seed)]["mean"]
            )
            for seed in args.generation_seeds
        }
        paired[label] = {
            "by_generation_seed": differences,
            "mean": float(np.mean(list(differences.values()))),
        }

    summary_dir = os.path.join(args.run_dir, "summary")
    os.makedirs(summary_dir, exist_ok=True)
    summary = {
        "generation_seeds": args.generation_seeds,
        "methods": method_summary,
        "paired_mean_differences": paired,
    }
    with open(os.path.join(summary_dir, "experiment_summary.json"), "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
        file.write("\n")

    rows = []
    first_result = results[METHODS[0]][str(args.generation_seeds[0])]
    for class_index, class_info in enumerate(first_result["class_summary"]):
        row = {
            "local_label": class_info["local_label"],
            "class_id": class_info["class_id"],
            "class_name": class_info["class_name"],
        }
        for method in METHODS:
            values = []
            for seed in args.generation_seeds:
                values.extend(
                    run["classes"][class_index]["accuracy"]
                    for run in results[method][str(seed)]["runs"]
                )
            row[f"{method}_mean"] = float(np.mean(values))
        row["alpha_0p5_minus_alpha_0"] = (
            row["v1_focused_alpha_0p5_mean"] - row["v1_focused_alpha_0_mean"]
        )
        row["alpha_0p5_minus_baseline"] = (
            row["v1_focused_alpha_0p5_mean"] - row["coda_baseline_mean"]
        )
        rows.append(row)

    csv_path = os.path.join(summary_dir, "per_class_comparison.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved ImageB experiment summary to: {summary_dir}")


if __name__ == "__main__":
    main()
