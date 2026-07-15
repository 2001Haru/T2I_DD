"""Summarize the cross-subset final prompt/refinement control experiment."""

import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np


METHODS = (
    "real_representative",
    "vae_reconstruction",
    "empty_prompt",
    "generic_prompt",
    "class_prompt",
    "montage_caption",
)


def _load(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Classifier result was not found: {path}")
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--result", nargs=3, action="append", metavar=("SPEC", "METHOD", "PATH"), required=True,
        help="One per_class_accuracy_all_seeds.json input.",
    )
    args = parser.parse_args()

    paths = defaultdict(dict)
    for spec, method, path in args.result:
        if method not in METHODS:
            parser.error(f"Unknown method {method!r}; expected one of {', '.join(METHODS)}")
        if method in paths[spec]:
            parser.error(f"Duplicate result for {spec}/{method}")
        paths[spec][method] = path

    for spec, methods in paths.items():
        missing = [method for method in METHODS if method not in methods]
        if missing:
            parser.error(f"Missing methods for {spec}: {', '.join(missing)}")

    loaded = {
        spec: {method: _load(methods[method]) for method in METHODS}
        for spec, methods in paths.items()
    }
    summary = {"methods": list(METHODS), "subsets": {}, "cross_subset": {}}
    per_class_rows = []

    for spec, methods in loaded.items():
        spec_summary = {"methods": {}, "deltas_vs_class_prompt": {}}
        for method, result in methods.items():
            scores = result["overall_top1"]
            spec_summary["methods"][method] = {
                "classifier_accuracies": scores,
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
                "result_path": paths[spec][method],
            }
        baseline = spec_summary["methods"]["class_prompt"]["mean"]
        for method in METHODS:
            spec_summary["deltas_vs_class_prompt"][method] = (
                spec_summary["methods"][method]["mean"] - baseline
            )
        summary["subsets"][spec] = spec_summary

        first = methods[METHODS[0]]
        for class_index, class_info in enumerate(first["class_summary"]):
            row = {
                "spec": spec,
                "local_label": class_info["local_label"],
                "class_id": class_info["class_id"],
                "class_name": class_info["class_name"],
            }
            for method in METHODS:
                values = [
                    run["classes"][class_index]["accuracy"]
                    for run in methods[method]["runs"]
                ]
                row[f"{method}_mean"] = float(np.mean(values))
            for method in METHODS:
                row[f"{method}_minus_class_prompt"] = (
                    row[f"{method}_mean"] - row["class_prompt_mean"]
                )
            per_class_rows.append(row)

    for method in METHODS:
        subset_means = {
            spec: summary["subsets"][spec]["methods"][method]["mean"]
            for spec in loaded
        }
        deltas = {
            spec: summary["subsets"][spec]["deltas_vs_class_prompt"][method]
            for spec in loaded
        }
        summary["cross_subset"][method] = {
            "subset_means": subset_means,
            "mean_across_subsets": float(np.mean(list(subset_means.values()))),
            "deltas_vs_class_prompt": deltas,
            "mean_delta_vs_class_prompt": float(np.mean(list(deltas.values()))),
        }

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "experiment_summary.json"), "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
        file.write("\n")
    with open(os.path.join(args.output_dir, "per_class_comparison.csv"), "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(per_class_rows[0]))
        writer.writeheader()
        writer.writerows(per_class_rows)
    print(f"Saved final prompt-control summary to: {args.output_dir}")


if __name__ == "__main__":
    main()
