"""Measure whether independently best classes remain best when combined."""

import argparse
import csv
import json
import os

import matplotlib.pyplot as plt
import numpy as np


def _load_json(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Required oracle artifact was not found: {path}")
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


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


def _method_summary(payload, path):
    scores = [float(value) for value in payload["overall_top1"]]
    return {
        "accuracies": scores,
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
        "result_path": os.path.abspath(path),
    }


def _class_means(payload):
    return {row["class_id"]: float(row["mean"]) for row in payload["class_summary"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--input", nargs=5, action="append", required=True,
        metavar=("SPEC", "MANIFEST", "REAL_RESULT", "DIFFUSION_RESULT", "ORACLE_RESULT"),
    )
    args = parser.parse_args()
    if os.path.exists(args.output_dir):
        raise FileExistsError(f"Refusing to overwrite class-oracle summary: {args.output_dir}")
    os.makedirs(args.output_dir)

    summary = {"diagnostic_only": True, "subsets": {}, "cross_subset": {}}
    per_class_rows = []
    for spec, manifest_path, real_path, diffusion_path, oracle_path in args.input:
        manifest = _load_json(manifest_path)
        real = _load_json(real_path)
        diffusion = _load_json(diffusion_path)
        oracle = _load_json(oracle_path)
        methods = {
            "real": _method_summary(real, real_path),
            "diffusion": _method_summary(diffusion, diffusion_path),
            "class_oracle": _method_summary(oracle, oracle_path),
        }
        expected = float(manifest["expected_independent_oracle_accuracy"])
        best_endpoint = max(methods["real"]["mean"], methods["diffusion"]["mean"])
        interaction_gap = methods["class_oracle"]["mean"] - expected
        summary["subsets"][spec] = {
            "methods": methods,
            "selected_class_counts": manifest["selected_class_counts"],
            "expected_independent_oracle_accuracy": expected,
            "actual_oracle_accuracy": methods["class_oracle"]["mean"],
            "interaction_gap": interaction_gap,
            "oracle_minus_best_endpoint": methods["class_oracle"]["mean"] - best_endpoint,
        }

        real_means = _class_means(real)
        diffusion_means = _class_means(diffusion)
        oracle_means = _class_means(oracle)
        for row in manifest["classes"]:
            class_id = row["class_id"]
            expected_class = max(real_means[class_id], diffusion_means[class_id])
            per_class_rows.append({
                "spec": spec,
                "local_label": row["local_label"],
                "class_id": class_id,
                "class_name": row["class_name"],
                "selected_source": row["selected_source"],
                "real_mean": real_means[class_id],
                "diffusion_mean": diffusion_means[class_id],
                "expected_selected_mean": expected_class,
                "oracle_hybrid_mean": oracle_means[class_id],
                "class_interaction_shift": oracle_means[class_id] - expected_class,
            })

    subset_values = list(summary["subsets"].values())
    summary["cross_subset"] = {
        "mean_expected_independent_oracle_accuracy": float(np.mean([
            row["expected_independent_oracle_accuracy"] for row in subset_values
        ])),
        "mean_actual_oracle_accuracy": float(np.mean([
            row["actual_oracle_accuracy"] for row in subset_values
        ])),
        "mean_interaction_gap": float(np.mean([
            row["interaction_gap"] for row in subset_values
        ])),
        "mean_oracle_minus_best_endpoint": float(np.mean([
            row["oracle_minus_best_endpoint"] for row in subset_values
        ])),
    }
    _write_json(os.path.join(args.output_dir, "experiment_summary.json"), summary)
    _write_csv(os.path.join(args.output_dir, "per_class_interactions.csv"), per_class_rows)

    specs = list(summary["subsets"])
    x = np.arange(len(specs))
    width = 0.2
    series = (
        ("real", [summary["subsets"][spec]["methods"]["real"]["mean"] for spec in specs]),
        ("diffusion", [summary["subsets"][spec]["methods"]["diffusion"]["mean"] for spec in specs]),
        ("expected oracle", [summary["subsets"][spec]["expected_independent_oracle_accuracy"] for spec in specs]),
        ("actual oracle", [summary["subsets"][spec]["actual_oracle_accuracy"] for spec in specs]),
    )
    figure, axis = plt.subplots(figsize=(9, 5))
    for index, (label, values) in enumerate(series):
        axis.bar(x + (index - 1.5) * width, values, width, label=label)
    axis.set_xticks(x, specs)
    axis.set_ylabel("Mean validation accuracy")
    axis.set_title("Independent class oracle vs combined dataset")
    axis.legend()
    figure.tight_layout()
    figure.savefig(os.path.join(args.output_dir, "class_oracle_interaction_gap.png"), dpi=180)
    plt.close(figure)
    print(f"Saved class-oracle summary to: {args.output_dir}")


if __name__ == "__main__":
    main()
