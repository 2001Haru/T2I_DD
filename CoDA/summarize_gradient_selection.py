"""Summarize gradient-selected datasets and their downstream evaluation."""

import argparse
import csv
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


METHODS = ("real", "diffusion", "gm_selected", "random_matched")


def _load_json(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Required result was not found: {path}")
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _load_csv(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Required diagnostic was not found: {path}")
    with open(path, "r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


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


def _class_means(payload):
    return {row["class_id"]: float(row["mean"]) for row in payload["class_summary"]}


def _correlation(x, y):
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return {"spearman_rho": None, "spearman_pvalue": None}
    result = stats.spearmanr(x, y)
    return {
        "spearman_rho": float(result.statistic),
        "spearman_pvalue": float(result.pvalue),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--input", nargs=6, action="append", required=True,
        metavar=("SPEC", "DIAGNOSTIC_DIR", "REAL_RESULT", "DIFFUSION_RESULT", "GM_RESULT", "RANDOM_RESULT"),
    )
    args = parser.parse_args()
    if os.path.exists(args.output_dir):
        raise FileExistsError(f"Refusing to overwrite gradient-selection summary: {args.output_dir}")
    os.makedirs(args.output_dir)

    summary = {"methods": list(METHODS), "subsets": {}, "cross_subset": {}}
    per_class_rows = []
    for spec, diagnostic_dir, real_path, diffusion_path, gm_path, random_path in args.input:
        paths = {
            "real": real_path,
            "diffusion": diffusion_path,
            "gm_selected": gm_path,
            "random_matched": random_path,
        }
        results = {method: _load_json(path) for method, path in paths.items()}
        diagnostic_rows = _load_csv(os.path.join(diagnostic_dir, "class_diagnostics.csv"))
        diagnostics_by_id = {row["class_id"]: row for row in diagnostic_rows}
        means_by_method = {method: _class_means(payload) for method, payload in results.items()}

        method_summary = {}
        for method, payload in results.items():
            scores = [float(value) for value in payload["overall_top1"]]
            method_summary[method] = {
                "accuracies": scores,
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
                "result_path": paths[method],
            }
        best_endpoint = max(method_summary["real"]["mean"], method_summary["diffusion"]["mean"])
        summary["subsets"][spec] = {
            "methods": method_summary,
            "gm_minus_random": method_summary["gm_selected"]["mean"] - method_summary["random_matched"]["mean"],
            "gm_minus_real": method_summary["gm_selected"]["mean"] - method_summary["real"]["mean"],
            "gm_minus_diffusion": method_summary["gm_selected"]["mean"] - method_summary["diffusion"]["mean"],
            "gm_minus_best_endpoint": method_summary["gm_selected"]["mean"] - best_endpoint,
        }

        for local_label, class_id in enumerate(means_by_method["real"]):
            diagnostic = diagnostics_by_id[class_id]
            row = {
                "spec": spec,
                "local_label": local_label,
                "class_id": class_id,
                "class_name": diagnostic["class_name"],
                "set_score_diffusion_vs_real": float(diagnostic["set_score_diffusion_vs_real"]),
                "mean_pair_delta_g": float(diagnostic["mean_pair_delta_g"]),
                "gm_selected_diffusion_count": int(diagnostic["gm_selected_diffusion_count"]),
            }
            for method in METHODS:
                row[f"{method}_mean"] = means_by_method[method][class_id]
            row["diffusion_minus_real"] = row["diffusion_mean"] - row["real_mean"]
            row["gm_minus_random"] = row["gm_selected_mean"] - row["random_matched_mean"]
            row["gm_minus_best_endpoint"] = row["gm_selected_mean"] - max(
                row["real_mean"], row["diffusion_mean"]
            )
            per_class_rows.append(row)

    for method in METHODS:
        subset_means = {
            spec: values["methods"][method]["mean"]
            for spec, values in summary["subsets"].items()
        }
        summary["cross_subset"][method] = {
            "subset_means": subset_means,
            "mean_across_subsets": float(np.mean(list(subset_means.values()))),
        }

    set_scores = [row["set_score_diffusion_vs_real"] for row in per_class_rows]
    pair_scores = [row["mean_pair_delta_g"] for row in per_class_rows]
    endpoint_gains = [row["diffusion_minus_real"] for row in per_class_rows]
    summary["cross_subset"]["diagnostic_relationship"] = {
        "set_score_vs_endpoint_gain": _correlation(set_scores, endpoint_gains),
        "mean_pair_score_vs_endpoint_gain": _correlation(pair_scores, endpoint_gains),
    }

    _write_json(os.path.join(args.output_dir, "experiment_summary.json"), summary)
    _write_csv(os.path.join(args.output_dir, "per_class_comparison.csv"), per_class_rows)

    specs = list(summary["subsets"])
    x = np.arange(len(specs))
    width = 0.2
    figure, axis = plt.subplots(figsize=(9, 5))
    for method_index, method in enumerate(METHODS):
        values = [summary["subsets"][spec]["methods"][method]["mean"] for spec in specs]
        axis.bar(x + (method_index - 1.5) * width, values, width, label=method)
    axis.set_xticks(x, specs)
    axis.set_ylabel("Mean validation accuracy")
    axis.set_title("Gradient-guided candidate selection")
    axis.legend()
    figure.tight_layout()
    figure.savefig(os.path.join(args.output_dir, "downstream_accuracy_comparison.png"), dpi=180)
    plt.close(figure)
    print(f"Saved gradient-selection summary to: {args.output_dir}")


if __name__ == "__main__":
    main()
