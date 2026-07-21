"""Summarize cross-fitted class-oracle results with paired classifier seeds."""

import argparse
import csv
import glob
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


METHODS = ("real", "diffusion", "class_oracle")


def _load_json(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Required paired-oracle artifact was not found: {path}")
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _write_json(path, payload):
    temporary = f"{path}.tmp"
    with open(temporary, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")
    os.replace(temporary, path)


def _write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _mean_std(values):
    values = [float(value) for value in values]
    return {
        "values": values,
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
    }


def _selection_seeds(manifest):
    seeds = set()
    for key in ("real_result", "diffusion_result"):
        payload = _load_json(manifest[key])
        seeds.update(int(seed) for seed in payload.get("training_seeds", []))
    return seeds


def _load_method_runs(results_root, spec, method):
    pattern = os.path.join(
        results_root, spec, method, "seed_start_*", "resnet_ap",
        "per_class_accuracy_all_seeds.json",
    )
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No paired results matched: {pattern}")

    runs = {}
    for path in paths:
        payload = _load_json(path)
        for run in payload.get("runs", []):
            seed = int(run["training_seed"])
            if seed in runs:
                raise ValueError(f"Duplicate {spec}/{method} training seed {seed}: {path}")
            runs[seed] = {
                "overall_top1": float(run["overall_top1"]),
                "classes": {
                    row["class_id"]: float(row["accuracy"])
                    for row in run["classes"]
                },
                "result_path": os.path.abspath(path),
            }
    return runs


def _summarize_spec(spec, manifest_path, results_root):
    manifest = _load_json(manifest_path)
    selected = {
        row["class_id"]: {
            "local_label": int(row["local_label"]),
            "class_name": row.get("class_name", row["class_id"]),
            "source": row["selected_source"],
        }
        for row in manifest["classes"]
    }
    runs = {
        method: _load_method_runs(results_root, spec, method)
        for method in METHODS
    }
    seed_sets = [set(method_runs) for method_runs in runs.values()]
    if not all(seed_set == seed_sets[0] for seed_set in seed_sets[1:]):
        raise ValueError(f"Paired methods for {spec} do not contain identical training seeds.")
    evaluation_seeds = sorted(seed_sets[0])
    selection_seeds = sorted(_selection_seeds(manifest))
    overlap = sorted(set(evaluation_seeds) & set(selection_seeds))
    if overlap:
        raise ValueError(
            f"Evaluation seeds overlap selection seeds for {spec}: {overlap}. "
            "Use new EVAL_SEED_STARTS."
        )

    seed_rows = []
    class_rows = []
    for seed in evaluation_seeds:
        expected_selected = []
        for class_id, info in selected.items():
            source_run = runs[info["source"]][seed]
            oracle_run = runs["class_oracle"][seed]
            selected_accuracy = source_run["classes"][class_id]
            oracle_accuracy = oracle_run["classes"][class_id]
            expected_selected.append(selected_accuracy)
            class_rows.append({
                "spec": spec,
                "training_seed": seed,
                "local_label": info["local_label"],
                "class_id": class_id,
                "class_name": info["class_name"],
                "selected_source": info["source"],
                "selected_endpoint_accuracy": selected_accuracy,
                "oracle_hybrid_accuracy": oracle_accuracy,
                "paired_interaction_shift": oracle_accuracy - selected_accuracy,
            })

        real_accuracy = runs["real"][seed]["overall_top1"]
        diffusion_accuracy = runs["diffusion"][seed]["overall_top1"]
        oracle_accuracy = runs["class_oracle"][seed]["overall_top1"]
        expected_accuracy = float(np.mean(expected_selected))
        seed_rows.append({
            "spec": spec,
            "training_seed": seed,
            "real_accuracy": real_accuracy,
            "diffusion_accuracy": diffusion_accuracy,
            "paired_expected_selected_accuracy": expected_accuracy,
            "oracle_hybrid_accuracy": oracle_accuracy,
            "paired_interaction_gap": oracle_accuracy - expected_accuracy,
            "oracle_minus_best_endpoint": (
                oracle_accuracy - max(real_accuracy, diffusion_accuracy)
            ),
        })

    class_summary = []
    grouped = defaultdict(list)
    for row in class_rows:
        grouped[row["class_id"]].append(row)
    for class_id, rows in sorted(
        grouped.items(), key=lambda item: item[1][0]["local_label"]
    ):
        shifts = [row["paired_interaction_shift"] for row in rows]
        endpoints = [row["selected_endpoint_accuracy"] for row in rows]
        hybrids = [row["oracle_hybrid_accuracy"] for row in rows]
        first = rows[0]
        class_summary.append({
            "spec": spec,
            "local_label": first["local_label"],
            "class_id": class_id,
            "class_name": first["class_name"],
            "selected_source": first["selected_source"],
            "selected_endpoint_mean": float(np.mean(endpoints)),
            "oracle_hybrid_mean": float(np.mean(hybrids)),
            "paired_shift_mean": float(np.mean(shifts)),
            "paired_shift_std": (
                float(np.std(shifts, ddof=1)) if len(shifts) > 1 else 0.0
            ),
            "negative_seed_fraction": float(np.mean(np.asarray(shifts) < 0)),
        })

    return {
        "spec": spec,
        "manifest_path": os.path.abspath(manifest_path),
        "selection_training_seeds": selection_seeds,
        "evaluation_training_seeds": evaluation_seeds,
        "seed_rows": seed_rows,
        "class_rows": class_rows,
        "class_summary": class_summary,
        "summary": {
            "real": _mean_std([row["real_accuracy"] for row in seed_rows]),
            "diffusion": _mean_std([row["diffusion_accuracy"] for row in seed_rows]),
            "paired_expected_selected": _mean_std([
                row["paired_expected_selected_accuracy"] for row in seed_rows
            ]),
            "class_oracle": _mean_std([
                row["oracle_hybrid_accuracy"] for row in seed_rows
            ]),
            "paired_interaction_gap": _mean_std([
                row["paired_interaction_gap"] for row in seed_rows
            ]),
            "oracle_minus_best_endpoint": _mean_std([
                row["oracle_minus_best_endpoint"] for row in seed_rows
            ]),
        },
    }


def summarize(inputs, output_dir):
    if os.path.exists(output_dir):
        raise FileExistsError(f"Refusing to overwrite paired-oracle summary: {output_dir}")
    os.makedirs(output_dir)

    results = [
        _summarize_spec(spec, manifest_path, results_root)
        for spec, manifest_path, results_root in inputs
    ]
    seed_rows = [row for result in results for row in result["seed_rows"]]
    class_rows = [row for result in results for row in result["class_rows"]]
    class_summary = [row for result in results for row in result["class_summary"]]
    payload = {
        "diagnostic_only": True,
        "protocol": (
            "Class choices are fixed by earlier selection seeds. Real, Diffusion, "
            "and hybrid datasets are evaluated with identical disjoint classifier seeds."
        ),
        "subsets": {
            result["spec"]: {
                "manifest_path": result["manifest_path"],
                "selection_training_seeds": result["selection_training_seeds"],
                "evaluation_training_seeds": result["evaluation_training_seeds"],
                "summary": result["summary"],
            }
            for result in results
        },
        "cross_subset_mean_interaction_gap": float(np.mean([
            result["summary"]["paired_interaction_gap"]["mean"]
            for result in results
        ])),
    }
    _write_json(os.path.join(output_dir, "paired_experiment_summary.json"), payload)
    _write_csv(os.path.join(output_dir, "paired_seed_results.csv"), seed_rows)
    _write_csv(os.path.join(output_dir, "paired_class_results_all_seeds.csv"), class_rows)
    _write_csv(os.path.join(output_dir, "paired_class_summary.csv"), class_summary)

    figure, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 4), squeeze=False)
    for axis, result in zip(axes[0], results):
        rows = result["seed_rows"]
        seeds = [row["training_seed"] for row in rows]
        gaps = [row["paired_interaction_gap"] for row in rows]
        colors = ["tab:red" if gap < 0 else "tab:green" for gap in gaps]
        axis.bar([str(seed) for seed in seeds], gaps, color=colors)
        axis.axhline(0.0, color="black", linewidth=1)
        axis.set_title(f"{result['spec']}: paired interaction gap")
        axis.set_xlabel("Classifier training seed")
        axis.set_ylabel("Hybrid - selected endpoints")
    figure.tight_layout()
    figure.savefig(os.path.join(output_dir, "paired_interaction_gap_by_seed.png"), dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(
        1, len(results), figsize=(6.5 * len(results), 6), squeeze=False, sharex=True,
    )
    for axis, result in zip(axes[0], results):
        rows = result["class_summary"]
        labels = [row["class_name"].split(",", 1)[0] for row in rows]
        means = [row["paired_shift_mean"] for row in rows]
        errors = [row["paired_shift_std"] for row in rows]
        positions = np.arange(len(labels))
        colors = ["tab:red" if value < 0 else "tab:green" for value in means]
        axis.barh(positions, means, xerr=errors, color=colors, alpha=0.9)
        axis.axvline(0.0, color="black", linewidth=1)
        axis.set_yticks(positions, labels)
        axis.invert_yaxis()
        axis.set_xlabel("Mean paired interaction shift")
        axis.set_title(result["spec"])
    figure.suptitle("Fixed class choices under disjoint paired evaluation seeds")
    figure.tight_layout()
    figure.savefig(os.path.join(output_dir, "paired_class_interaction_shifts.png"), dpi=180)
    plt.close(figure)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--input", nargs=3, action="append", required=True,
        metavar=("SPEC", "ORACLE_MANIFEST", "RESULTS_ROOT"),
    )
    args = parser.parse_args()
    payload = summarize(args.input, args.output_dir)
    for spec, result in payload["subsets"].items():
        gap = result["summary"]["paired_interaction_gap"]
        print(
            f"{spec}: paired interaction gap "
            f"{gap['mean']:.2f} +/- {gap['std']:.2f} over "
            f"{len(result['evaluation_training_seeds'])} seeds"
        )
    print(f"Saved paired class-oracle summary to: {args.output_dir}")


if __name__ == "__main__":
    main()
