#!/usr/bin/env python3
"""Summarize fixed-m4 replication and matched Label-inference controls."""

import argparse
import ast
import csv
import json
import random
import re
import statistics
from pathlib import Path

import matplotlib.pyplot as plt


RESULT = re.compile(r"Best, last acc:----(\[[^\]]+\])")
CONTROLS = ("empty_ft", "label_ft", "unpaired_ft")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sparse-index", required=True)
    parser.add_argument("--old-nette-sparse-index", default="")
    parser.add_argument("--control-index", action="append", default=[])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    return parser.parse_args()


def read_scores(path):
    matches = RESULT.findall(Path(path).read_text(encoding="utf-8", errors="replace"))
    if not matches:
        raise ValueError(f"No completed classifier result: {path}")
    return [float(value) for value in ast.literal_eval(matches[-1])]


def visual_strength(row):
    value = row.get("strength")
    if value is None:
        token = str(row.get("visual", ""))
        if token.startswith("strength_"):
            value = token.removeprefix("strength_").replace("p", ".")
    return None if value is None else round(float(value), 6)


def eligible(row, spec):
    return (
        row.get("spec", "nette") == spec
        and int(row.get("ipc", 50)) == 50
        and row.get("prompt") == "label"
        and row.get("visual_mode", "prototype") == "prototype"
        and visual_strength(row) == 0.8
    )


def load_rows(args):
    rows = []
    sparse_indexes = [(args.sparse_index, False)]
    if args.old_nette_sparse_index:
        sparse_indexes.append((args.old_nette_sparse_index, True))
    for path, old in sparse_indexes:
        for row in json.loads(Path(path).read_text(encoding="utf-8")):
            if int(row.get("budget", 4)) != 4 or int(row.get("bank_seed", 0)) != 0:
                continue
            if row.get("prompt") != "label" or int(row.get("ipc", 50)) != 50:
                continue
            if visual_strength(row) != 0.8:
                continue
            spec = row.get("spec", "nette")
            record = {
                "spec": spec, "method": "sparse_m4", "training_seed": int(row["training_seed"]),
                "generation_seed": int(row["generation_seed"]), "scores": read_scores(row["evaluation_log"]),
                "source": "old_sparse" if old else "new_sparse",
            }
            rows.append(record)
    for path in args.control_index:
        for row in json.loads(Path(path).read_text(encoding="utf-8")):
            supervision = row.get("supervision")
            if supervision not in CONTROLS:
                continue
            spec = row.get("spec", "nette")
            if not eligible(row, spec) or row.get("training_seed") is None:
                continue
            rows.append({
                "spec": spec, "method": supervision, "training_seed": int(row["training_seed"]),
                "generation_seed": int(row["generation_seed"]), "scores": read_scores(row["evaluation_log"]),
                "source": str(path),
            })
    deduplicated = {}
    for row in rows:
        key = (row["spec"], row["method"], row["training_seed"], row["generation_seed"])
        deduplicated[key] = row
    return list(deduplicated.values())


def percentile(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))]


def group(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["training_seed"], {})[row["generation_seed"]] = row["scores"]
    return grouped


def bootstrap_mean(rows, samples, rng):
    grouped = group(rows)
    seeds = sorted(grouped)
    estimates = []
    for _ in range(samples):
        values = []
        for _ in seeds:
            training_seed = rng.choice(seeds)
            generations = grouped[training_seed]
            generation_seeds = sorted(generations)
            for _ in generation_seeds:
                scores = generations[rng.choice(generation_seeds)]
                values.extend(rng.choice(scores) for _ in scores)
        estimates.append(statistics.fmean(values))
    return percentile(estimates, .025), percentile(estimates, .975)


def paired_contrast(left_rows, right_rows, samples, rng):
    left, right = group(left_rows), group(right_rows)
    training_seeds = sorted(set(left) & set(right))
    if not training_seeds:
        return None
    cells = []
    for training_seed in training_seeds:
        generations = sorted(set(left[training_seed]) & set(right[training_seed]))
        for generation_seed in generations:
            a, b = left[training_seed][generation_seed], right[training_seed][generation_seed]
            if len(a) != len(b):
                raise ValueError(f"Classifier-repeat mismatch: seed {training_seed}/{generation_seed}")
            cells.append((training_seed, generation_seed, [x - y for x, y in zip(a, b)]))
    by_training = {}
    for training_seed, generation_seed, values in cells:
        by_training.setdefault(training_seed, {})[generation_seed] = values
    estimates = []
    for _ in range(samples):
        values = []
        for _ in training_seeds:
            training_seed = rng.choice(training_seeds)
            generations = by_training[training_seed]
            generation_seeds = sorted(generations)
            for _ in generation_seeds:
                differences = generations[rng.choice(generation_seeds)]
                values.extend(rng.choice(differences) for _ in differences)
        estimates.append(statistics.fmean(values))
    point = statistics.fmean(value for _, _, values in cells for value in values)
    return point, percentile(estimates, .025), percentile(estimates, .975), len(training_seeds), len(cells)


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args)
    performance, contrasts = [], []
    for spec in ("nette", "woof"):
        methods = sorted({row["method"] for row in rows if row["spec"] == spec})
        for method in methods:
            selected = [row for row in rows if row["spec"] == spec and row["method"] == method]
            values = [value for row in selected for value in row["scores"]]
            low, high = bootstrap_mean(selected, args.bootstrap_samples, random.Random(100 + len(performance)))
            performance.append({
                "spec": spec, "method": method, "mean_accuracy": statistics.fmean(values),
                "bootstrap_ci_lower": low, "bootstrap_ci_upper": high,
                "training_seeds": len({row["training_seed"] for row in selected}),
                "training_generation_cells": len(selected), "classifier_observations": len(values),
            })
        sparse = [row for row in rows if row["spec"] == spec and row["method"] == "sparse_m4"]
        for control in CONTROLS:
            other = [row for row in rows if row["spec"] == spec and row["method"] == control]
            if not sparse or not other:
                continue
            result = paired_contrast(sparse, other, args.bootstrap_samples, random.Random(900 + len(contrasts)))
            if result is None:
                continue
            mean, low, high, training_seeds, cells = result
            contrasts.append({
                "spec": spec, "contrast": f"sparse_m4_minus_{control}_label", "mean_difference": mean,
                "bootstrap_ci_lower": low, "bootstrap_ci_upper": high,
                "paired_training_seeds": training_seeds, "paired_training_generation_cells": cells,
                "bootstrap_order": "paired training seed -> paired generation seed -> paired classifier repeat",
            })
    write_csv(output / "performance.csv", performance, performance[0].keys())
    if contrasts:
        write_csv(output / "paired_contrasts.csv", contrasts, contrasts[0].keys())
    summary = {
        "format_version": 1,
        "estimand": "fixed bank seed 0, m=4, IPC50, prototype strength 0.8, Label inference",
        "performance": performance, "paired_contrasts": contrasts,
        "interpretation_boundary": "Sparse performance uses every available checkpoint; contrasts use only shared training and generation seeds.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    methods = ("empty_ft", "label_ft", "unpaired_ft", "sparse_m4")
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
    for axis, spec in zip(axes, ("nette", "woof")):
        selected = [row for method in methods for row in performance if row["spec"] == spec and row["method"] == method]
        if not selected:
            axis.set_visible(False)
            continue
        x = range(len(selected))
        means = [row["mean_accuracy"] for row in selected]
        error = [[mean - row["bootstrap_ci_lower"] for mean, row in zip(means, selected)],
                 [row["bootstrap_ci_upper"] - mean for mean, row in zip(means, selected)]]
        axis.errorbar(x, means, yerr=error, fmt="o", capsize=4)
        axis.set_xticks(list(x), [row["method"] for row in selected], rotation=20)
        axis.set_title(spec)
        axis.set_ylabel("Validation accuracy")
        axis.grid(alpha=.25)
    figure.suptitle("Fixed-m4 sparse caption supervision generality")
    figure.tight_layout()
    figure.savefig(output / "summary.png", dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
