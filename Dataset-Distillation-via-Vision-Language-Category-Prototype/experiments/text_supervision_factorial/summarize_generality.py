#!/usr/bin/env python3
"""Summarize endpoint, IPC scaling, and ImageWoof replication cells."""

import argparse
import ast
import csv
import json
import random
import re
import statistics
from pathlib import Path


RESULT = re.compile(r"Best, last acc:----(\[[^\]]+\])")


def read_scores(path):
    matches = RESULT.findall(Path(path).read_text(encoding="utf-8", errors="replace"))
    if not matches:
        raise ValueError(f"No completed classifier result: {path}")
    return [float(value) for value in ast.literal_eval(matches[-1])]


def paired(left, right):
    if len(left) != len(right):
        raise ValueError("Classifier repeat counts differ")
    return [a - b for a, b in zip(left, right)]


def hierarchical_bootstrap(rows, samples=10000, seed=20260807):
    """Resample training seeds, generation seeds, then paired classifier repeats."""
    grouped = {}
    for row in rows:
        grouped.setdefault(row["training_seed"], {}).setdefault(row["generation_seed"], []).append(row["values"])
    rng = random.Random(seed)
    estimates = []
    training_seeds = list(grouped)
    for _ in range(samples):
        draw = []
        for _ in training_seeds:
            training_seed = rng.choice(training_seeds)
            generation = grouped[training_seed]
            generation_seeds = list(generation)
            for _ in generation_seeds:
                generation_seed = rng.choice(generation_seeds)
                candidates = generation[generation_seed]
                values = rng.choice(candidates)
                draw.extend(rng.choice(values) for _ in values)
        estimates.append(statistics.fmean(draw))
    estimates.sort()
    return estimates[int(0.025 * samples)], estimates[min(samples - 1, int(0.975 * samples))]


def summarize_rows(rows, value_key="values"):
    normalized = [{**row, "values": row[value_key]} for row in rows]
    flat = [value for row in normalized for value in row["values"]]
    lower, upper = hierarchical_bootstrap(normalized)
    return {
        "mean": statistics.fmean(flat),
        "hierarchical_bootstrap_ci_lower": lower,
        "hierarchical_bootstrap_ci_upper": upper,
        "training_generation_cells": len(normalized),
        "observations": len(flat),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-index", required=True, nargs="+")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    index = []
    for index_path in args.evaluation_index:
        index.extend(json.loads(Path(index_path).read_text(encoding="utf-8")))
    cells = []
    lookup = {}
    for item in index:
        scores = read_scores(item["evaluation_log"])
        row = {
            **item,
            "mean_accuracy": statistics.fmean(scores),
            "std_accuracy": statistics.pstdev(scores),
            "classifier_accuracies": scores,
        }
        cells.append(row)
        key = (item["spec"], item["ipc"], item["training_seed"], item["generation_seed"], item["supervision"], item["prompt"])
        if key in lookup:
            raise RuntimeError(f"Duplicate evaluation cell: {key}")
        lookup[key] = scores

    performance = []
    for key in sorted({(row["spec"], row["ipc"], row["supervision"], row["prompt"]) for row in cells}):
        spec, ipc, supervision, prompt = key
        selected = [
            {"training_seed": row["training_seed"], "generation_seed": row["generation_seed"], "values": row["classifier_accuracies"]}
            for row in cells
            if (row["spec"], row["ipc"], row["supervision"], row["prompt"]) == key
        ]
        performance.append({"spec": spec, "ipc": ipc, "supervision": supervision, "prompt": prompt, **summarize_rows(selected)})

    contrast_specs = (
        ("endpoint_matched_correct_minus_empty_label", ("matched_ft", "correct"), ("empty_ft", "label"), False),
        ("empty_minus_frozen_label", ("empty_ft", "label"), ("frozen", "label"), True),
        ("matched_correct_minus_shuffled", ("matched_ft", "correct"), ("matched_ft", "shuffled"), False),
        ("unpaired_correct_minus_shuffled", ("unpaired_ft", "correct"), ("unpaired_ft", "shuffled"), False),
        ("matched_minus_unpaired_correct", ("matched_ft", "correct"), ("unpaired_ft", "correct"), False),
    )
    contrasts = []
    scopes = sorted({(row["spec"], row["ipc"]) for row in cells})
    train_gen = sorted({(row["spec"], row["ipc"], row["training_seed"], row["generation_seed"]) for row in cells}, key=str)
    for spec, ipc in scopes:
        for name, left, right, right_shared in contrast_specs:
            rows = []
            for current_spec, current_ipc, training_seed, generation_seed in train_gen:
                if (current_spec, current_ipc) != (spec, ipc) or training_seed is None:
                    continue
                left_key = (spec, ipc, training_seed, generation_seed, *left)
                right_key = (spec, ipc, None if right_shared else training_seed, generation_seed, *right)
                if left_key in lookup and right_key in lookup:
                    rows.append({
                        "training_seed": training_seed, "generation_seed": generation_seed,
                        "values": paired(lookup[left_key], lookup[right_key]),
                    })
            if rows:
                contrasts.append({"spec": spec, "ipc": ipc, "contrast": name, **summarize_rows(rows)})

    payload = {
        "format_version": 1,
        "bootstrap_order": "training_seed -> generation_seed -> paired classifier repeat",
        "performance": performance,
        "contrasts": contrasts,
        "interpretation_boundary": (
            "ImageNette IPC scaling defaults to training seeds 0/1; IPC10 seed extension uses seeds 0-3. "
            "ImageWoof is a separate dataset replication and should not be pooled with ImageNette."
        ),
    }
    (output / "generality_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(output / "cells.csv", cells, (
        "spec", "ipc", "training_seed", "generation_seed", "supervision", "prompt", "mean_accuracy",
        "std_accuracy", "classifier_accuracies", "source", "evaluation_log",
    ))
    write_csv(output / "performance.csv", performance, (
        "spec", "ipc", "supervision", "prompt", "mean", "hierarchical_bootstrap_ci_lower",
        "hierarchical_bootstrap_ci_upper", "training_generation_cells", "observations",
    ))
    write_csv(output / "contrasts.csv", contrasts, (
        "spec", "ipc", "contrast", "mean", "hierarchical_bootstrap_ci_lower",
        "hierarchical_bootstrap_ci_upper", "training_generation_cells", "observations",
    ))
    plot_summary(performance, contrasts, output / "generality_summary.png")
    print(json.dumps(contrasts, indent=2, sort_keys=True))


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def plot_summary(performance, contrasts, destination):
    import matplotlib.pyplot as plt
    import numpy as np

    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    policies = (("empty_ft", "label", "Empty+Label"), ("matched_ft", "correct", "Matched+Correct"))
    scopes = sorted({(row["spec"], row["ipc"]) for row in performance})
    x = np.arange(len(scopes))
    width = 0.36
    for offset, (supervision, prompt, label) in zip((-width / 2, width / 2), policies):
        values = []
        for spec, ipc in scopes:
            match = next((row for row in performance if row["spec"] == spec and row["ipc"] == ipc and row["supervision"] == supervision and row["prompt"] == prompt), None)
            values.append(match["mean"] if match else np.nan)
        axes[0].bar(x + offset, values, width, label=label)
    axes[0].set_xticks(x, [f"{spec}\nIPC {ipc}" for spec, ipc in scopes])
    axes[0].set_ylabel("Validation accuracy")
    axes[0].set_title("Endpoint policies across scale and dataset")
    axes[0].legend()

    endpoint = [row for row in contrasts if row["contrast"] == "endpoint_matched_correct_minus_empty_label"]
    means = [row["mean"] for row in endpoint]
    lower = [row["mean"] - row["hierarchical_bootstrap_ci_lower"] for row in endpoint]
    upper = [row["hierarchical_bootstrap_ci_upper"] - row["mean"] for row in endpoint]
    axes[1].bar(range(len(endpoint)), means, yerr=np.array([lower, upper]), capsize=4)
    axes[1].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[1].set_xticks(range(len(endpoint)), [f"{row['spec']}\nIPC {row['ipc']}" for row in endpoint])
    axes[1].set_ylabel("Matched+Correct minus Empty+Label")
    axes[1].set_title("Caption-intensive policy premium")
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(destination, dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
