#!/usr/bin/env python3
"""Summarize the paired Sparse-m4/Matched-FT interface transfer experiment."""

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
PROMPTS = ("label", "bank", "shuffled", "correct")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-index", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--equivalence-margin", type=float, default=1.0)
    return parser.parse_args()


def scores(path):
    matches = RESULT.findall(Path(path).read_text(encoding="utf-8", errors="replace"))
    if not matches:
        raise ValueError(f"No completed classifier result: {path}")
    return [float(value) for value in ast.literal_eval(matches[-1])]


def percentile(values, probability):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(probability * len(ordered)))]


def bootstrap(rows, value_key, samples, seed):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["training_seed"], {})[row["generation_seed"]] = row[value_key]
    training_seeds = sorted(grouped)
    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        values = []
        for _ in training_seeds:
            train_seed = rng.choice(training_seeds)
            generations = grouped[train_seed]
            generation_seeds = sorted(generations)
            for _ in generation_seeds:
                generation_seed = rng.choice(generation_seeds)
                repeats = generations[generation_seed]
                values.extend(rng.choice(repeats) for _ in repeats)
        estimates.append(statistics.fmean(values))
    return percentile(estimates, 0.025), percentile(estimates, 0.975), percentile(estimates, 0.05), percentile(estimates, 0.95)


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    raw = json.loads(Path(args.evaluation_index).read_text(encoding="utf-8"))
    cells, lookup = [], {}
    for row in raw:
        values = scores(row["evaluation_log"])
        cell = {**row, "scores": values, "mean_accuracy": statistics.fmean(values)}
        cells.append(cell)
        lookup[(row["checkpoint_family"], row["training_seed"], row["generation_seed"], row["prompt"])] = values
    expected = {
        (family, training_seed, generation_seed, prompt)
        for family in ("sparse_m4_ft", "matched_ft")
        for training_seed in sorted({row["training_seed"] for row in cells})
        for generation_seed in sorted({row["generation_seed"] for row in cells})
        for prompt in PROMPTS
    }
    missing = sorted(expected - set(lookup))
    if missing:
        raise RuntimeError(f"Incomplete 2x4 interface matrix; missing {missing}")

    performance = []
    for family in ("sparse_m4_ft", "matched_ft"):
        for prompt in PROMPTS:
            selected = [row for row in cells if row["checkpoint_family"] == family and row["prompt"] == prompt]
            lower, upper, _, _ = bootstrap(selected, "scores", args.bootstrap_samples, 20260820)
            values = [value for row in selected for value in row["scores"]]
            performance.append({
                "checkpoint_family": family, "prompt": prompt,
                "mean_accuracy": statistics.fmean(values),
                "bootstrap_ci_lower": lower, "bootstrap_ci_upper": upper,
                "training_generation_cells": len(selected), "classifier_observations": len(values),
            })

    definitions = [
        ("sparse_full_correct_minus_label", ("sparse_m4_ft", "correct"), ("sparse_m4_ft", "label")),
        ("sparse_full_shuffled_minus_label", ("sparse_m4_ft", "shuffled"), ("sparse_m4_ft", "label")),
        ("sparse_bank_minus_label", ("sparse_m4_ft", "bank"), ("sparse_m4_ft", "label")),
        ("matched_bank_minus_label", ("matched_ft", "bank"), ("matched_ft", "label")),
        ("matched_bank_minus_correct", ("matched_ft", "bank"), ("matched_ft", "correct")),
        ("matched_bank_minus_shuffled", ("matched_ft", "bank"), ("matched_ft", "shuffled")),
        ("matched_correct_minus_bank", ("matched_ft", "correct"), ("matched_ft", "bank")),
        ("matched_shuffled_minus_bank", ("matched_ft", "shuffled"), ("matched_ft", "bank")),
        ("sparse_minus_matched_correct", ("sparse_m4_ft", "correct"), ("matched_ft", "correct")),
        ("sparse_minus_matched_shuffled", ("sparse_m4_ft", "shuffled"), ("matched_ft", "shuffled")),
        ("sparse_minus_matched_bank", ("sparse_m4_ft", "bank"), ("matched_ft", "bank")),
        ("matched_label_minus_sparse_label", ("matched_ft", "label"), ("sparse_m4_ft", "label")),
        ("matched_shuffled_minus_matched_label", ("matched_ft", "shuffled"), ("matched_ft", "label")),
        ("matched_correct_minus_matched_shuffled", ("matched_ft", "correct"), ("matched_ft", "shuffled")),
        ("matched_correct_minus_sparse_label", ("matched_ft", "correct"), ("sparse_m4_ft", "label")),
    ]
    contrasts = []
    seeds = sorted({row["training_seed"] for row in cells})
    generations = sorted({row["generation_seed"] for row in cells})
    for offset, (name, left, right) in enumerate(definitions):
        paired = []
        for training_seed in seeds:
            for generation_seed in generations:
                a = lookup[(left[0], training_seed, generation_seed, left[1])]
                b = lookup[(right[0], training_seed, generation_seed, right[1])]
                paired.append({
                    "training_seed": training_seed, "generation_seed": generation_seed,
                    "differences": [x - y for x, y in zip(a, b)],
                })
        lower, upper, lower90, upper90 = bootstrap(
            paired, "differences", args.bootstrap_samples, 20260830 + offset
        )
        values = [value for row in paired for value in row["differences"]]
        contrasts.append({
            "contrast": name, "mean_difference": statistics.fmean(values),
            "bootstrap_ci95_lower": lower, "bootstrap_ci95_upper": upper,
            "bootstrap_ci90_lower": lower90, "bootstrap_ci90_upper": upper90,
            "equivalent_within_1pt_by_90pct_ci": (
                lower90 >= -args.equivalence_margin and upper90 <= args.equivalence_margin
            ),
            "noninferior_within_1pt_by_95pct_ci": lower >= -args.equivalence_margin,
            "training_generation_cells": len(paired), "paired_classifier_observations": len(values),
        })

    component_names = (
        "matched_label_minus_sparse_label",
        "matched_shuffled_minus_matched_label",
        "matched_correct_minus_matched_shuffled",
    )
    component_rows = {row["contrast"]: row for row in contrasts}
    decomposition = {
        "total_gap": component_rows["matched_correct_minus_sparse_label"]["mean_difference"],
        "checkpoint_gap_under_label": component_rows[component_names[0]]["mean_difference"],
        "descriptive_interface_value": component_rows[component_names[1]]["mean_difference"],
        "cluster_correspondence_value": component_rows[component_names[2]]["mean_difference"],
    }
    decomposition["component_sum"] = sum(decomposition[key] for key in (
        "checkpoint_gap_under_label", "descriptive_interface_value", "cluster_correspondence_value"
    ))

    write_csv(output / "performance.csv", performance)
    write_csv(output / "paired_contrasts.csv", contrasts)
    summary = {
        "format_version": 1, "performance": performance, "paired_contrasts": contrasts,
        "gap_decomposition": decomposition,
        "bootstrap_order": "training seed -> generation seed -> shared paired classifier repeat",
        "primary_questions": {
            "can_sparse_checkpoint_use_full_dcs": "sparse_minus_matched_correct/shuffled",
            "can_sparse_bank_replace_full_dcs_on_matched_checkpoint": "matched_bank_minus_correct/shuffled",
            "noninferiority_rule": "95% CI lower bound >= -1 accuracy point",
            "equivalence_rule": "90% CI entirely within +/-1 accuracy point",
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = range(len(PROMPTS))
    for family in ("sparse_m4_ft", "matched_ft"):
        rows = [next(row for row in performance if row["checkpoint_family"] == family and row["prompt"] == prompt) for prompt in PROMPTS]
        axes[0].errorbar(
            x, [row["mean_accuracy"] for row in rows],
            yerr=[
                [row["mean_accuracy"] - row["bootstrap_ci_lower"] for row in rows],
                [row["bootstrap_ci_upper"] - row["mean_accuracy"] for row in rows],
            ], marker="o", capsize=4, label=family,
        )
    axes[0].set_xticks(list(x), PROMPTS)
    axes[0].set_ylabel("Validation accuracy")
    axes[0].set_title("Checkpoint x inference interface")
    axes[0].legend()
    labels = ("checkpoint gap", "descriptive interface", "correspondence", "total gap")
    values = (
        decomposition["checkpoint_gap_under_label"], decomposition["descriptive_interface_value"],
        decomposition["cluster_correspondence_value"], decomposition["total_gap"],
    )
    axes[1].bar(labels, values)
    axes[1].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[1].tick_params(axis="x", rotation=18)
    axes[1].set_ylabel("Paired accuracy difference")
    axes[1].set_title("Matched Correct - Sparse Label decomposition")
    fig.tight_layout()
    fig.savefig(output / "sparse_interface_transfer.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
