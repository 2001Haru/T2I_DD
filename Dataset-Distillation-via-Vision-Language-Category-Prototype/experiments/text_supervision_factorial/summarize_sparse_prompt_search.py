#!/usr/bin/env python3
"""Summarize nested sparse-caption bank performance and saturation."""

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


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-index", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    return parser.parse_args()


def scores(path):
    matches = RESULT.findall(Path(path).read_text(encoding="utf-8", errors="replace"))
    if not matches:
        raise ValueError(f"No completed classifier result: {path}")
    return [float(value) for value in ast.literal_eval(matches[-1])]


def percentile(values, probability):
    values = sorted(values)
    return values[min(len(values) - 1, int(probability * len(values)))]


def hierarchical_bootstrap(rows, value_key, samples, seed=20260812):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["bank_seed"], {})[row["generation_seed"]] = row[value_key]
    bank_seeds = sorted(grouped)
    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        drawn = []
        for _ in bank_seeds:
            bank_seed = rng.choice(bank_seeds)
            generations = grouped[bank_seed]
            generation_seeds = sorted(generations)
            for _ in generation_seeds:
                generation_seed = rng.choice(generation_seeds)
                values = generations[generation_seed]
                drawn.extend(rng.choice(values) for _ in values)
        estimates.append(statistics.fmean(drawn))
    return percentile(estimates, 0.025), percentile(estimates, 0.975)


def write_csv(path, rows, fields):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    index = json.loads(Path(args.evaluation_index).read_text(encoding="utf-8"))
    cells = []
    by_key = {}
    for row in index:
        values = scores(row["evaluation_log"])
        cell = {**row, "scores": values, "mean_accuracy": statistics.fmean(values)}
        cells.append(cell)
        by_key[(row["bank_seed"], row["budget"], row["generation_seed"], row["prompt"])] = values

    performance = []
    for budget in sorted({row["budget"] for row in cells}):
        for prompt in ("label", "bank"):
            selected = [row for row in cells if row["budget"] == budget and row["prompt"] == prompt]
            lower, upper = hierarchical_bootstrap(selected, "scores", args.bootstrap_samples)
            flattened = [value for row in selected for value in row["scores"]]
            performance.append({
                "budget": budget, "prompt": prompt,
                "mean_accuracy": statistics.fmean(flattened),
                "bootstrap_ci_lower": lower, "bootstrap_ci_upper": upper,
                "bank_generation_cells": len(selected),
                "classifier_observations": len(flattened),
            })

    contrasts = []
    for budget in sorted({row["budget"] for row in cells}):
        paired_rows = []
        for bank_seed in sorted({row["bank_seed"] for row in cells}):
            for generation_seed in sorted({row["generation_seed"] for row in cells}):
                bank = by_key[(bank_seed, budget, generation_seed, "bank")]
                label = by_key[(bank_seed, budget, generation_seed, "label")]
                paired_rows.append({
                    "bank_seed": bank_seed, "generation_seed": generation_seed,
                    "differences": [left - right for left, right in zip(bank, label)],
                })
        lower, upper = hierarchical_bootstrap(
            paired_rows, "differences", args.bootstrap_samples, seed=20260812 + budget
        )
        values = [value for row in paired_rows for value in row["differences"]]
        contrasts.append({
            "budget": budget, "contrast": "bank_minus_label",
            "mean_difference": statistics.fmean(values),
            "bootstrap_ci_lower": lower, "bootstrap_ci_upper": upper,
            "bank_generation_cells": len(paired_rows),
            "paired_classifier_observations": len(values),
        })

    maximum = max(row["budget"] for row in cells)
    saturation = []
    for budget in sorted({row["budget"] for row in cells}):
        paired_rows = []
        for bank_seed in sorted({row["bank_seed"] for row in cells}):
            for generation_seed in sorted({row["generation_seed"] for row in cells}):
                current = by_key[(bank_seed, budget, generation_seed, "bank")]
                endpoint = by_key[(bank_seed, maximum, generation_seed, "bank")]
                paired_rows.append({
                    "bank_seed": bank_seed, "generation_seed": generation_seed,
                    "differences": [left - right for left, right in zip(current, endpoint)],
                })
        lower, upper = hierarchical_bootstrap(
            paired_rows, "differences", args.bootstrap_samples, seed=20260900 + budget
        )
        values = [value for row in paired_rows for value in row["differences"]]
        saturation.append({
            "budget": budget, "reference_budget": maximum,
            "contrast": "bank_budget_minus_bank_maximum",
            "mean_difference": statistics.fmean(values),
            "bootstrap_ci_lower": lower, "bootstrap_ci_upper": upper,
            "noninferior_within_1pt_by_95pct_ci": lower >= -1.0,
        })

    write_csv(output / "performance_by_budget.csv", performance, performance[0].keys())
    write_csv(output / "bank_gain_by_budget.csv", contrasts, contrasts[0].keys())
    write_csv(output / "saturation_vs_m32.csv", saturation, saturation[0].keys())
    summary = {
        "format_version": 1,
        "performance": performance,
        "bank_minus_label": contrasts,
        "saturation": saturation,
        "bootstrap_order": "bank seed -> generation seed -> paired classifier repeat",
        "selection_design": "nested random class-caption banks",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    budgets = [row["budget"] for row in performance if row["prompt"] == "bank"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for prompt in ("label", "bank"):
        rows = [row for row in performance if row["prompt"] == prompt]
        axes[0].errorbar(
            [row["budget"] for row in rows], [row["mean_accuracy"] for row in rows],
            yerr=[
                [row["mean_accuracy"] - row["bootstrap_ci_lower"] for row in rows],
                [row["bootstrap_ci_upper"] - row["mean_accuracy"] for row in rows],
            ], marker="o", capsize=4, label=prompt,
        )
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks(budgets, labels=budgets)
    axes[0].set_xlabel("Captions per class (m)")
    axes[0].set_ylabel("Validation accuracy")
    axes[0].set_title("Sparse prompt-bank performance")
    axes[0].legend()
    axes[1].errorbar(
        [row["budget"] for row in contrasts], [row["mean_difference"] for row in contrasts],
        yerr=[
            [row["mean_difference"] - row["bootstrap_ci_lower"] for row in contrasts],
            [row["bootstrap_ci_upper"] - row["mean_difference"] for row in contrasts],
        ], marker="o", capsize=4,
    )
    axes[1].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[1].set_xscale("log", base=2)
    axes[1].set_xticks(budgets, labels=budgets)
    axes[1].set_xlabel("Captions per class (m)")
    axes[1].set_ylabel("Bank minus label accuracy")
    axes[1].set_title("Marginal utility and saturation")
    fig.tight_layout()
    fig.savefig(output / "sparse_prompt_search.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
