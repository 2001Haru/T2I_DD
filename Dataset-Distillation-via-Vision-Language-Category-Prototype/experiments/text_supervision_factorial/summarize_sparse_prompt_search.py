#!/usr/bin/env python3
"""Summarize nested sparse-caption bank performance and saturation."""

import argparse
import ast
import csv
import json
import math
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


def linear_slope(xs, ys):
    x_mean = statistics.fmean(xs)
    y_mean = statistics.fmean(ys)
    denominator = sum((value - x_mean) ** 2 for value in xs)
    if denominator <= 0:
        raise ValueError("At least two distinct caption budgets are required for regression")
    return sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denominator


def paired_label_budget_bootstrap(rows, budgets, samples, seed=20261001):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["bank_seed"], {}).setdefault(row["generation_seed"], {})[
            row["budget"]
        ] = row["scores"]
    bank_seeds = sorted(grouped)
    if not bank_seeds:
        raise ValueError("No Label rows available for budget analysis")
    for bank_seed, generations in grouped.items():
        for generation_seed, cells in generations.items():
            missing = set(budgets) - set(cells)
            if missing:
                raise ValueError(
                    f"Incomplete Label budget grid for bank={bank_seed}, "
                    f"generation={generation_seed}: missing {sorted(missing)}"
                )
            repeat_counts = {len(cells[budget]) for budget in budgets}
            if len(repeat_counts) != 1:
                raise ValueError(
                    f"Classifier-repeat mismatch across budgets for bank={bank_seed}, "
                    f"generation={generation_seed}: {sorted(repeat_counts)}"
                )

    minimum, maximum = min(budgets), max(budgets)

    def estimate(draw_rng=None):
        differences, xs, ys = [], [], []
        selected_banks = bank_seeds if draw_rng is None else [
            draw_rng.choice(bank_seeds) for _ in bank_seeds
        ]
        for bank_seed in selected_banks:
            generations = grouped[bank_seed]
            generation_seeds = sorted(generations)
            selected_generations = generation_seeds if draw_rng is None else [
                draw_rng.choice(generation_seeds) for _ in generation_seeds
            ]
            for generation_seed in selected_generations:
                cells = generations[generation_seed]
                repeat_count = len(cells[budgets[0]])
                repeat_indices = range(repeat_count) if draw_rng is None else [
                    draw_rng.randrange(repeat_count) for _ in range(repeat_count)
                ]
                for repeat_index in repeat_indices:
                    differences.append(
                        cells[minimum][repeat_index] - cells[maximum][repeat_index]
                    )
                    for budget in budgets:
                        xs.append(math.log2(budget))
                        ys.append(cells[budget][repeat_index])
        return statistics.fmean(differences), linear_slope(xs, ys)

    point_difference, point_slope = estimate()
    rng = random.Random(seed)
    difference_estimates, slope_estimates = [], []
    for _ in range(samples):
        difference, slope = estimate(rng)
        difference_estimates.append(difference)
        slope_estimates.append(slope)
    cells = sum(len(generations) for generations in grouped.values())
    observations = sum(
        len(cells_by_budget[budgets[0]])
        for generations in grouped.values()
        for cells_by_budget in generations.values()
    )
    return {
        "minimum_budget": minimum,
        "maximum_budget": maximum,
        "m_min_minus_m_max_mean_difference": point_difference,
        "m_min_minus_m_max_bootstrap_ci_lower": percentile(difference_estimates, 0.025),
        "m_min_minus_m_max_bootstrap_ci_upper": percentile(difference_estimates, 0.975),
        "log2_budget_slope": point_slope,
        "log2_budget_slope_bootstrap_ci_lower": percentile(slope_estimates, 0.025),
        "log2_budget_slope_bootstrap_ci_upper": percentile(slope_estimates, 0.975),
        "bank_generation_cells": cells,
        "paired_classifier_observations": observations,
        "bootstrap_order": (
            "bank seed -> generation seed -> shared classifier-repeat draw across budgets"
        ),
    }


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

    available_prompts = sorted({row["prompt"] for row in cells})
    if not available_prompts:
        raise ValueError("Evaluation index contains no prompt conditions")

    performance = []
    for budget in sorted({row["budget"] for row in cells}):
        for prompt in available_prompts:
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

    bank_prompt = "bank_t77" if "bank_t77" in available_prompts else (
        "bank" if "bank" in available_prompts else None
    )
    contrasts = []
    if "label" in available_prompts and bank_prompt:
        for budget in sorted({row["budget"] for row in cells}):
            paired_rows = []
            for bank_seed in sorted({row["bank_seed"] for row in cells}):
                for generation_seed in sorted({row["generation_seed"] for row in cells}):
                    bank = by_key[(bank_seed, budget, generation_seed, bank_prompt)]
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
                "budget": budget, "contrast": f"{bank_prompt}_minus_label",
                "mean_difference": statistics.fmean(values),
                "bootstrap_ci_lower": lower, "bootstrap_ci_upper": upper,
                "bank_generation_cells": len(paired_rows),
                "paired_classifier_observations": len(values),
            })

    saturation = []
    if bank_prompt:
        maximum = max(row["budget"] for row in cells)
        for budget in sorted({row["budget"] for row in cells}):
            paired_rows = []
            for bank_seed in sorted({row["bank_seed"] for row in cells}):
                for generation_seed in sorted({row["generation_seed"] for row in cells}):
                    current = by_key[(bank_seed, budget, generation_seed, bank_prompt)]
                    endpoint = by_key[(bank_seed, maximum, generation_seed, bank_prompt)]
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
                "noninferior_within_1pt_by_95pct_ci": lower >= -1.0 - 1e-12,
            })

    label_budget_analysis = []
    budgets = sorted({row["budget"] for row in cells})
    if "label" in available_prompts and len(budgets) >= 2:
        result = paired_label_budget_bootstrap(
            [row for row in cells if row["prompt"] == "label"],
            budgets, args.bootstrap_samples,
        )
        label_budget_analysis.append({
            "prompt": "label",
            "budgets": ",".join(str(value) for value in budgets),
            **result,
        })

    write_csv(output / "performance_by_budget.csv", performance, performance[0].keys())
    write_csv(
        output / "bank_gain_by_budget.csv", contrasts,
        contrasts[0].keys() if contrasts else (
            "budget", "contrast", "mean_difference", "bootstrap_ci_lower",
            "bootstrap_ci_upper", "bank_generation_cells", "paired_classifier_observations",
        ),
    )
    write_csv(
        output / "saturation_vs_maximum.csv", saturation,
        saturation[0].keys() if saturation else (
            "budget", "reference_budget", "contrast", "mean_difference",
            "bootstrap_ci_lower", "bootstrap_ci_upper", "noninferior_within_1pt_by_95pct_ci",
        ),
    )
    if saturation and max(row["budget"] for row in saturation) == 32:
        write_csv(output / "saturation_vs_m32.csv", saturation, saturation[0].keys())
    write_csv(
        output / "label_budget_regression.csv", label_budget_analysis,
        label_budget_analysis[0].keys() if label_budget_analysis else (
            "prompt", "budgets", "minimum_budget", "maximum_budget",
            "m_min_minus_m_max_mean_difference",
            "m_min_minus_m_max_bootstrap_ci_lower",
            "m_min_minus_m_max_bootstrap_ci_upper", "log2_budget_slope",
            "log2_budget_slope_bootstrap_ci_lower",
            "log2_budget_slope_bootstrap_ci_upper", "bank_generation_cells",
            "paired_classifier_observations", "bootstrap_order",
        ),
    )
    summary = {
        "format_version": 2,
        "performance": performance,
        "bank_minus_label": contrasts,
        "saturation": saturation,
        "label_budget_analysis": label_budget_analysis,
        "bootstrap_order": "bank seed -> generation seed -> paired classifier repeat",
        "selection_design": "nested random class-caption banks",
        "bank_inference_prompt": bank_prompt,
        "available_prompts": available_prompts,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    budgets = sorted({row["budget"] for row in performance})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for prompt in available_prompts:
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
    if contrasts:
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
    else:
        axes[1].axis("off")
        axes[1].text(
            0.5, 0.5, "Bank inference was not requested",
            ha="center", va="center", transform=axes[1].transAxes,
        )
    fig.tight_layout()
    fig.savefig(output / "sparse_prompt_search.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
