#!/usr/bin/env python3
"""Pool T77 sparse-budget and dense-checkpoint noise/contrast statistics."""

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
SPARSE_PROMPTS = ("label", "bank_t77")
EXPECTED_BUDGETS = (4, 8, 16, 32, 64, 128, 256, 512)
DENSE_FAMILIES = ("matched_ft", "unpaired_ft")
DENSE_PROMPTS = ("correct_t77", "shuffled_t77")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sparse-index", nargs="+", required=True)
    parser.add_argument("--fixed-index", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260818)
    return parser.parse_args()


def read_scores(path):
    matches = RESULT.findall(Path(path).read_text(encoding="utf-8", errors="replace"))
    if not matches:
        raise ValueError(f"No completed classifier result: {path}")
    return [float(value) for value in ast.literal_eval(matches[-1])]


def percentile(values, probability):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(probability * len(ordered)))]


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_sparse(paths):
    records = {}
    sources = []
    for path in paths:
        resolved = Path(path).resolve()
        sources.append(str(resolved))
        for row in json.loads(resolved.read_text(encoding="utf-8")):
            if row.get("prompt") not in SPARSE_PROMPTS:
                continue
            key = (
                int(row["budget"]), int(row["bank_seed"]),
                int(row["generation_seed"]), row["prompt"],
            )
            payload = {**row, "scores": read_scores(row["evaluation_log"]), "source_index": str(resolved)}
            if key in records:
                raise ValueError(f"Duplicate sparse cell across indexes: {key}")
            records[key] = payload
    budgets = sorted({key[0] for key in records})
    if tuple(budgets) != EXPECTED_BUDGETS:
        raise ValueError(f"Expected sparse budgets {EXPECTED_BUDGETS}, found {budgets}")
    banks = sorted({key[1] for key in records})
    generations = sorted({key[2] for key in records})
    for budget in budgets:
        for bank in banks:
            for generation in generations:
                for prompt in SPARSE_PROMPTS:
                    if (budget, bank, generation, prompt) not in records:
                        raise ValueError(
                            f"Missing sparse cell: {(budget, bank, generation, prompt)}"
                        )
                repeat_counts = {
                    len(records[(budget, bank, generation, prompt)]["scores"])
                    for prompt in SPARSE_PROMPTS
                }
                if len(repeat_counts) != 1:
                    raise ValueError(
                        f"Sparse repeat mismatch: budget={budget}, bank={bank}, gen={generation}"
                    )
    return records, budgets, banks, generations, sources


def sparse_value(records, budget, bank, generation, repeat, outcome):
    label = records[(budget, bank, generation, "label")]["scores"][repeat]
    bank_value = records[(budget, bank, generation, "bank_t77")]["scores"][repeat]
    if outcome == "label":
        return label
    if outcome == "bank_t77":
        return bank_value
    if outcome == "bank_t77_minus_label":
        return bank_value - label
    raise ValueError(outcome)


def nested_moments(values):
    """Balanced budget-fixed, checkpoint -> generation -> classifier ANOVA."""
    budget_count = len(values)
    bank_count = len(values[0])
    generation_count = len(values[0][0])
    repeat_count = len(values[0][0][0])
    if min(budget_count, bank_count, generation_count, repeat_count) < 2:
        raise ValueError("Every nested level needs at least two observations")

    ss_bank = ss_generation = ss_classifier = 0.0
    for budget_values in values:
        budget_flat = [
            value for bank_values in budget_values
            for generation_values in bank_values for value in generation_values
        ]
        budget_mean = statistics.fmean(budget_flat)
        for bank_values in budget_values:
            bank_flat = [value for generation_values in bank_values for value in generation_values]
            bank_mean = statistics.fmean(bank_flat)
            ss_bank += generation_count * repeat_count * (bank_mean - budget_mean) ** 2
            for generation_values in bank_values:
                generation_mean = statistics.fmean(generation_values)
                ss_generation += repeat_count * (generation_mean - bank_mean) ** 2
                ss_classifier += sum(
                    (value - generation_mean) ** 2 for value in generation_values
                )

    df_bank = budget_count * (bank_count - 1)
    df_generation = budget_count * bank_count * (generation_count - 1)
    df_classifier = budget_count * bank_count * generation_count * (repeat_count - 1)
    ms_bank = ss_bank / df_bank
    ms_generation = ss_generation / df_generation
    ms_classifier = ss_classifier / df_classifier
    variances = {
        "classifier": max(0.0, ms_classifier),
        "generation": max(0.0, (ms_generation - ms_classifier) / repeat_count),
        "bank_checkpoint": max(
            0.0,
            (ms_bank - ms_generation) / (generation_count * repeat_count),
        ),
    }
    return {
        "variances": variances,
        "mean_squares": {
            "bank_checkpoint": ms_bank,
            "generation": ms_generation,
            "classifier": ms_classifier,
        },
        "degrees_of_freedom": {
            "bank_checkpoint": df_bank,
            "generation": df_generation,
            "classifier": df_classifier,
        },
    }


def sparse_array(records, budgets, banks, generations, outcome):
    output = []
    for budget in budgets:
        budget_values = []
        for bank in banks:
            bank_values = []
            for generation in generations:
                repeat_count = len(records[(budget, bank, generation, "label")]["scores"])
                bank_values.append([
                    sparse_value(records, budget, bank, generation, repeat, outcome)
                    for repeat in range(repeat_count)
                ])
            budget_values.append(bank_values)
        output.append(budget_values)
    return output


def resample_nested(values, rng):
    result = []
    for budget_values in values:  # Budget levels remain fixed strata.
        bank_count = len(budget_values)
        sampled_budget = []
        for _ in range(bank_count):
            source_bank = budget_values[rng.randrange(bank_count)]
            generation_count = len(source_bank)
            sampled_bank = []
            for _ in range(generation_count):
                source_generation = source_bank[rng.randrange(generation_count)]
                repeat_count = len(source_generation)
                sampled_bank.append([
                    source_generation[rng.randrange(repeat_count)]
                    for _ in range(repeat_count)
                ])
            sampled_budget.append(sampled_bank)
        result.append(sampled_budget)
    return result


def variance_decomposition(records, budgets, banks, generations, samples, seed):
    rng = random.Random(seed)
    rows, details = [], []
    for outcome_index, outcome in enumerate(("label", "bank_t77", "bank_t77_minus_label")):
        values = sparse_array(records, budgets, banks, generations, outcome)
        point = nested_moments(values)
        draws = {component: [] for component in point["variances"]}
        for _ in range(samples):
            estimate = nested_moments(resample_nested(values, rng))["variances"]
            for component, value in estimate.items():
                draws[component].append(value)
        total = sum(point["variances"].values())
        for component in ("bank_checkpoint", "generation", "classifier"):
            variance = point["variances"][component]
            lower = percentile(draws[component], 0.025)
            upper = percentile(draws[component], 0.975)
            rows.append({
                "outcome": outcome, "component": component,
                "variance": variance, "standard_deviation": math.sqrt(variance),
                "bootstrap_ci95_variance_lower": lower,
                "bootstrap_ci95_variance_upper": upper,
                "bootstrap_ci95_sd_lower": math.sqrt(lower),
                "bootstrap_ci95_sd_upper": math.sqrt(upper),
                "fraction_of_total_variance": variance / total if total else 0.0,
                "fixed_budget_strata": len(budgets),
                "bank_checkpoint_levels_per_budget": len(banks),
                "generation_levels_per_checkpoint": len(generations),
                "classifier_repeats_per_generation": len(values[0][0][0]),
            })
        details.append({"outcome": outcome, **point})
    return rows, details


def paired_sparse_bootstrap(records, budget, banks, generations, samples, seed):
    rng = random.Random(seed)

    def estimate(draw_rng=None):
        selected_banks = banks if draw_rng is None else [
            draw_rng.choice(banks) for _ in banks
        ]
        values = []
        for bank in selected_banks:
            selected_generations = generations if draw_rng is None else [
                draw_rng.choice(generations) for _ in generations
            ]
            for generation in selected_generations:
                repeat_count = len(records[(budget, bank, generation, "label")]["scores"])
                repeats = range(repeat_count) if draw_rng is None else [
                    draw_rng.randrange(repeat_count) for _ in range(repeat_count)
                ]
                values.extend(
                    sparse_value(
                        records, budget, bank, generation, repeat,
                        "bank_t77_minus_label",
                    )
                    for repeat in repeats
                )
        return statistics.fmean(values)

    point = estimate()
    draws = [estimate(rng) for _ in range(samples)]
    return point, percentile(draws, 0.025), percentile(draws, 0.975)


def sparse_contrasts(records, budgets, banks, generations, samples, seed):
    rows = []
    for offset, budget in enumerate(budgets):
        mean, lower, upper = paired_sparse_bootstrap(
            records, budget, banks, generations, samples, seed + offset
        )
        rows.append({
            "budget": budget, "contrast": "bank_t77_minus_label",
            "mean_difference": mean, "bootstrap_ci95_lower": lower,
            "bootstrap_ci95_upper": upper,
            "bank_checkpoint_levels": len(banks),
            "bank_generation_cells": len(banks) * len(generations),
            "paired_classifier_observations": sum(
                len(records[(budget, bank, generation, "label")]["scores"])
                for bank in banks for generation in generations
            ),
            "bootstrap_order": "bank/checkpoint -> generation seed -> shared classifier repeat",
        })
    return rows


def load_fixed(path):
    source = Path(path).resolve()
    records = {}
    for row in json.loads(source.read_text(encoding="utf-8")):
        if row.get("checkpoint_family") not in DENSE_FAMILIES:
            continue
        if row.get("prompt") not in DENSE_PROMPTS:
            continue
        key = (
            int(row["training_seed"]), int(row["generation_seed"]),
            row["checkpoint_family"], row["prompt"],
        )
        if key in records:
            raise ValueError(f"Duplicate fixed cell: {key}")
        records[key] = {**row, "scores": read_scores(row["evaluation_log"])}
    training_seeds = sorted({key[0] for key in records})
    generation_seeds = sorted({key[1] for key in records})
    for training_seed in training_seeds:
        for generation_seed in generation_seeds:
            for family in DENSE_FAMILIES:
                for prompt in DENSE_PROMPTS:
                    if (training_seed, generation_seed, family, prompt) not in records:
                        raise ValueError(
                            "Missing fixed cell: "
                            f"{(training_seed, generation_seed, family, prompt)}"
                        )
    return records, training_seeds, generation_seeds, str(source)


def fixed_contrast_value(records, training, generation, repeat, contrast, prompt=None, family=None):
    if contrast == "matched_minus_unpaired":
        return (
            records[(training, generation, "matched_ft", prompt)]["scores"][repeat]
            - records[(training, generation, "unpaired_ft", prompt)]["scores"][repeat]
        )
    if contrast == "correct_minus_shuffled":
        return (
            records[(training, generation, family, "correct_t77")]["scores"][repeat]
            - records[(training, generation, family, "shuffled_t77")]["scores"][repeat]
        )
    if contrast == "matching_specific_interaction":
        return (
            records[(training, generation, "matched_ft", "correct_t77")]["scores"][repeat]
            - records[(training, generation, "unpaired_ft", "correct_t77")]["scores"][repeat]
            - records[(training, generation, "matched_ft", "shuffled_t77")]["scores"][repeat]
            + records[(training, generation, "unpaired_ft", "shuffled_t77")]["scores"][repeat]
        )
    raise ValueError(contrast)


def fixed_contrast_bootstrap(
    records, training_seeds, generation_seeds, contrast, samples, seed,
    prompt=None, family=None,
):
    rng = random.Random(seed)

    def estimate(draw_rng=None):
        selected_training = training_seeds if draw_rng is None else [
            draw_rng.choice(training_seeds) for _ in training_seeds
        ]
        values = []
        for training in selected_training:
            selected_generation = generation_seeds if draw_rng is None else [
                draw_rng.choice(generation_seeds) for _ in generation_seeds
            ]
            for generation in selected_generation:
                reference = (
                    records[(training, generation, "matched_ft", prompt)]
                    if contrast == "matched_minus_unpaired"
                    else (
                        records[(training, generation, family, "correct_t77")]
                        if contrast == "correct_minus_shuffled"
                        else records[(training, generation, "matched_ft", "correct_t77")]
                    )
                )
                repeat_count = len(reference["scores"])
                repeats = range(repeat_count) if draw_rng is None else [
                    draw_rng.randrange(repeat_count) for _ in range(repeat_count)
                ]
                values.extend(
                    fixed_contrast_value(
                        records, training, generation, repeat, contrast,
                        prompt=prompt, family=family,
                    )
                    for repeat in repeats
                )
        return statistics.fmean(values)

    point = estimate()
    draws = [estimate(rng) for _ in range(samples)]
    return point, percentile(draws, 0.025), percentile(draws, 0.975)


def fixed_contrasts(records, training_seeds, generation_seeds, samples, seed):
    rows = []
    specifications = [
        ("matched_minus_unpaired", prompt, None) for prompt in DENSE_PROMPTS
    ] + [
        ("correct_minus_shuffled", None, family) for family in DENSE_FAMILIES
    ] + [("matching_specific_interaction", None, None)]
    for offset, (contrast, prompt, family) in enumerate(specifications):
        mean, lower, upper = fixed_contrast_bootstrap(
            records, training_seeds, generation_seeds, contrast, samples,
            seed + offset, prompt=prompt, family=family,
        )
        rows.append({
            "contrast": contrast, "prompt": prompt or "",
            "checkpoint_family": family or "",
            "mean_difference": mean, "bootstrap_ci95_lower": lower,
            "bootstrap_ci95_upper": upper,
            "training_seed_levels": len(training_seeds),
            "training_generation_cells": len(training_seeds) * len(generation_seeds),
            "paired_classifier_observations": sum(
                len(
                    records[(training, generation, "matched_ft", prompt)]["scores"]
                    if contrast == "matched_minus_unpaired"
                    else (
                        records[(training, generation, family, "correct_t77")]["scores"]
                        if contrast == "correct_minus_shuffled"
                        else records[(training, generation, "matched_ft", "correct_t77")]["scores"]
                    )
                )
                for training in training_seeds for generation in generation_seeds
            ),
            "bootstrap_order": "training seed -> generation seed -> shared classifier repeat",
        })
    return rows


def plot_results(variance_rows, sparse_rows, fixed_rows, output):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    outcomes = ("label", "bank_t77", "bank_t77_minus_label")
    components = ("bank_checkpoint", "generation", "classifier")
    width = 0.24
    for component_index, component in enumerate(components):
        selected = [
            next(row for row in variance_rows if row["outcome"] == outcome and row["component"] == component)
            for outcome in outcomes
        ]
        axes[0].bar(
            [index + (component_index - 1) * width for index in range(len(outcomes))],
            [row["standard_deviation"] for row in selected], width=width, label=component,
        )
    axes[0].set_xticks(range(len(outcomes)), ["Label", "Bank-T77", "Bank-Label"])
    axes[0].set_ylabel("Estimated SD (accuracy points)")
    axes[0].set_title("Nested noise components")
    axes[0].legend()

    axes[1].errorbar(
        [row["budget"] for row in sparse_rows],
        [row["mean_difference"] for row in sparse_rows],
        yerr=[
            [row["mean_difference"] - row["bootstrap_ci95_lower"] for row in sparse_rows],
            [row["bootstrap_ci95_upper"] - row["mean_difference"] for row in sparse_rows],
        ],
        marker="o", capsize=4,
    )
    axes[1].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[1].set_xscale("log", base=2)
    axes[1].set_xticks(EXPECTED_BUDGETS, [str(value) for value in EXPECTED_BUDGETS])
    axes[1].set_xlabel("Caption budget m")
    axes[1].set_ylabel("Bank-T77 - Label accuracy")
    axes[1].set_title("Within-checkpoint prompt utility")

    labels = [
        f"M-U\n{row['prompt']}" if row["contrast"] == "matched_minus_unpaired"
        else (
            f"C-S\n{row['checkpoint_family']}"
            if row["contrast"] == "correct_minus_shuffled"
            else "Matching-specific\ninteraction"
        )
        for row in fixed_rows
    ]
    axes[2].errorbar(
        range(len(fixed_rows)), [row["mean_difference"] for row in fixed_rows],
        yerr=[
            [row["mean_difference"] - row["bootstrap_ci95_lower"] for row in fixed_rows],
            [row["bootstrap_ci95_upper"] - row["mean_difference"] for row in fixed_rows],
        ],
        fmt="o", capsize=4,
    )
    axes[2].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[2].set_xticks(range(len(labels)), labels)
    axes[2].set_ylabel("Paired accuracy difference")
    axes[2].set_title("Dense-checkpoint contrasts")
    fig.tight_layout()
    fig.savefig(output / "t77_noise_and_contrasts.png", dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    sparse, budgets, banks, generations, sparse_sources = load_sparse(args.sparse_index)
    variance_rows, variance_details = variance_decomposition(
        sparse, budgets, banks, generations,
        args.bootstrap_samples, args.bootstrap_seed,
    )
    sparse_rows = sparse_contrasts(
        sparse, budgets, banks, generations,
        args.bootstrap_samples, args.bootstrap_seed + 100,
    )
    fixed, training_seeds, fixed_generations, fixed_source = load_fixed(args.fixed_index)
    fixed_rows = fixed_contrasts(
        fixed, training_seeds, fixed_generations,
        args.bootstrap_samples, args.bootstrap_seed + 200,
    )
    write_csv(output / "variance_components.csv", variance_rows)
    write_csv(output / "bank_minus_label_by_budget.csv", sparse_rows)
    write_csv(output / "dense_paired_contrasts.csv", fixed_rows)
    plot_results(variance_rows, sparse_rows, fixed_rows, output)
    summary = {
        "format_version": 1,
        "protocol": {
            "sparse_variance_model": (
                "budget fixed effect; bank/checkpoint nested within budget; generation seed "
                "nested within bank/checkpoint; classifier repeat residual"
            ),
            "variance_bootstrap": (
                "fixed budget strata -> bank/checkpoint -> generation seed -> classifier repeat"
            ),
            "paired_sparse_bootstrap": (
                "bank/checkpoint -> generation seed -> shared classifier repeat"
            ),
            "paired_dense_bootstrap": (
                "training seed -> generation seed -> shared classifier repeat"
            ),
            "interpretation_boundary": (
                "Sparse bank_seed changes both the selected caption bank and the trained checkpoint; "
                "bank_checkpoint variance is not a pure optimization-seed variance. Pairing removes "
                "an additive checkpoint intercept but not checkpoint-by-prompt heterogeneity."
            ),
        },
        "sources": {"sparse_indexes": sparse_sources, "fixed_index": fixed_source},
        "variance_components": variance_rows,
        "variance_anova_details": variance_details,
        "bank_minus_label_by_budget": sparse_rows,
        "dense_paired_contrasts": fixed_rows,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Saved pooled T77 noise analysis to {output}")


if __name__ == "__main__":
    main()
