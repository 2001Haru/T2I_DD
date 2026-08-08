#!/usr/bin/env python3
"""Summarize paired prompt utility across initialization strength and IPC."""

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


def hierarchical_bootstrap(rows, samples=10000, seed=20260808):
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
            generations = grouped[training_seed]
            generation_seeds = list(generations)
            for _ in generation_seeds:
                generation_seed = rng.choice(generation_seeds)
                values = rng.choice(generations[generation_seed])
                draw.extend(rng.choice(values) for _ in values)
        estimates.append(statistics.fmean(draw))
    estimates.sort()
    return estimates[int(0.025 * samples)], estimates[min(samples - 1, int(0.975 * samples))]


def summarize(rows):
    flat = [value for row in rows for value in row["values"]]
    lower, upper = hierarchical_bootstrap(rows)
    return {
        "mean": statistics.fmean(flat),
        "hierarchical_bootstrap_ci_lower": lower,
        "hierarchical_bootstrap_ci_upper": upper,
        "training_generation_cells": len(rows),
        "paired_classifier_observations": len(flat),
    }


def difference_rows(lookup, ipc, strength, left, right):
    rows = []
    keys = sorted({(key[2], key[3]) for key in lookup if key[0] == ipc and key[1] == strength})
    for training_seed, generation_seed in keys:
        left_key = (ipc, strength, training_seed, generation_seed, left)
        right_key = (ipc, strength, training_seed, generation_seed, right)
        if left_key in lookup and right_key in lookup:
            rows.append({
                "training_seed": training_seed,
                "generation_seed": generation_seed,
                "values": paired(lookup[left_key], lookup[right_key]),
            })
    return rows


def interaction_rows(lookup, ipc, strength, reference=0.7):
    rows = []
    keys = sorted({(key[2], key[3]) for key in lookup if key[0] == ipc and key[1] == strength})
    for training_seed, generation_seed in keys:
        required = [
            (ipc, strength, training_seed, generation_seed, "correct"),
            (ipc, strength, training_seed, generation_seed, "label"),
            (ipc, reference, training_seed, generation_seed, "correct"),
            (ipc, reference, training_seed, generation_seed, "label"),
        ]
        if all(key in lookup for key in required):
            current = paired(lookup[required[0]], lookup[required[1]])
            baseline = paired(lookup[required[2]], lookup[required[3]])
            rows.append({
                "training_seed": training_seed, "generation_seed": generation_seed,
                "values": paired(current, baseline),
            })
    return rows


def ipc_interaction_rows(lookup, strength, low_ipc, high_ipc, reference=None):
    rows = []
    keys = sorted({(key[2], key[3]) for key in lookup if key[0] == low_ipc and key[1] == strength})
    for training_seed, generation_seed in keys:
        def utility(ipc, current_strength):
            correct = (ipc, current_strength, training_seed, generation_seed, "correct")
            label = (ipc, current_strength, training_seed, generation_seed, "label")
            if correct not in lookup or label not in lookup:
                return None
            return paired(lookup[correct], lookup[label])

        low = utility(low_ipc, strength)
        high = utility(high_ipc, strength)
        if low is None or high is None:
            continue
        values = paired(high, low)
        if reference is not None:
            low_reference = utility(low_ipc, reference)
            high_reference = utility(high_ipc, reference)
            if low_reference is None or high_reference is None:
                continue
            values = paired(values, paired(high_reference, low_reference))
        rows.append({"training_seed": training_seed, "generation_seed": generation_seed, "values": values})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-index", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    index = json.loads(Path(args.evaluation_index).read_text(encoding="utf-8"))
    cells, lookup = [], {}
    for item in index:
        scores = read_scores(item["evaluation_log"])
        row = {**item, "mean_accuracy": statistics.fmean(scores), "std_accuracy": statistics.pstdev(scores), "classifier_accuracies": scores}
        cells.append(row)
        key = (item["ipc"], float(item["strength"]), item["training_seed"], item["generation_seed"], item["prompt"])
        if key in lookup:
            raise RuntimeError(f"Duplicate evaluation cell: {key}")
        lookup[key] = scores

    performance = []
    for ipc, strength, prompt in sorted({(row["ipc"], float(row["strength"]), row["prompt"]) for row in cells}):
        rows = [
            {"training_seed": row["training_seed"], "generation_seed": row["generation_seed"], "values": row["classifier_accuracies"]}
            for row in cells if (row["ipc"], float(row["strength"]), row["prompt"]) == (ipc, strength, prompt)
        ]
        performance.append({"ipc": ipc, "strength": strength, "prompt": prompt, **summarize(rows)})

    contrasts = []
    strengths = sorted({float(row["strength"]) for row in cells})
    ipcs = sorted({row["ipc"] for row in cells})
    contrast_specs = (("correct_minus_label", "correct", "label"), ("shuffled_minus_label", "shuffled", "label"), ("correct_minus_shuffled", "correct", "shuffled"))
    for ipc in ipcs:
        for strength in strengths:
            for name, left, right in contrast_specs:
                rows = difference_rows(lookup, ipc, strength, left, right)
                if rows:
                    contrasts.append({"ipc": ipc, "strength": strength, "contrast": name, **summarize(rows)})

    interactions = []
    if 0.7 in strengths:
        for ipc in ipcs:
            for strength in strengths:
                rows = interaction_rows(lookup, ipc, strength)
                if rows:
                    interactions.append({
                        "ipc": ipc, "strength": strength, "reference_strength": 0.7,
                        "interaction": "(correct-label)_strength_minus_(correct-label)_0p7",
                        **summarize(rows),
                    })

    ipc_interactions = []
    if len(ipcs) >= 2:
        low_ipc, high_ipc = min(ipcs), max(ipcs)
        for strength in strengths:
            rows = ipc_interaction_rows(lookup, strength, low_ipc, high_ipc)
            if rows:
                ipc_interactions.append({
                    "strength": strength, "low_ipc": low_ipc, "high_ipc": high_ipc,
                    "interaction": "(correct-label)_high_ipc_minus_(correct-label)_low_ipc",
                    **summarize(rows),
                })
            if 0.7 in strengths:
                rows = ipc_interaction_rows(lookup, strength, low_ipc, high_ipc, reference=0.7)
                if rows:
                    ipc_interactions.append({
                        "strength": strength, "low_ipc": low_ipc, "high_ipc": high_ipc,
                        "interaction": "strength_x_prompt_x_ipc_relative_to_0p7",
                        **summarize(rows),
                    })

    optima = []
    for ipc in ipcs:
        for prompt in sorted({row["prompt"] for row in performance}):
            candidates = [row for row in performance if row["ipc"] == ipc and row["prompt"] == prompt]
            if candidates:
                best = max(candidates, key=lambda row: row["mean"])
                optima.append({"ipc": ipc, "prompt": prompt, "best_strength": best["strength"], "best_mean_accuracy": best["mean"]})

    payload = {
        "format_version": 1,
        "estimand": "paired marginal DCS utility A_correct(strength)-A_label(strength) within matched_ft",
        "bootstrap_order": "training seed -> generation seed -> paired classifier repeat",
        "performance": performance, "contrasts": contrasts, "interactions_relative_to_0p7": interactions,
        "ipc_interactions": ipc_interactions,
        "descriptive_optima": optima,
        "interpretation_boundary": (
            "Strength changes both prototype corruption and effective denoising horizon in StableDiffusionImg2ImgPipeline. "
            "The optimum table is descriptive and is not an unbiased estimate after hyperparameter selection."
        ),
    }
    (output / "strength_interaction_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(output / "cells.csv", cells, (
        "ipc", "strength", "training_seed", "generation_seed", "prompt", "mean_accuracy", "std_accuracy",
        "classifier_accuracies", "source", "synthetic_dir", "evaluation_log",
    ))
    write_csv(output / "performance.csv", performance, (
        "ipc", "strength", "prompt", "mean", "hierarchical_bootstrap_ci_lower", "hierarchical_bootstrap_ci_upper",
        "training_generation_cells", "paired_classifier_observations",
    ))
    write_csv(output / "prompt_utility.csv", contrasts, (
        "ipc", "strength", "contrast", "mean", "hierarchical_bootstrap_ci_lower", "hierarchical_bootstrap_ci_upper",
        "training_generation_cells", "paired_classifier_observations",
    ))
    write_csv(output / "interactions_relative_to_0p7.csv", interactions, (
        "ipc", "strength", "reference_strength", "interaction", "mean", "hierarchical_bootstrap_ci_lower",
        "hierarchical_bootstrap_ci_upper", "training_generation_cells", "paired_classifier_observations",
    ))
    write_csv(output / "ipc_interactions.csv", ipc_interactions, (
        "strength", "low_ipc", "high_ipc", "interaction", "mean", "hierarchical_bootstrap_ci_lower",
        "hierarchical_bootstrap_ci_upper", "training_generation_cells", "paired_classifier_observations",
    ))
    write_csv(output / "descriptive_optima.csv", optima, ("ipc", "prompt", "best_strength", "best_mean_accuracy"))
    plot_summary(performance, contrasts, interactions, output / "strength_interaction_summary.png")
    print(json.dumps({"prompt_utility": contrasts, "interactions": interactions, "ipc_interactions": ipc_interactions}, indent=2, sort_keys=True))


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def plot_summary(performance, contrasts, interactions, destination):
    import matplotlib.pyplot as plt

    ipcs = sorted({row["ipc"] for row in performance})
    figure, axes = plt.subplots(1, len(ipcs) + 1, figsize=(6 * (len(ipcs) + 1), 5), squeeze=False)
    for axis, ipc in zip(axes[0], ipcs):
        for prompt in ("label", "correct", "shuffled"):
            rows = sorted((row for row in performance if row["ipc"] == ipc and row["prompt"] == prompt), key=lambda row: row["strength"])
            if rows:
                axis.plot([row["strength"] for row in rows], [row["mean"] for row in rows], marker="o", label=prompt)
        axis.set_title(f"IPC {ipc}: downstream accuracy")
        axis.set_xlabel("Prototype initialization strength")
        axis.set_ylabel("Validation accuracy")
        axis.legend()
        axis.grid(alpha=0.25)
    axis = axes[0][-1]
    for ipc in ipcs:
        rows = sorted((row for row in contrasts if row["ipc"] == ipc and row["contrast"] == "correct_minus_label"), key=lambda row: row["strength"])
        axis.plot([row["strength"] for row in rows], [row["mean"] for row in rows], marker="o", label=f"IPC {ipc}")
        axis.fill_between(
            [row["strength"] for row in rows],
            [row["hierarchical_bootstrap_ci_lower"] for row in rows],
            [row["hierarchical_bootstrap_ci_upper"] for row in rows], alpha=0.15,
        )
    axis.axhline(0, color="black", linestyle="--", linewidth=1)
    axis.set_title("Marginal utility of Correct DCS")
    axis.set_xlabel("Prototype initialization strength")
    axis.set_ylabel("Correct minus Label accuracy")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.suptitle("Prompt utility is conditional on prototype initialization strength")
    figure.tight_layout()
    figure.savefig(destination, dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
