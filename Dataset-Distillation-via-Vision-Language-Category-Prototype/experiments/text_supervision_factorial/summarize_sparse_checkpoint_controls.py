#!/usr/bin/env python3
"""Compare sparse prompt-bank checkpoints with existing causal-ladder checkpoints."""

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
SUPERVISIONS = ("empty_ft", "label_ft", "unpaired_ft")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-index", required=True)
    parser.add_argument("--sparse-index", required=True)
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


def grouped(rows, checkpoint_key):
    result = {}
    for row in rows:
        result.setdefault(row[checkpoint_key], {})[row["generation_seed"]] = row["scores"]
    return result


def hierarchical_mean_ci(rows, checkpoint_key, samples, seed):
    groups = grouped(rows, checkpoint_key)
    checkpoints = sorted(groups)
    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        values = []
        for _ in checkpoints:
            checkpoint = rng.choice(checkpoints)
            generations = groups[checkpoint]
            generation_seeds = sorted(generations)
            for _ in generation_seeds:
                generation_seed = rng.choice(generation_seeds)
                repeats = generations[generation_seed]
                values.extend(rng.choice(repeats) for _ in repeats)
        estimates.append(statistics.fmean(values))
    return percentile(estimates, 0.025), percentile(estimates, 0.975)


def independent_checkpoint_contrast(sparse_rows, control_rows, samples, seed):
    sparse = grouped(sparse_rows, "bank_seed")
    control = grouped(control_rows, "training_seed")
    sparse_checkpoints = sorted(sparse)
    control_checkpoints = sorted(control)
    common_generations = sorted(
        set.intersection(
            *(set(value) for value in list(sparse.values()) + list(control.values()))
        )
    )
    if not common_generations:
        raise ValueError("Sparse and control rows have no common generation seeds")
    rng = random.Random(seed)
    estimates = []
    draws = max(len(sparse_checkpoints), len(control_checkpoints))
    for _ in range(samples):
        values = []
        for _ in range(draws):
            sparse_checkpoint = rng.choice(sparse_checkpoints)
            control_checkpoint = rng.choice(control_checkpoints)
            for _ in common_generations:
                generation_seed = rng.choice(common_generations)
                left = sparse[sparse_checkpoint][generation_seed]
                right = control[control_checkpoint][generation_seed]
                repeats = min(len(left), len(right))
                for _ in range(repeats):
                    repeat = rng.randrange(repeats)
                    values.append(left[repeat] - right[repeat])
        estimates.append(statistics.fmean(values))
    point = statistics.fmean([value for row in sparse_rows for value in row["scores"]]) - statistics.fmean(
        [value for row in control_rows for value in row["scores"]]
    )
    return point, percentile(estimates, 0.025), percentile(estimates, 0.975)


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    control_index = json.loads(Path(args.control_index).read_text(encoding="utf-8"))
    sparse_index = json.loads(Path(args.sparse_index).read_text(encoding="utf-8"))
    controls = [{**row, "scores": scores(row["evaluation_log"])} for row in control_index]
    sparse = [
        {**row, "scores": scores(row["evaluation_log"])}
        for row in sparse_index if row["prompt"] == "bank"
    ]

    performance = []
    for supervision in sorted({row["supervision"] for row in controls}):
        selected = [row for row in controls if row["supervision"] == supervision]
        lower, upper = hierarchical_mean_ci(
            selected, "training_seed", args.bootstrap_samples, 20260813 + len(performance)
        )
        performance.append({
            "family": "checkpoint_control",
            "method": supervision,
            "caption_budget": {
                "empty_ft": "empty",
                "label_ft": "class_label",
                "unpaired_ft": "full_unpaired",
                "matched_ft": "full_matched",
            }[supervision],
            "mean_accuracy": statistics.fmean([value for row in selected for value in row["scores"]]),
            "bootstrap_ci_lower": lower,
            "bootstrap_ci_upper": upper,
            "checkpoint_generation_cells": len(selected),
            "classifier_observations": sum(len(row["scores"]) for row in selected),
        })
    for budget in sorted({row["budget"] for row in sparse}):
        selected = [row for row in sparse if row["budget"] == budget]
        lower, upper = hierarchical_mean_ci(
            selected, "bank_seed", args.bootstrap_samples, 20260900 + budget
        )
        performance.append({
            "family": "sparse_unpaired",
            "method": f"sparse_m{budget}",
            "caption_budget": budget,
            "mean_accuracy": statistics.fmean([value for row in selected for value in row["scores"]]),
            "bootstrap_ci_lower": lower,
            "bootstrap_ci_upper": upper,
            "checkpoint_generation_cells": len(selected),
            "classifier_observations": sum(len(row["scores"]) for row in selected),
        })

    contrasts = []
    for budget in sorted({row["budget"] for row in sparse}):
        sparse_rows = [row for row in sparse if row["budget"] == budget]
        for supervision in SUPERVISIONS:
            control_rows = [row for row in controls if row["supervision"] == supervision]
            mean, lower, upper = independent_checkpoint_contrast(
                sparse_rows, control_rows, args.bootstrap_samples,
                seed=20261000 + budget * 10 + list(SUPERVISIONS).index(supervision),
            )
            contrasts.append({
                "budget": budget,
                "contrast": f"sparse_m{budget}_minus_{supervision}_label",
                "mean_difference": mean,
                "bootstrap_ci_lower": lower,
                "bootstrap_ci_upper": upper,
                "noninferior_within_1pt_by_95pct_ci": lower >= -1.0 - 1e-12,
                "bootstrap_order": (
                    "independent checkpoint realization -> shared generation seed -> paired classifier repeat"
                ),
            })

    write_csv(output / "checkpoint_and_sparse_performance.csv", performance)
    write_csv(output / "sparse_vs_checkpoint_controls.csv", contrasts)
    summary = {
        "format_version": 1,
        "performance": performance,
        "contrasts": contrasts,
        "primary_question": (
            "Does sparse unpaired-caption fine-tuning improve over Label-FT, and how much of the "
            "full Unpaired-FT checkpoint is retained?"
        ),
        "boundary": (
            "Sparse bank seeds change caption selection while retaining training seed 0; control checkpoint "
            "seeds change optimization initialization. Contrasts therefore bootstrap the two checkpoint "
            "realization axes independently and pair only generation seed and classifier repeat."
        ),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    order = ("empty_ft", "label_ft", "unpaired_ft", "sparse_m4", "sparse_m8", "sparse_m16", "sparse_m32")
    lookup = {row["method"]: row for row in performance}
    rows = [lookup[key] for key in order if key in lookup]
    fig, axis = plt.subplots(figsize=(12, 5.5))
    axis.errorbar(
        range(len(rows)), [row["mean_accuracy"] for row in rows],
        yerr=[
            [row["mean_accuracy"] - row["bootstrap_ci_lower"] for row in rows],
            [row["bootstrap_ci_upper"] - row["mean_accuracy"] for row in rows],
        ], fmt="o", capsize=5,
    )
    axis.set_xticks(range(len(rows)), [row["method"] for row in rows], rotation=25, ha="right")
    axis.set_ylabel("Validation accuracy")
    axis.set_title("Sparse caption supervision versus existing checkpoint controls")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "sparse_checkpoint_controls.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
