#!/usr/bin/env python3
"""Summarize the A/B/C matrix without treating shuffle shifts as independent samples."""

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


def mean_vectors(vectors):
    if not vectors or len({len(values) for values in vectors}) != 1:
        raise ValueError("Cannot average absent or unequal classifier vectors")
    return [statistics.fmean(values) for values in zip(*vectors)]


def bootstrap(rows, samples=10000, seed=20260809):
    grouped = {}
    for row in rows:
        grouped.setdefault(str(row["training_seed"]), {}).setdefault(row["generation_seed"], []).append(row["values"])
    rng = random.Random(seed)
    training_seeds = list(grouped)
    estimates = []
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
    values = [value for row in rows for value in row["values"]]
    lower, upper = bootstrap(rows)
    return {
        "mean": statistics.fmean(values),
        "hierarchical_bootstrap_ci_lower": lower,
        "hierarchical_bootstrap_ci_upper": upper,
        "training_generation_cells": len(rows),
        "paired_classifier_observations": len(values),
    }


def visual_label(row):
    return "pure_noise" if row["visual_mode"] == "pure_noise" else f"strength_{float(row['strength']):g}"


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


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
        row = {
            **item, "visual": visual_label(item), "mean_accuracy": statistics.fmean(scores),
            "std_accuracy": statistics.pstdev(scores), "classifier_accuracies": scores,
        }
        cells.append(row)
        shift = int(item.get("shuffle_shift") or 1) if item["prompt"] == "shuffled" else None
        key = (
            item["matrix"], item["spec"], int(item["ipc"]), row["visual"], item["supervision"],
            item.get("training_seed"), int(item["generation_seed"]), item["prompt"], shift,
        )
        if key in lookup:
            raise RuntimeError(f"Duplicate evaluation cell: {key}")
        lookup[key] = scores

    groups = sorted({key[:7] for key in lookup}, key=str)
    normalized = []
    for group in groups:
        prefix = group
        label = lookup.get((*prefix, "label", None))
        correct = lookup.get((*prefix, "correct", None))
        shifts = sorted(key[-1] for key in lookup if key[:7] == prefix and key[7] == "shuffled")
        shuffled_vectors = [lookup[(*prefix, "shuffled", shift)] for shift in shifts]
        shuffled = mean_vectors(shuffled_vectors) if shuffled_vectors else None
        shuffled_primary = lookup.get((*prefix, "shuffled", 1))
        metadata = {
            "matrix": prefix[0], "spec": prefix[1], "ipc": prefix[2], "visual": prefix[3],
            "supervision": prefix[4], "training_seed": prefix[5], "generation_seed": prefix[6],
            "shuffle_shifts": shifts,
        }
        if label is not None:
            normalized.append({**metadata, "prompt": "label", "values": label})
        if correct is not None:
            normalized.append({**metadata, "prompt": "correct", "values": correct})
        if shuffled is not None:
            normalized.append({**metadata, "prompt": "shuffled_mean", "values": shuffled})
        if shuffled_primary is not None:
            normalized.append({**metadata, "prompt": "shuffled_s1", "values": shuffled_primary})

    performance = []
    performance_keys = sorted({
        (row["matrix"], row["spec"], row["ipc"], row["visual"], row["supervision"], row["prompt"])
        for row in normalized
    }, key=str)
    for key in performance_keys:
        rows = [
            {"training_seed": row["training_seed"], "generation_seed": row["generation_seed"], "values": row["values"]}
            for row in normalized
            if (row["matrix"], row["spec"], row["ipc"], row["visual"], row["supervision"], row["prompt"]) == key
        ]
        performance.append({
            "matrix": key[0], "spec": key[1], "ipc": key[2], "visual": key[3],
            "supervision": key[4], "prompt": key[5], **summarize(rows),
        })

    normalized_lookup = {
        (
            row["matrix"], row["spec"], row["ipc"], row["visual"], row["supervision"],
            row["training_seed"], row["generation_seed"], row["prompt"],
        ): row["values"]
        for row in normalized
    }
    contrasts = []
    contrast_specs = (
        ("correct_minus_label", "correct", "label"),
        ("shuffled_s1_minus_label", "shuffled_s1", "label"),
        ("correct_minus_shuffled_s1", "correct", "shuffled_s1"),
        ("shuffled_mean_minus_label_robustness", "shuffled_mean", "label"),
        ("correct_minus_shuffled_mean_robustness", "correct", "shuffled_mean"),
    )
    contrast_groups = sorted({key[:5] for key in normalized_lookup}, key=str)
    for group in contrast_groups:
        pairs = sorted({key[5:7] for key in normalized_lookup if key[:5] == group}, key=str)
        for name, left, right in contrast_specs:
            rows = []
            for training_seed, generation_seed in pairs:
                left_key = (*group, training_seed, generation_seed, left)
                right_key = (*group, training_seed, generation_seed, right)
                if left_key in normalized_lookup and right_key in normalized_lookup:
                    rows.append({
                        "training_seed": training_seed, "generation_seed": generation_seed,
                        "values": paired(normalized_lookup[left_key], normalized_lookup[right_key]),
                    })
            if rows:
                contrasts.append({
                    "matrix": group[0], "spec": group[1], "ipc": group[2], "visual": group[3],
                    "supervision": group[4], "contrast": name, **summarize(rows),
                })

    shift_effects = []
    raw_groups = sorted({key[:7] for key in lookup if key[7] == "shuffled"}, key=str)
    for group in raw_groups:
        correct = lookup.get((*group, "correct", None))
        label = lookup.get((*group, "label", None))
        if correct is None or label is None:
            continue
        for shift in sorted(key[-1] for key in lookup if key[:7] == group and key[7] == "shuffled"):
            shuffled = lookup[(*group, "shuffled", shift)]
            for name, values in (
                ("shuffled_minus_label", paired(shuffled, label)),
                ("correct_minus_shuffled", paired(correct, shuffled)),
            ):
                shift_effects.append({
                    "matrix": group[0], "spec": group[1], "ipc": group[2], "visual": group[3],
                    "supervision": group[4], "training_seed": group[5], "generation_seed": group[6],
                    "shuffle_shift": shift, "contrast": name,
                    "mean_paired_difference": statistics.fmean(values), "paired_differences": values,
                })

    payload = {
        "format_version": 1,
        "estimands": {
            "prompt_marginal_primary": "shuffle shift 1-label at every visual setting",
            "correspondence_primary": "correct-shuffle shift 1 at every visual setting",
            "prompt_marginal_robustness": "mean(shuffled shifts)-label, with shifts averaged before bootstrap",
            "correspondence_robustness": "correct-mean(shuffled shifts), with shifts averaged before bootstrap",
            "correct_utility": "correct-label",
        },
        "bootstrap_order": "training seed -> generation seed -> paired classifier repeat",
        "performance": performance, "contrasts": contrasts,
        "interpretation_boundary": (
            "Strength changes prototype corruption and effective denoising horizon. Pure-noise is a distinct "
            "text-to-image interface. Shuffle shifts are randomization realizations, not independent experimental units."
        ),
    }
    (output / "conditioning_interface_matrix_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_csv(output / "cells.csv", cells, (
        "matrix", "spec", "ipc", "visual", "supervision", "training_seed", "generation_seed",
        "prompt", "shuffle_shift", "mean_accuracy", "std_accuracy", "classifier_accuracies",
        "source", "evaluation_log",
    ))
    write_csv(output / "performance.csv", performance, (
        "matrix", "spec", "ipc", "visual", "supervision", "prompt", "mean",
        "hierarchical_bootstrap_ci_lower", "hierarchical_bootstrap_ci_upper",
        "training_generation_cells", "paired_classifier_observations",
    ))
    write_csv(output / "contrasts.csv", contrasts, (
        "matrix", "spec", "ipc", "visual", "supervision", "contrast", "mean",
        "hierarchical_bootstrap_ci_lower", "hierarchical_bootstrap_ci_upper",
        "training_generation_cells", "paired_classifier_observations",
    ))
    write_csv(output / "shuffle_shift_effects.csv", shift_effects, (
        "matrix", "spec", "ipc", "visual", "supervision", "training_seed", "generation_seed",
        "shuffle_shift", "contrast", "mean_paired_difference", "paired_differences",
    ))
    plot(performance, contrasts, output / "conditioning_interface_matrix_summary.png")
    print(json.dumps({"performance_rows": len(performance), "contrast_rows": len(contrasts)}, indent=2))


def plot(performance, contrasts, destination):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(18, 5))
    for axis, matrix in zip(axes, ("A", "B", "C")):
        rows = [
            row for row in contrasts
            if row["matrix"] == matrix and row["contrast"] in {
                "correct_minus_label", "shuffled_s1_minus_label", "correct_minus_shuffled_s1"
            }
        ]
        for (ipc, contrast), selected in _group(rows, lambda row: (row["ipc"], row["contrast"])).items():
            selected = sorted(selected, key=lambda row: _visual_order(row["visual"]))
            axis.plot(
                [_visual_order(row["visual"]) for row in selected], [row["mean"] for row in selected],
                marker="o", label=f"IPC{ipc} {contrast}",
            )
        axis.axhline(0, color="black", linestyle="--", linewidth=1)
        axis.set_title(f"Matrix {matrix}")
        axis.set_xlabel("Visual interface (pure noise=-1; otherwise strength)")
        axis.set_ylabel("Paired accuracy difference")
        axis.grid(alpha=0.25)
        if rows:
            axis.legend(fontsize=7)
    figure.suptitle("Prompt utility across conditioning interfaces")
    figure.tight_layout()
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def _group(rows, key_fn):
    grouped = {}
    for row in rows:
        grouped.setdefault(key_fn(row), []).append(row)
    return grouped


def _visual_order(value):
    return -1.0 if value == "pure_noise" else float(value.split("_", 1)[1])


if __name__ == "__main__":
    main()
