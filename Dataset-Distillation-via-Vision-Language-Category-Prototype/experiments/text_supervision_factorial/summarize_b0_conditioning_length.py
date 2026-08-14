#!/usr/bin/env python3
"""Summarize B-0 conditioning-content and sequence-length controls."""

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
FAMILIES = ("label_ft", "matched_ft", "sparse_m4_ft")
PROMPTS = ("label", "first_sentence", "correct_t77", "correct", "label_pad_dcs")
CONTRASTS = (
    ("first_sentence_minus_label", "first_sentence", "label"),
    ("correct_t77_minus_label", "correct_t77", "label"),
    ("full_minus_t77", "correct", "correct_t77"),
    ("label_pad_dcs_minus_label", "label_pad_dcs", "label"),
    ("full_minus_label_pad_dcs", "correct", "label_pad_dcs"),
    ("full_minus_first_sentence", "correct", "first_sentence"),
)


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


def bootstrap_generation(rows, value_key, samples, seed):
    grouped = {row["generation_seed"]: row[value_key] for row in rows}
    generations = sorted(grouped)
    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        values = []
        for _ in generations:
            generation = rng.choice(generations)
            repeats = grouped[generation]
            values.extend(rng.choice(repeats) for _ in repeats)
        estimates.append(statistics.fmean(values))
    return (
        percentile(estimates, 0.025), percentile(estimates, 0.975),
        percentile(estimates, 0.05), percentile(estimates, 0.95),
    )


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def conditioning_audit(index_path, rows):
    run_root = Path(index_path).resolve().parent
    audit = []
    for family in FAMILIES:
        supervision = "sparse_ft" if family == "sparse_m4_ft" else family
        for generation_seed in sorted({row["generation_seed"] for row in rows}):
            for prompt in PROMPTS:
                path = (
                    run_root / "synthetic" / family / f"seed_{generation_seed}"
                    / f"{supervision}_{prompt}" / "prompt_records.json"
                )
                records = json.loads(path.read_text(encoding="utf-8"))
                lengths = [int(row["conditioning_sequence_length"]) for row in records]
                reference = [int(row["reference_dcs_chunks"]) for row in records]
                audit.append({
                    "checkpoint_family": family,
                    "generation_seed": generation_seed,
                    "prompt": prompt,
                    "records": len(records),
                    "sequence_length_mean": statistics.fmean(lengths),
                    "sequence_length_min": min(lengths),
                    "sequence_length_max": max(lengths),
                    "reference_dcs_chunks_mean": statistics.fmean(reference),
                    "length_matches_reference_fraction": statistics.fmean(
                        length == chunks * 77 for length, chunks in zip(lengths, reference)
                    ),
                })
    return audit


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
        lookup[(row["checkpoint_family"], row["generation_seed"], row["prompt"])] = values
    expected = {
        (family, seed, prompt)
        for family in FAMILIES
        for seed in sorted({row["generation_seed"] for row in cells})
        for prompt in PROMPTS
    }
    missing = sorted(expected - set(lookup))
    if missing:
        raise RuntimeError(f"Incomplete B-0 matrix; missing {missing}")

    performance = []
    for family in FAMILIES:
        for prompt in PROMPTS:
            selected = [
                row for row in cells
                if row["checkpoint_family"] == family and row["prompt"] == prompt
            ]
            lower, upper, _, _ = bootstrap_generation(
                selected, "scores", args.bootstrap_samples, 20260901
            )
            values = [value for row in selected for value in row["scores"]]
            performance.append({
                "checkpoint_family": family, "prompt": prompt,
                "mean_accuracy": statistics.fmean(values),
                "bootstrap_ci95_lower": lower, "bootstrap_ci95_upper": upper,
                "generation_cells": len(selected), "classifier_observations": len(values),
            })

    contrasts = []
    for family_index, family in enumerate(FAMILIES):
        for contrast_index, (name, left, right) in enumerate(CONTRASTS):
            paired = []
            for generation_seed in sorted({row["generation_seed"] for row in cells}):
                a = lookup[(family, generation_seed, left)]
                b = lookup[(family, generation_seed, right)]
                paired.append({
                    "generation_seed": generation_seed,
                    "differences": [x - y for x, y in zip(a, b)],
                })
            lower, upper, lower90, upper90 = bootstrap_generation(
                paired, "differences", args.bootstrap_samples,
                20260910 + family_index * 20 + contrast_index,
            )
            values = [value for row in paired for value in row["differences"]]
            contrasts.append({
                "checkpoint_family": family, "contrast": name,
                "mean_difference": statistics.fmean(values),
                "bootstrap_ci95_lower": lower, "bootstrap_ci95_upper": upper,
                "bootstrap_ci90_lower": lower90, "bootstrap_ci90_upper": upper90,
                "equivalent_within_1pt_by_90pct_ci": (
                    lower90 >= -args.equivalence_margin and upper90 <= args.equivalence_margin
                ),
                "noninferior_within_1pt_by_95pct_ci": lower >= -args.equivalence_margin,
                "generation_cells": len(paired), "paired_classifier_observations": len(values),
            })

    audit = conditioning_audit(args.evaluation_index, cells)
    for family in FAMILIES:
        family_audit = [row for row in audit if row["checkpoint_family"] == family]
        for seed in sorted({row["generation_seed"] for row in family_audit}):
            by_prompt = {
                row["prompt"]: row for row in family_audit if row["generation_seed"] == seed
            }
            if by_prompt["label"]["sequence_length_max"] != 77:
                raise RuntimeError("Label condition is not one 77-position block")
            if by_prompt["first_sentence"]["sequence_length_max"] != 77:
                raise RuntimeError("First-sentence condition is not one 77-position block")
            if by_prompt["correct_t77"]["sequence_length_max"] != 77:
                raise RuntimeError("T77 condition is not one 77-position block")
            if by_prompt["correct"]["length_matches_reference_fraction"] != 1.0:
                raise RuntimeError("Full DCS does not match its reference chunk count")
            if by_prompt["label_pad_dcs"]["length_matches_reference_fraction"] != 1.0:
                raise RuntimeError("Padded Label does not match its reference DCS chunk count")

    write_csv(output / "performance.csv", performance)
    write_csv(output / "paired_contrasts.csv", contrasts)
    write_csv(output / "conditioning_length_audit.csv", audit)
    summary = {
        "format_version": 1,
        "performance": performance,
        "paired_contrasts": contrasts,
        "conditioning_length_audit": audit,
        "bootstrap_order": "generation seed -> shared paired classifier-repeat draw",
        "interpretation": {
            "full_minus_t77_near_zero": "extra DCS chunks add no downstream value",
            "label_pad_dcs_minus_label_positive": "extra padding-derived KV positions have structural value",
            "full_minus_label_pad_dcs_positive": "DCS content beyond length alone has value",
            "full_minus_first_sentence_near_zero": "the first DCS sentence is sufficient",
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    x = list(range(len(PROMPTS)))
    for family in FAMILIES:
        rows = [
            next(row for row in performance if row["checkpoint_family"] == family and row["prompt"] == prompt)
            for prompt in PROMPTS
        ]
        axes[0].errorbar(
            x, [row["mean_accuracy"] for row in rows],
            yerr=[
                [row["mean_accuracy"] - row["bootstrap_ci95_lower"] for row in rows],
                [row["bootstrap_ci95_upper"] - row["mean_accuracy"] for row in rows],
            ], marker="o", capsize=4, label=family,
        )
    axes[0].set_xticks(x, PROMPTS, rotation=18)
    axes[0].set_ylabel("Validation accuracy")
    axes[0].set_title("Content and sequence-length controls")
    axes[0].legend()
    key_names = ("correct_t77_minus_label", "full_minus_t77", "label_pad_dcs_minus_label")
    positions = list(range(len(key_names)))
    width = 0.24
    for family_index, family in enumerate(FAMILIES):
        rows = [
            next(row for row in contrasts if row["checkpoint_family"] == family and row["contrast"] == name)
            for name in key_names
        ]
        axes[1].bar(
            [position + (family_index - 1) * width for position in positions],
            [row["mean_difference"] for row in rows], width=width, label=family,
        )
    axes[1].set_xticks(positions, ("T77-Label", "Full-T77", "Pad-Label"))
    axes[1].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[1].set_ylabel("Paired accuracy difference")
    axes[1].set_title("Semantic content versus extra KV positions")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output / "b0_conditioning_length.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
