#!/usr/bin/env python3
"""Summarize the paired padded-versus-official-variable Label protocol."""

import argparse
import ast
import csv
import json
import random
import re
import statistics
from pathlib import Path


RESULT = re.compile(r"Best, last acc:----(\[[^\]]+\])")
PROMPTS = ("label", "label_var")


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
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(probability * len(ordered)))]


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    index_path = Path(args.evaluation_index).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    by_prompt = {prompt: {} for prompt in PROMPTS}
    performance = []
    for prompt in PROMPTS:
        selected = [row for row in index if row["prompt"] == prompt]
        for row in selected:
            by_prompt[prompt][int(row["generation_seed"])] = scores(row["evaluation_log"])
        flat = [value for values in by_prompt[prompt].values() for value in values]
        performance.append({
            "prompt": prompt,
            "conditioning_protocol": (
                "padded_77" if prompt == "label" else "vlcp_variable_token_aligned"
            ),
            "mean_accuracy": statistics.fmean(flat),
            "generation_cells": len(by_prompt[prompt]),
            "classifier_observations": len(flat),
        })

    generations = sorted(set(by_prompt["label"]) & set(by_prompt["label_var"]))
    paired = {}
    for seed in generations:
        padded = by_prompt["label"][seed]
        variable = by_prompt["label_var"][seed]
        if len(padded) != len(variable):
            raise ValueError(f"Classifier repeat mismatch at generation seed {seed}")
        paired[seed] = [right - left for right, left in zip(variable, padded)]
    observed = [value for seed in generations for value in paired[seed]]
    rng = random.Random(20260814)
    estimates = []
    for _ in range(args.bootstrap_samples):
        values = []
        for _ in generations:
            seed = rng.choice(generations)
            differences = paired[seed]
            values.extend(rng.choice(differences) for _ in differences)
        estimates.append(statistics.fmean(values))
    contrast = [{
        "contrast": "label_var_minus_label_pad",
        "mean_difference": statistics.fmean(observed),
        "bootstrap_ci95_lower": percentile(estimates, 0.025),
        "bootstrap_ci95_upper": percentile(estimates, 0.975),
        "generation_cells": len(generations),
        "paired_classifier_observations": len(observed),
        "bootstrap_order": "shared generation seed -> paired classifier repeat",
    }]

    sequence_rows = []
    run_root = index_path.parent
    for prompt in PROMPTS:
        lengths = []
        for seed in generations:
            records_path = (
                run_root / "synthetic" / f"seed_{seed}" / f"matched_ft_{prompt}"
                / "prompt_records.json"
            )
            records = json.loads(records_path.read_text(encoding="utf-8"))
            lengths.extend(int(row["conditioning_sequence_length"]) for row in records)
        sequence_rows.append({
            "prompt": prompt,
            "sequence_length_mean": statistics.fmean(lengths),
            "sequence_length_min": min(lengths),
            "sequence_length_max": max(lengths),
            "records": len(lengths),
        })

    write_csv(output / "performance.csv", performance)
    write_csv(output / "paired_contrast.csv", contrast)
    write_csv(output / "conditioning_lengths.csv", sequence_rows)
    (output / "summary.json").write_text(json.dumps({
        "format_version": 1,
        "performance": performance,
        "primary": contrast[0],
        "conditioning_lengths": sequence_rows,
        "interpretation_boundary": (
            "Both arms use identical text, visual prototypes, image seeds, generation seeds, "
            "and classifier repeats. Only the SD1.5 conditioning sequence-length protocol differs."
        ),
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
