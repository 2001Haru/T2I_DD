import argparse
import ast
import csv
import json
import re
import statistics
from pathlib import Path

from common import PROMPT_MODES, SUPERVISION_MODES, condition_name


RESULT = re.compile(r"Best, last acc:----(\[[^\]]+\])")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-root", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def parse_log(path):
    matches = RESULT.findall(Path(path).read_text(encoding="utf-8", errors="replace"))
    if not matches:
        raise ValueError(f"No completed result in {path}")
    return [float(value) for value in ast.literal_eval(matches[-1])]


def subtract(left, right):
    if len(left) != len(right):
        raise ValueError("Classifier repeat counts differ")
    return [a - b for a, b in zip(left, right)]


def main():
    args = parse_args()
    root, output = Path(args.evaluation_root), Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cells, contrasts = [], []
    for seed_dir in sorted(root.glob("seed_*"), key=lambda p: int(p.name.split("_")[-1])):
        seed = int(seed_dir.name.split("_")[-1])
        values = {}
        for supervision in SUPERVISION_MODES:
            for prompt in PROMPT_MODES:
                condition = condition_name(supervision, prompt)
                scores = parse_log(seed_dir / f"{condition}.log")
                values[condition] = scores
                cells.append({
                    "generation_seed": seed,
                    "supervision": supervision,
                    "prompt": prompt,
                    "condition": condition,
                    "mean_accuracy": statistics.fmean(scores),
                    "std_accuracy": statistics.pstdev(scores) if len(scores) > 1 else 0.0,
                    "classifier_accuracies": scores,
                })
        current = {}
        for supervision in SUPERVISION_MODES:
            label = values[condition_name(supervision, "label")]
            correct = values[condition_name(supervision, "correct")]
            shuffled = values[condition_name(supervision, "shuffled")]
            current[f"{supervision}_correct_minus_label"] = subtract(correct, label)
            current[f"{supervision}_shuffled_minus_label"] = subtract(shuffled, label)
            current[f"{supervision}_correct_minus_shuffled"] = subtract(correct, shuffled)
        for prompt in PROMPT_MODES:
            current[f"label_ft_minus_frozen_{prompt}"] = subtract(values[condition_name("label_ft", prompt)], values[condition_name("frozen", prompt)])
            current[f"unpaired_minus_label_ft_{prompt}"] = subtract(values[condition_name("unpaired_ft", prompt)], values[condition_name("label_ft", prompt)])
            current[f"matched_minus_unpaired_{prompt}"] = subtract(values[condition_name("matched_ft", prompt)], values[condition_name("unpaired_ft", prompt)])
        current["matching_supervision_x_inference_correspondence"] = subtract(
            current["matched_ft_correct_minus_shuffled"], current["unpaired_ft_correct_minus_shuffled"]
        )
        for name, paired in current.items():
            contrasts.append({
                "generation_seed": seed,
                "contrast": name,
                "mean_difference": statistics.fmean(paired),
                "std_difference": statistics.pstdev(paired) if len(paired) > 1 else 0.0,
                "paired_differences": paired,
            })

    aggregate = {}
    for name in sorted({row["contrast"] for row in contrasts}):
        values = [row["mean_difference"] for row in contrasts if row["contrast"] == name]
        aggregate[name] = {
            "mean_over_generation_seeds": statistics.fmean(values),
            "std_over_generation_seeds": statistics.pstdev(values) if len(values) > 1 else 0.0,
            "generation_seed_values": values,
        }
    payload = {
        "cells": cells,
        "contrasts": contrasts,
        "aggregate_contrasts": aggregate,
        "primary_estimand": "matched_ft minus unpaired_ft under each inference prompt",
        "primary_interaction": "matching_supervision_x_inference_correspondence",
    }
    (output / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for filename, rows, fields in (
        ("cells.csv", cells, ("generation_seed", "supervision", "prompt", "condition", "mean_accuracy", "std_accuracy")),
        ("contrasts.csv", contrasts, ("generation_seed", "contrast", "mean_difference", "std_difference")),
    ):
        with (output / filename).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows({key: row[key] for key in fields} for row in rows)
    print(json.dumps(aggregate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
