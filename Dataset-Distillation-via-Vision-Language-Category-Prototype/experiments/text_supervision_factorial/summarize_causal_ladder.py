import argparse
import ast
import csv
import json
import re
import random
import statistics
from pathlib import Path


RESULT = re.compile(r"Best, last acc:----(\[[^\]]+\])")
PROMPTS = ("label", "correct", "shuffled")
ROWS = ("empty_ft", "constant_ft", "label_ft", "unpaired_ft", "matched_ft")


def scores(path):
    matches = RESULT.findall(path.read_text(encoding="utf-8", errors="replace"))
    if not matches:
        raise ValueError(f"No completed classifier result: {path}")
    return [float(value) for value in ast.literal_eval(matches[-1])]


def paired(left, right):
    if len(left) != len(right):
        raise ValueError("Classifier repeat counts differ")
    return [a - b for a, b in zip(left, right)]


def hierarchical_bootstrap(rows, samples=10000, seed=20260807):
    """Resample training seeds, generation seeds, then paired classifier repeats."""
    grouped = {}
    for row in rows:
        grouped.setdefault(row["training_seed"], {})[row["generation_seed"]] = row["paired_differences"]
    training_seeds = sorted(grouped)
    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        drawn = []
        for _ in training_seeds:
            training_seed = rng.choice(training_seeds)
            generation = grouped[training_seed]
            generation_seeds = sorted(generation)
            for _ in generation_seeds:
                generation_seed = rng.choice(generation_seeds)
                differences = generation[generation_seed]
                drawn.extend(rng.choice(differences) for _ in differences)
        estimates.append(statistics.fmean(drawn))
    estimates.sort()
    return estimates[int(0.025 * samples)], estimates[min(samples - 1, int(0.975 * samples))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-evaluation-root", required=True)
    parser.add_argument("--extension-evaluation-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    base = Path(args.base_evaluation_root)
    extension = Path(args.extension_evaluation_root)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cells = []
    values = {}

    generation_seeds = sorted(int(path.name.split("_")[-1]) for path in base.glob("seed_*"))
    for generation_seed in generation_seeds:
        for prompt in PROMPTS:
            frozen = scores(base / f"seed_{generation_seed}" / f"frozen_{prompt}.log")
            values[(None, generation_seed, "frozen", prompt)] = frozen
            cells.append({
                "training_seed": "", "generation_seed": generation_seed, "supervision": "frozen",
                "prompt": prompt, "mean_accuracy": statistics.fmean(frozen),
                "std_accuracy": statistics.pstdev(frozen), "classifier_accuracies": frozen,
            })
        for training_seed in (0, 1):
            for row in ROWS:
                root = base if training_seed == 0 and row in ("label_ft", "unpaired_ft", "matched_ft") else extension / f"train_seed_{training_seed}"
                seed_dir = root / f"seed_{generation_seed}"
                for prompt in PROMPTS:
                    current = scores(seed_dir / f"{row}_{prompt}.log")
                    values[(training_seed, generation_seed, row, prompt)] = current
                    cells.append({
                        "training_seed": training_seed, "generation_seed": generation_seed,
                        "supervision": row, "prompt": prompt,
                        "mean_accuracy": statistics.fmean(current),
                        "std_accuracy": statistics.pstdev(current), "classifier_accuracies": current,
                    })

    contrasts = []
    comparisons = (
        ("empty_minus_frozen", "empty_ft", "frozen"),
        ("constant_minus_empty", "constant_ft", "empty_ft"),
        ("label_minus_empty", "label_ft", "empty_ft"),
        ("label_minus_constant", "label_ft", "constant_ft"),
        ("unpaired_minus_label", "unpaired_ft", "label_ft"),
        ("matched_minus_unpaired", "matched_ft", "unpaired_ft"),
    )
    for training_seed in (0, 1):
        for generation_seed in generation_seeds:
            for name, left, right in comparisons:
                for prompt in PROMPTS:
                    left_scores = values[(training_seed, generation_seed, left, prompt)]
                    right_seed = None if right == "frozen" else training_seed
                    differences = paired(left_scores, values[(right_seed, generation_seed, right, prompt)])
                    contrasts.append({
                        "training_seed": training_seed, "generation_seed": generation_seed,
                        "contrast": f"{name}_{prompt}", "mean_difference": statistics.fmean(differences),
                        "std_difference": statistics.pstdev(differences), "paired_differences": differences,
                    })
                left_dcs = values[(training_seed, generation_seed, left, "correct")] + values[(training_seed, generation_seed, left, "shuffled")]
                right_seed = None if right == "frozen" else training_seed
                right_dcs = values[(right_seed, generation_seed, right, "correct")] + values[(right_seed, generation_seed, right, "shuffled")]
                differences = paired(left_dcs, right_dcs)
                contrasts.append({
                    "training_seed": training_seed, "generation_seed": generation_seed,
                    "contrast": f"{name}_descriptive_average",
                    "mean_difference": statistics.fmean(differences),
                    "std_difference": statistics.pstdev(differences), "paired_differences": differences,
                })

    aggregate = {}
    for name in sorted({row["contrast"] for row in contrasts}):
        selected_rows = [row for row in contrasts if row["contrast"] == name]
        selected = [row["mean_difference"] for row in selected_rows]
        lower, upper = hierarchical_bootstrap(selected_rows)
        aggregate[name] = {
            "mean_over_training_and_generation_seeds": statistics.fmean(selected),
            "std_over_training_and_generation_seeds": statistics.pstdev(selected),
            "hierarchical_bootstrap_ci_lower": lower,
            "hierarchical_bootstrap_ci_upper": upper,
            "bootstrap_samples": 10000,
            "bootstrap_order": "training_seed -> generation_seed -> paired_classifier_repeat",
            "values": selected,
        }
    payload = {
        "causal_ladder": ["frozen", "empty_ft", "constant_ft", "label_ft", "unpaired_ft", "matched_ft"],
        "estimands": {
            "empty_minus_frozen": "target-domain image adaptation under empty conditioning",
            "constant_minus_empty": "generic conditioning style beyond the special empty embedding",
            "label_minus_empty": "class-language supervision beyond image adaptation",
            "unpaired_minus_label": "rich within-class text marginal",
            "matched_minus_unpaired": "instance-level image-caption correspondence",
        },
        "cells": cells, "contrasts": contrasts, "aggregate_contrasts": aggregate,
    }
    (output / "causal_ladder_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for filename, records, fields in (
        ("causal_ladder_cells.csv", cells, ("training_seed", "generation_seed", "supervision", "prompt", "mean_accuracy", "std_accuracy")),
        ("causal_ladder_contrasts.csv", contrasts, ("training_seed", "generation_seed", "contrast", "mean_difference", "std_difference")),
    ):
        with (output / filename).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows({field: row[field] for field in fields} for row in records)
    print(json.dumps(aggregate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
