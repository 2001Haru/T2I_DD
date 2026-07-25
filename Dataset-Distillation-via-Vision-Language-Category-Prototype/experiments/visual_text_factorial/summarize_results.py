import argparse
import ast
import csv
import json
import re
import statistics
from pathlib import Path

from common import condition_matrix


CONDITIONS = tuple(item["condition"] for item in condition_matrix())
RESULT_PATTERN = re.compile(r"Best, last acc:----(\[[^\]]+\])")


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize the visual x text factorial")
    parser.add_argument("--evaluation-root", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def parse_log(path):
    matches = RESULT_PATTERN.findall(Path(path).read_text(encoding="utf-8", errors="replace"))
    if not matches:
        raise ValueError(f"No completed Minimax result found in {path}")
    values = [float(value) for value in ast.literal_eval(matches[-1])]
    if not values:
        raise ValueError(f"Empty result list in {path}")
    return values


def mean(values):
    return statistics.fmean(values)


def paired_subtract(left, right):
    if len(left) != len(right):
        raise ValueError("Paired classifier result lengths differ")
    return [left_value - right_value for left_value, right_value in zip(left, right)]


def condition_contrasts(values):
    contrasts = {}
    for visual_mode in ("no_visual", "prototype"):
        label = values[f"{visual_mode}_label"]
        dcs = values[f"{visual_mode}_dcs"]
        shuffled = values[f"{visual_mode}_dcs_shuffled"]
        contrasts[f"{visual_mode}_dcs_minus_label"] = paired_subtract(dcs, label)
        contrasts[f"{visual_mode}_shuffled_minus_label"] = paired_subtract(shuffled, label)
        contrasts[f"{visual_mode}_dcs_minus_shuffled"] = paired_subtract(dcs, shuffled)

    for prompt_mode in ("label", "dcs", "dcs_shuffled"):
        contrasts[f"prototype_minus_no_visual_{prompt_mode}"] = paired_subtract(
            values[f"prototype_{prompt_mode}"],
            values[f"no_visual_{prompt_mode}"],
        )
    contrasts["visual_x_dcs_label_interaction"] = paired_subtract(
        contrasts["prototype_dcs_minus_label"],
        contrasts["no_visual_dcs_minus_label"],
    )
    contrasts["visual_x_cluster_correspondence_interaction"] = paired_subtract(
        contrasts["prototype_dcs_minus_shuffled"],
        contrasts["no_visual_dcs_minus_shuffled"],
    )
    return contrasts


def main():
    args = parse_args()
    evaluation_root = Path(args.evaluation_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    condition_rows = []
    contrast_rows = []

    seed_dirs = sorted(
        evaluation_root.glob("seed_*"), key=lambda path: int(path.name.split("_")[-1])
    )
    if not seed_dirs:
        raise FileNotFoundError(f"No seed_* directories found in {evaluation_root}")
    for seed_dir in seed_dirs:
        generation_seed = int(seed_dir.name.split("_")[-1])
        values = {}
        for condition in CONDITIONS:
            accuracies = parse_log(seed_dir / f"{condition}.log")
            values[condition] = accuracies
            condition_rows.append(
                {
                    "generation_seed": generation_seed,
                    "condition": condition,
                    "classifier_accuracies": accuracies,
                    "mean_accuracy": mean(accuracies),
                    "std_accuracy": statistics.pstdev(accuracies)
                    if len(accuracies) > 1
                    else 0.0,
                }
            )
        for name, paired_values in condition_contrasts(values).items():
            contrast_rows.append(
                {
                    "generation_seed": generation_seed,
                    "contrast": name,
                    "paired_classifier_differences": paired_values,
                    "mean_difference": mean(paired_values),
                    "std_difference": statistics.pstdev(paired_values)
                    if len(paired_values) > 1
                    else 0.0,
                }
            )

    aggregate = {}
    for name in sorted({row["contrast"] for row in contrast_rows}):
        seed_means = [
            row["mean_difference"] for row in contrast_rows if row["contrast"] == name
        ]
        aggregate[name] = {
            "mean_over_generation_seeds": mean(seed_means),
            "std_over_generation_seeds": statistics.pstdev(seed_means)
            if len(seed_means) > 1
            else 0.0,
            "generation_seed_values": seed_means,
        }

    payload = {
        "conditions": condition_rows,
        "contrasts": contrast_rows,
        "aggregate_contrasts": aggregate,
        "primary_estimand": "dcs_minus_dcs_shuffled within each visual mode",
        "primary_interaction": "visual_x_cluster_correspondence_interaction",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    condition_fields = (
        "generation_seed",
        "condition",
        "mean_accuracy",
        "std_accuracy",
    )
    with (output_dir / "conditions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=condition_fields)
        writer.writeheader()
        for row in condition_rows:
            writer.writerow({key: row[key] for key in condition_fields})

    contrast_fields = (
        "generation_seed",
        "contrast",
        "mean_difference",
        "std_difference",
    )
    with (output_dir / "contrasts.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=contrast_fields)
        writer.writeheader()
        for row in contrast_rows:
            writer.writerow({key: row[key] for key in contrast_fields})

    print(json.dumps(aggregate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
