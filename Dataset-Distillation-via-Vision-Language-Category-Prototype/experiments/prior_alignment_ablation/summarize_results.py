import argparse
import ast
import csv
import json
import re
import statistics
from pathlib import Path


CONDITIONS = ("frozen_label", "frozen_dcs", "finetuned_label", "finetuned_dcs")
RESULT_PATTERN = re.compile(r"Best, last acc:----(\[[^\]]+\])")


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize the prior-alignment 2x2 ablation")
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


def main():
    args = parse_args()
    evaluation_root = Path(args.evaluation_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    seed_dirs = sorted(evaluation_root.glob("seed_*"), key=lambda path: int(path.name.split("_")[-1]))
    if not seed_dirs:
        raise FileNotFoundError(f"No seed_* directories found in {evaluation_root}")
    for seed_dir in seed_dirs:
        generation_seed = int(seed_dir.name.split("_")[-1])
        condition_means = {}
        for condition in CONDITIONS:
            values = parse_log(seed_dir / f"{condition}.log")
            condition_means[condition] = mean(values)
            rows.append(
                {
                    "generation_seed": generation_seed,
                    "condition": condition,
                    "classifier_accuracies": values,
                    "mean_accuracy": condition_means[condition],
                    "std_accuracy": statistics.pstdev(values) if len(values) > 1 else 0.0,
                }
            )

        frozen_prompt_gain = condition_means["frozen_dcs"] - condition_means["frozen_label"]
        finetuned_prompt_gain = condition_means["finetuned_dcs"] - condition_means["finetuned_label"]
        rows.append(
            {
                "generation_seed": generation_seed,
                "condition": "contrasts",
                "frozen_prompt_gain": frozen_prompt_gain,
                "finetuned_prompt_gain": finetuned_prompt_gain,
                "prompt_x_finetuning_interaction": finetuned_prompt_gain - frozen_prompt_gain,
                "finetuning_gain_label": condition_means["finetuned_label"]
                - condition_means["frozen_label"],
                "finetuning_gain_dcs": condition_means["finetuned_dcs"] - condition_means["frozen_dcs"],
            }
        )

    contrasts = [row for row in rows if row["condition"] == "contrasts"]
    aggregate = {}
    for key in (
        "frozen_prompt_gain",
        "finetuned_prompt_gain",
        "prompt_x_finetuning_interaction",
        "finetuning_gain_label",
        "finetuning_gain_dcs",
    ):
        values = [row[key] for row in contrasts]
        aggregate[key] = {
            "mean": mean(values),
            "std_over_generation_seeds": statistics.pstdev(values) if len(values) > 1 else 0.0,
            "values": values,
        }

    payload = {"rows": rows, "aggregate_contrasts": aggregate}
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    fieldnames = sorted({key for row in rows for key in row if key != "classifier_accuracies"})
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: value for key, value in row.items() if key != "classifier_accuracies"})
    print(json.dumps(aggregate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
