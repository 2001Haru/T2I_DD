import argparse
import csv
import json
import os
from statistics import mean, stdev


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trained-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--specs", nargs="+", required=True)
    parser.add_argument("--generation-seeds", nargs="+", type=int, required=True)
    parser.add_argument("--methods", nargs="+", required=True)
    return parser.parse_args()


def load_result(path):
    with open(path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    values = [float(run["overall_top1"]) for run in payload["runs"]]
    if not values:
        raise ValueError(f"No classifier runs found in {path}.")
    return values


def main():
    args = parse_args()
    rows = []
    summary = {}
    for spec in args.specs:
        summary[spec] = {}
        for method in args.methods:
            generation_means = []
            summary[spec][method] = {}
            for seed in args.generation_seeds:
                path = os.path.join(
                    args.trained_root,
                    spec,
                    f"seed_{seed}",
                    f"{method}-resnet_ap",
                    "per_class_accuracy_all_seeds.json",
                )
                values = load_result(path)
                row = {
                    "spec": spec,
                    "method": method,
                    "generation_seed": seed,
                    "classifier_accuracies": values,
                    "mean_accuracy": mean(values),
                    "std_accuracy": stdev(values) if len(values) > 1 else 0.0,
                }
                rows.append(row)
                generation_means.append(row["mean_accuracy"])
                summary[spec][method][str(seed)] = row
            summary[spec][method]["across_generation_seeds"] = {
                "mean_accuracy": mean(generation_means),
                "std_accuracy": stdev(generation_means) if len(generation_means) > 1 else 0.0,
            }

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "experiment_summary.json"), "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
        file.write("\n")
    with open(
        os.path.join(args.output_dir, "condition_results.csv"),
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "spec",
                "method",
                "generation_seed",
                "classifier_accuracies",
                "mean_accuracy",
                "std_accuracy",
            ],
        )
        writer.writeheader()
        for row in rows:
            row = dict(row)
            row["classifier_accuracies"] = json.dumps(row["classifier_accuracies"])
            writer.writerow(row)
    print(f"Saved DCS transfer summary to: {args.output_dir}")


if __name__ == "__main__":
    main()
