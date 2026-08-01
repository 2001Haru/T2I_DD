import argparse
import ast
import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

from diagnostic_common import atomic_write_json


RESULT_PATTERN = re.compile(r"Best, last acc:----(\[[^\]]+\])")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare small-cluster shuffle against multiple random masks"
    )
    parser.add_argument("--small-base-run-root", required=True)
    parser.add_argument("--small-extension-run-root", required=True)
    parser.add_argument("--control-run-root", required=True)
    parser.add_argument("--existing-mask-seed", type=int, default=20260731)
    parser.add_argument(
        "--new-mask-seeds",
        type=int,
        nargs="+",
        default=(20260801, 20260802, 20260803),
    )
    parser.add_argument("--base-generation-seeds", type=int, nargs="+", default=(0, 1))
    parser.add_argument(
        "--extension-generation-seeds", type=int, nargs="+", default=(2, 3)
    )
    parser.add_argument("--shuffle-shifts", type=int, nargs="+", default=(1, 2, 4, 7))
    parser.add_argument("--classifier-repeats", type=int, default=3)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def parse_log(path, classifier_repeats=None):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing evaluation log: {path}")
    matches = RESULT_PATTERN.findall(
        path.read_text(encoding="utf-8", errors="replace")
    )
    if not matches:
        raise ValueError(f"No completed Minimax result found in {path}")
    values = [float(value) for value in ast.literal_eval(matches[-1])]
    if not values:
        raise ValueError(f"Empty classifier result list in {path}")
    if classifier_repeats is not None:
        if classifier_repeats <= 0:
            raise ValueError("classifier-repeats must be positive")
        if len(values) < classifier_repeats:
            raise ValueError(
                f"Expected at least {classifier_repeats} classifier repeats, "
                f"found {len(values)}: {path}"
            )
        values = values[:classifier_repeats]
    return values


def selective_run_root(args, generation_seed):
    if generation_seed in args.base_generation_seeds:
        return Path(args.small_base_run_root)
    if generation_seed in args.extension_generation_seeds:
        return Path(args.small_extension_run_root)
    raise ValueError(f"Generation seed has no selective source run: {generation_seed}")


def selective_log(args, condition, shift, generation_seed):
    return (
        selective_run_root(args, generation_seed)
        / f"shift_{shift}"
        / "evaluation"
        / f"seed_{generation_seed}"
        / f"{condition}.log"
    )


def new_random_log(args, mask_seed, shift, generation_seed):
    return (
        Path(args.control_run_root)
        / f"mask_{mask_seed}"
        / f"shift_{shift}"
        / "evaluation"
        / f"seed_{generation_seed}"
        / "random3_shuffled.log"
    )


def mean(values):
    return statistics.fmean(values)


def paired_subtract(left, right):
    if len(left) != len(right):
        raise ValueError("Paired classifier result lengths differ")
    return [left_value - right_value for left_value, right_value in zip(left, right)]


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def grouped_rows(rows, keys, observation_name):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    output = []
    for key_values, selected in sorted(grouped.items()):
        differences = [float(row["mean_difference"]) for row in selected]
        output.append(
            {
                **dict(zip(keys, key_values)),
                observation_name: len(selected),
                "mean_difference": mean(differences),
                "std_difference": statistics.pstdev(differences),
                "positive_fraction": mean([value > 0.0 for value in differences]),
                "values": json.dumps(differences),
            }
        )
    return output


def aggregate_by_generation(generation_rows):
    grouped = defaultdict(list)
    for row in generation_rows:
        grouped[int(row["mask_seed"])].append(row)
    output = []
    for mask_seed, selected in sorted(grouped.items()):
        differences = [float(row["mean_difference"]) for row in selected]
        output.append(
            {
                "mask_seed": mask_seed,
                "mask_role": selected[0]["mask_role"],
                "generation_seed_observations": len(selected),
                "mean_over_generation_seed": mean(differences),
                "std_over_generation_seed": statistics.pstdev(differences),
                "positive_generation_fraction": mean(
                    [value > 0.0 for value in differences]
                ),
                "generation_values": json.dumps(differences),
            }
        )
    return output


def plot_summary(aggregate, generation_rows, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as pyplot

    labels = [
        "existing" if row["mask_role"] == "existing" else str(row["mask_seed"])
        for row in aggregate
    ]
    means = [float(row["mean_over_generation_seed"]) for row in aggregate]
    errors = [float(row["std_over_generation_seed"]) for row in aggregate]
    figure, axes = pyplot.subplots(1, 2, figsize=(13, 5))
    axes[0].bar(range(len(labels)), means, yerr=errors, capsize=4)
    axes[0].axhline(0.0, color="black", linestyle="--")
    axes[0].set_xticks(range(len(labels)), labels, rotation=25, ha="right")
    axes[0].set_ylabel("Small3 - random3 accuracy")
    axes[0].set_title("Mean over generation seeds")
    axes[0].grid(axis="y", alpha=0.25)

    grouped = defaultdict(list)
    for row in generation_rows:
        grouped[int(row["mask_seed"])].append(row)
    for aggregate_row in aggregate:
        mask_seed = int(aggregate_row["mask_seed"])
        selected = sorted(
            grouped[mask_seed], key=lambda row: int(row["generation_seed"])
        )
        label = "existing" if aggregate_row["mask_role"] == "existing" else str(mask_seed)
        axes[1].plot(
            [int(row["generation_seed"]) for row in selected],
            [float(row["mean_difference"]) for row in selected],
            marker="o",
            label=label,
        )
    axes[1].axhline(0.0, color="black", linestyle="--")
    axes[1].set_xlabel("Generation seed")
    axes[1].set_ylabel("Mean small3 - random3 over shifts")
    axes[1].set_title("Generation-seed consistency")
    axes[1].legend(title="Random mask")
    axes[1].grid(alpha=0.25)
    figure.suptitle("Small-cluster localization against random-mask controls")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    pyplot.close(figure)


def main():
    args = parse_args()
    generation_seeds = [
        *args.base_generation_seeds,
        *args.extension_generation_seeds,
    ]
    if len(set(generation_seeds)) != len(generation_seeds):
        raise ValueError("Base and extension generation seeds must be disjoint")
    mask_seeds = [args.existing_mask_seed, *args.new_mask_seeds]
    if len(set(mask_seeds)) != len(mask_seeds):
        raise ValueError("Random mask seeds must be unique")

    rows = []
    for mask_seed in mask_seeds:
        mask_role = "existing" if mask_seed == args.existing_mask_seed else "new"
        for shift in args.shuffle_shifts:
            for generation_seed in generation_seeds:
                small_path = selective_log(
                    args, "small3_shuffled", shift, generation_seed
                )
                if mask_role == "existing":
                    random_path = selective_log(
                        args, "random3_shuffled", shift, generation_seed
                    )
                else:
                    random_path = new_random_log(
                        args, mask_seed, shift, generation_seed
                    )
                small_values = parse_log(small_path, args.classifier_repeats)
                random_values = parse_log(random_path, args.classifier_repeats)
                differences = paired_subtract(small_values, random_values)
                rows.append(
                    {
                        "mask_seed": mask_seed,
                        "mask_role": mask_role,
                        "shuffle_shift": shift,
                        "generation_seed": generation_seed,
                        "small3_mean": mean(small_values),
                        "random3_mean": mean(random_values),
                        "mean_difference": mean(differences),
                        "std_paired_difference": statistics.pstdev(differences),
                        "paired_differences": json.dumps(differences),
                        "small3_log": str(small_path.resolve()),
                        "random3_log": str(random_path.resolve()),
                    }
                )

    generation_rows = grouped_rows(
        rows,
        ("mask_seed", "mask_role", "generation_seed"),
        "shift_observations",
    )
    shift_rows = grouped_rows(
        rows,
        ("mask_seed", "mask_role", "shuffle_shift"),
        "generation_seed_observations",
    )
    aggregate = aggregate_by_generation(generation_rows)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "random_mask_cells.csv", rows)
    write_csv(output_dir / "random_mask_by_generation.csv", generation_rows)
    write_csv(output_dir / "random_mask_by_shift.csv", shift_rows)
    write_csv(output_dir / "random_mask_aggregate.csv", aggregate)
    atomic_write_json(
        output_dir / "random_mask_summary.json",
        {
            "cells": rows,
            "by_generation": generation_rows,
            "by_shift": shift_rows,
            "aggregate": aggregate,
            "primary_statistic": (
                "For each random mask, average small3-random3 over shifts within "
                "each generation seed, then summarize the four generation means."
            ),
            "classifier_repeats": args.classifier_repeats,
            "classifier_repeat_policy": (
                "Use the first N classifier repeats from every log so existing "
                "repeat-3 and new repeat-2 conditions remain paired."
            ),
            "caveat": (
                "The four generation seeds are the primary repeats. Shifts and "
                "classifier repeats are paired measurements, not independent datasets."
            ),
        },
    )
    plot_summary(
        aggregate,
        generation_rows,
        output_dir / "random_mask_controls.png",
    )
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
