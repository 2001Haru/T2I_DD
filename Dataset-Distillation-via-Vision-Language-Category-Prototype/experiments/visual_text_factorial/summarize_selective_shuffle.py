import argparse
import ast
import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

from diagnostic_common import atomic_write_json, parse_shift_runs


RESULT_PATTERN = re.compile(r"Best, last acc:----(\[[^\]]+\])")
CONDITIONS = (
    "correct",
    "all_shuffled",
    "small3_shuffled",
    "random3_shuffled",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize selective small-cluster shuffle controls"
    )
    parser.add_argument("--base-run-root", required=True)
    parser.add_argument(
        "--shuffle-run",
        action="append",
        default=[],
        metavar="SHIFT=RUN_ROOT",
    )
    parser.add_argument("--hybrid-run-root", required=True)
    parser.add_argument("--source-per-class-root")
    parser.add_argument("--generation-seeds", type=int, nargs="+", default=(0, 1))
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def parse_log(path):
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
    return values


def evaluation_log(
    condition,
    shift,
    generation_seed,
    base_run_root,
    shuffle_runs,
    hybrid_run_root,
):
    if condition == "correct":
        return (
            Path(base_run_root)
            / "evaluation"
            / f"seed_{generation_seed}"
            / "prototype_dcs.log"
        )
    if condition == "all_shuffled":
        run_root = Path(base_run_root) if shift == 1 else Path(shuffle_runs[shift])
        return (
            run_root
            / "evaluation"
            / f"seed_{generation_seed}"
            / "prototype_dcs_shuffled.log"
        )
    return (
        Path(hybrid_run_root)
        / f"shift_{shift}"
        / "evaluation"
        / f"seed_{generation_seed}"
        / f"{condition}.log"
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


def plot_contrasts(rows, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as pyplot

    contrast_names = (
        "all_shuffled_minus_correct",
        "small3_shuffled_minus_correct",
        "random3_shuffled_minus_correct",
        "small3_shuffled_minus_random3_shuffled",
        "all_shuffled_minus_small3_shuffled",
    )
    shifts = sorted({int(row["shuffle_shift"]) for row in rows})
    figure, axes = pyplot.subplots(1, len(shifts), figsize=(5 * len(shifts), 5))
    if len(shifts) == 1:
        axes = [axes]
    for axis, shift in zip(axes, shifts):
        selected = [row for row in rows if int(row["shuffle_shift"]) == shift]
        means = [
            statistics.fmean(
                row["mean_difference"]
                for row in selected
                if row["contrast"] == name
            )
            for name in contrast_names
        ]
        axis.bar(range(len(contrast_names)), means)
        axis.axhline(0.0, linestyle="--", color="black")
        axis.set_xticks(
            range(len(contrast_names)),
            [
                "all-correct",
                "small-correct",
                "random-correct",
                "small-random",
                "all-small",
            ],
            rotation=35,
            ha="right",
        )
        axis.set_ylabel("Paired classifier accuracy difference")
        axis.set_title(f"shuffle shift {shift}")
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Selective small-cluster shuffle controls")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    pyplot.close(figure)


def per_class_result_path(
    condition,
    shift,
    generation_seed,
    source_per_class_root,
    hybrid_run_root,
):
    if condition == "correct":
        return (
            Path(source_per_class_root)
            / "results"
            / f"seed_{generation_seed}"
            / "prototype_correct.json"
        )
    if condition == "all_shuffled":
        return (
            Path(source_per_class_root)
            / "results"
            / f"seed_{generation_seed}"
            / f"prototype_shift{shift}.json"
        )
    return (
        Path(hybrid_run_root)
        / f"shift_{shift}"
        / "evaluation"
        / f"seed_{generation_seed}"
        / f"{condition}.per_class.json"
    )


def load_per_class_payload(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing per-class result: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise RuntimeError(f"Unsupported per-class result schema: {path}")
    return payload


def summarize_per_class(args, shifts):
    rows = []
    for shift in shifts:
        for generation_seed in args.generation_seeds:
            payloads = {
                condition: load_per_class_payload(
                    per_class_result_path(
                        condition,
                        shift,
                        generation_seed,
                        args.source_per_class_root,
                        args.hybrid_run_root,
                    )
                )
                for condition in CONDITIONS
            }
            class_names = payloads["correct"]["class_names"]
            for condition, payload in payloads.items():
                if payload["class_names"] != class_names:
                    raise RuntimeError(
                        f"Per-class order mismatch for {condition}, shift={shift}, "
                        f"seed={generation_seed}"
                    )
            for synset in class_names:
                values = {
                    condition: float(payload["mean_per_class_accuracy"][synset])
                    for condition, payload in payloads.items()
                }
                rows.append(
                    {
                        "shuffle_shift": shift,
                        "generation_seed": generation_seed,
                        "synset": synset,
                        **{f"{key}_mean": value for key, value in values.items()},
                        "all_shuffled_minus_correct": (
                            values["all_shuffled"] - values["correct"]
                        ),
                        "small3_shuffled_minus_correct": (
                            values["small3_shuffled"] - values["correct"]
                        ),
                        "random3_shuffled_minus_correct": (
                            values["random3_shuffled"] - values["correct"]
                        ),
                        "small3_shuffled_minus_random3_shuffled": (
                            values["small3_shuffled"] - values["random3_shuffled"]
                        ),
                        "all_shuffled_minus_small3_shuffled": (
                            values["all_shuffled"] - values["small3_shuffled"]
                        ),
                    }
                )

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["synset"]].append(row)
    aggregate = []
    numeric_fields = [
        field
        for field in rows[0]
        if field not in ("shuffle_shift", "generation_seed", "synset")
    ]
    for synset, selected in sorted(grouped.items()):
        result = {
            "synset": synset,
            "shift_generation_observations": len(selected),
        }
        for field in numeric_fields:
            result[field] = statistics.fmean(float(row[field]) for row in selected)
        aggregate.append(result)
    return rows, aggregate


def main():
    args = parse_args()
    shuffle_runs = parse_shift_runs(args.shuffle_run)
    shifts = [1, *sorted(shuffle_runs)]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    condition_rows = []
    contrast_rows = []
    contrast_pairs = (
        ("all_shuffled_minus_correct", "all_shuffled", "correct"),
        ("small3_shuffled_minus_correct", "small3_shuffled", "correct"),
        ("random3_shuffled_minus_correct", "random3_shuffled", "correct"),
        (
            "small3_shuffled_minus_random3_shuffled",
            "small3_shuffled",
            "random3_shuffled",
        ),
        (
            "all_shuffled_minus_small3_shuffled",
            "all_shuffled",
            "small3_shuffled",
        ),
    )
    for shift in shifts:
        for generation_seed in args.generation_seeds:
            values = {}
            for condition in CONDITIONS:
                log_path = evaluation_log(
                    condition,
                    shift,
                    generation_seed,
                    args.base_run_root,
                    shuffle_runs,
                    args.hybrid_run_root,
                )
                accuracies = parse_log(log_path)
                values[condition] = accuracies
                condition_rows.append(
                    {
                        "shuffle_shift": shift,
                        "generation_seed": generation_seed,
                        "condition": condition,
                        "mean_accuracy": mean(accuracies),
                        "std_accuracy": statistics.pstdev(accuracies),
                        "classifier_accuracies": json.dumps(accuracies),
                        "evaluation_log": str(log_path.resolve()),
                    }
                )
            for name, left, right in contrast_pairs:
                differences = paired_subtract(values[left], values[right])
                contrast_rows.append(
                    {
                        "shuffle_shift": shift,
                        "generation_seed": generation_seed,
                        "contrast": name,
                        "mean_difference": mean(differences),
                        "std_difference": statistics.pstdev(differences),
                        "paired_differences": json.dumps(differences),
                    }
                )

    grouped = defaultdict(list)
    for row in contrast_rows:
        grouped[row["contrast"]].append(float(row["mean_difference"]))
    aggregate = [
        {
            "contrast": contrast,
            "shift_generation_observations": len(values),
            "mean_over_shift_generation": mean(values),
            "std_over_shift_generation": statistics.pstdev(values),
            "values": json.dumps(values),
        }
        for contrast, values in sorted(grouped.items())
    ]

    write_csv(output_dir / "selective_conditions.csv", condition_rows)
    write_csv(output_dir / "selective_contrasts.csv", contrast_rows)
    write_csv(output_dir / "selective_aggregate.csv", aggregate)
    per_class_rows = []
    per_class_aggregate = []
    if args.source_per_class_root:
        per_class_rows, per_class_aggregate = summarize_per_class(args, shifts)
        write_csv(output_dir / "selective_per_class.csv", per_class_rows)
        write_csv(
            output_dir / "selective_per_class_aggregate.csv",
            per_class_aggregate,
        )
    atomic_write_json(
        output_dir / "selective_summary.json",
        {
            "conditions": condition_rows,
            "contrasts": contrast_rows,
            "aggregate": aggregate,
            "per_class": per_class_rows,
            "per_class_aggregate": per_class_aggregate,
            "primary_contrast": "small3_shuffled_minus_random3_shuffled",
            "localization_contrast": "small3_shuffled_minus_correct",
            "caveat": (
                "Shift and generation observations share classes and are paired "
                "controls, not independent datasets."
            ),
        },
    )
    plot_contrasts(
        contrast_rows,
        output_dir / "selective_shuffle_contrasts.png",
    )
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
