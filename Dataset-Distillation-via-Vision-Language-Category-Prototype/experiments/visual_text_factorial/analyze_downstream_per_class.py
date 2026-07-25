import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

from diagnostic_common import load_json


DIAGNOSTIC_FIELDS = (
    "coverage_distance_improvement",
    "fidelity_distance_improvement",
    "coverage_fraction_change",
    "precision_fraction_change",
    "diversity_change",
    "centroid_distance_improvement",
    "conditioning_relative_l2",
    "conditioning_mean_hidden_cosine",
    "conditioning_token_jaccard",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Relate per-class downstream gains to DINO and conditioning changes"
    )
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--diagnostic-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--visual-modes", nargs="+", default=("prototype",))
    parser.add_argument("--generation-seeds", type=int, nargs="+", default=(0, 1))
    parser.add_argument("--shuffle-shifts", type=int, nargs="+", default=(1, 2, 4, 7))
    return parser.parse_args()


def read_csv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"Cannot write an empty CSV: {path}")
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def result_path(results_root, generation_seed, visual_mode, condition):
    return (
        Path(results_root)
        / f"seed_{generation_seed}"
        / f"{visual_mode}_{condition}.json"
    )


def validate_result(payload, expected_classes, path):
    if payload.get("schema_version") != 1:
        raise RuntimeError(f"Unsupported per-class result schema in {path}")
    if payload["class_names"] != expected_classes:
        raise RuntimeError(
            f"Class order mismatch in {path}: "
            f"{payload['class_names']} != {expected_classes}"
        )
    if not payload["repeats"]:
        raise RuntimeError(f"No classifier repeats in {path}")
    for item in payload["repeats"]:
        if len(item["per_class_accuracy"]) != len(expected_classes):
            raise RuntimeError(f"Per-class accuracy length mismatch in {path}")


def rankdata(values):
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average_rank = 0.5 * (start + 1 + end)
        for position in range(start, end):
            ranks[order[position]] = average_rank
        start = end
    return ranks


def pearson(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def spearman(left, right):
    return pearson(rankdata(left), rankdata(right))


def aggregate_generation_seeds(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[
            (
                row["visual_mode"],
                int(row["shuffle_shift"]),
                row["synset"],
            )
        ].append(row)
    aggregated = []
    numeric_fields = (
        "correct_mean_accuracy",
        "shuffled_mean_accuracy",
        "downstream_gain",
        "downstream_gain_std_over_classifier_repeats",
        *DIAGNOSTIC_FIELDS,
    )
    for (visual_mode, shift, synset), selected in grouped.items():
        row = {
            "visual_mode": visual_mode,
            "shuffle_shift": shift,
            "synset": synset,
            "generation_seeds": len(selected),
        }
        for field in numeric_fields:
            row[field] = statistics.fmean(float(item[field]) for item in selected)
        aggregated.append(row)
    return sorted(
        aggregated,
        key=lambda row: (row["visual_mode"], row["shuffle_shift"], row["synset"]),
    )


def correlation_rows(rows, unit):
    output = []
    for visual_mode in sorted({row["visual_mode"] for row in rows}):
        selected = [row for row in rows if row["visual_mode"] == visual_mode]
        downstream = [float(row["downstream_gain"]) for row in selected]
        for field in DIAGNOSTIC_FIELDS:
            metric = [float(row[field]) for row in selected]
            output.append(
                {
                    "analysis_unit": unit,
                    "visual_mode": visual_mode,
                    "metric": field,
                    "n": len(selected),
                    "pearson": pearson(metric, downstream),
                    "spearman": spearman(metric, downstream),
                }
            )
    return output


def class_mean_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["visual_mode"], row["synset"])].append(row)
    output = []
    for (visual_mode, synset), selected in grouped.items():
        gains = [float(row["downstream_gain"]) for row in selected]
        output.append(
            {
                "visual_mode": visual_mode,
                "synset": synset,
                "mean_downstream_gain": statistics.fmean(gains),
                "std_over_generation_seed_and_shift": (
                    statistics.pstdev(gains) if len(gains) > 1 else 0.0
                ),
                "positive_fraction": sum(value > 0 for value in gains) / len(gains),
                "observations": len(gains),
            }
        )
    return sorted(output, key=lambda row: (row["visual_mode"], row["synset"]))


def shift_summary_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["visual_mode"], int(row["shuffle_shift"]))].append(row)
    output = []
    for (visual_mode, shift), selected in grouped.items():
        gains = [float(row["downstream_gain"]) for row in selected]
        output.append(
            {
                "visual_mode": visual_mode,
                "shuffle_shift": shift,
                "mean_class_gain": statistics.fmean(gains),
                "std_over_classes_and_generation_seeds": (
                    statistics.pstdev(gains) if len(gains) > 1 else 0.0
                ),
                "positive_class_fraction": sum(value > 0 for value in gains) / len(gains),
                "class_seed_observations": len(gains),
            }
        )
    return sorted(output, key=lambda row: (row["visual_mode"], row["shuffle_shift"]))


def plot_correlations(rows, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as pyplot

    panels = (
        ("coverage_distance_improvement", "Coverage improvement"),
        ("fidelity_distance_improvement", "Fidelity improvement"),
        ("diversity_change", "Diversity change"),
        ("centroid_distance_improvement", "Centroid improvement"),
        ("conditioning_relative_l2", "Conditioning relative L2"),
    )
    figure, axes = pyplot.subplots(2, 3, figsize=(16, 9))
    colors = {"prototype": "tab:blue", "no_visual": "tab:orange"}
    for axis, (field, title) in zip(axes.flat, panels):
        for visual_mode in sorted({row["visual_mode"] for row in rows}):
            selected = [row for row in rows if row["visual_mode"] == visual_mode]
            axis.scatter(
                [float(row[field]) for row in selected],
                [float(row["downstream_gain"]) for row in selected],
                label=visual_mode,
                color=colors.get(visual_mode),
                alpha=0.7,
            )
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_xlabel(title)
        axis.set_ylabel("Shuffled - correct per-class accuracy")
        axis.grid(alpha=0.25)
    axes.flat[0].legend()
    shift_axis = axes.flat[-1]
    for visual_mode in sorted({row["visual_mode"] for row in rows}):
        selected = [row for row in rows if row["visual_mode"] == visual_mode]
        shift_axis.scatter(
            [int(row["shuffle_shift"]) for row in selected],
            [float(row["downstream_gain"]) for row in selected],
            label=visual_mode,
            color=colors.get(visual_mode),
            alpha=0.65,
        )
    shift_axis.axhline(0, color="black", linewidth=0.8)
    shift_axis.set_xlabel("Shuffle shift")
    shift_axis.set_ylabel("Shuffled - correct per-class accuracy")
    shift_axis.set_title("Gain distribution by shuffle")
    shift_axis.grid(alpha=0.25)
    figure.suptitle("Downstream class gain versus semantic coverage diagnostics")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    pyplot.close(figure)


def main():
    args = parse_args()
    results_root = Path(args.results_root).resolve()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostic_rows = read_csv(args.diagnostic_csv)
    diagnostic_index = {
        (
            int(float(row["generation_seed"])),
            row["visual_mode"],
            int(float(row["shuffle_shift"])),
            row["synset"],
        ): row
        for row in diagnostic_rows
    }
    expected_classes = sorted({row["synset"] for row in diagnostic_rows})

    joined_rows = []
    for visual_mode in args.visual_modes:
        for generation_seed in args.generation_seeds:
            correct_path = result_path(
                results_root, generation_seed, visual_mode, "correct"
            )
            correct = load_json(correct_path)
            validate_result(correct, expected_classes, correct_path)
            for shift in args.shuffle_shifts:
                shuffled_path = result_path(
                    results_root,
                    generation_seed,
                    visual_mode,
                    f"shift{shift}",
                )
                shuffled = load_json(shuffled_path)
                validate_result(shuffled, expected_classes, shuffled_path)
                if len(correct["repeats"]) != len(shuffled["repeats"]):
                    raise RuntimeError(
                        f"Classifier repeat mismatch: {correct_path} vs {shuffled_path}"
                    )
                for class_index, synset in enumerate(expected_classes):
                    repeat_gains = [
                        float(shuffled_repeat["per_class_accuracy"][class_index])
                        - float(correct_repeat["per_class_accuracy"][class_index])
                        for correct_repeat, shuffled_repeat in zip(
                            correct["repeats"], shuffled["repeats"]
                        )
                    ]
                    diagnostic = diagnostic_index[
                        (generation_seed, visual_mode, shift, synset)
                    ]
                    row = {
                        "generation_seed": generation_seed,
                        "visual_mode": visual_mode,
                        "shuffle_shift": shift,
                        "synset": synset,
                        "correct_mean_accuracy": correct["mean_per_class_accuracy"][
                            synset
                        ],
                        "shuffled_mean_accuracy": shuffled[
                            "mean_per_class_accuracy"
                        ][synset],
                        "downstream_gain": statistics.fmean(repeat_gains),
                        "downstream_gain_std_over_classifier_repeats": (
                            statistics.pstdev(repeat_gains)
                            if len(repeat_gains) > 1
                            else 0.0
                        ),
                        "classifier_repeat_gains": json.dumps(repeat_gains),
                    }
                    for field in DIAGNOSTIC_FIELDS:
                        row[field] = float(diagnostic[field])
                    joined_rows.append(row)

    aggregated_rows = aggregate_generation_seeds(joined_rows)
    correlations = [
        *correlation_rows(joined_rows, "class_x_shift_x_generation_seed"),
        *correlation_rows(
            aggregated_rows,
            "class_x_shift_averaged_over_generation_seeds",
        ),
    ]
    class_rows = class_mean_rows(joined_rows)
    shift_rows = shift_summary_rows(joined_rows)
    write_csv(output_dir / "downstream_dino_per_class.csv", joined_rows)
    write_csv(
        output_dir / "downstream_dino_averaged_generation_seeds.csv",
        aggregated_rows,
    )
    write_csv(output_dir / "downstream_dino_correlations.csv", correlations)
    write_csv(output_dir / "downstream_class_mean_gains.csv", class_rows)
    write_csv(output_dir / "downstream_shift_summary.csv", shift_rows)
    payload = {
        "results_root": str(results_root),
        "diagnostic_csv": str(Path(args.diagnostic_csv).resolve()),
        "visual_modes": args.visual_modes,
        "generation_seeds": args.generation_seeds,
        "shuffle_shifts": args.shuffle_shifts,
        "correlations": correlations,
        "class_mean_gains": class_rows,
        "shift_summary": shift_rows,
        "notes": {
            "downstream_gain": "shuffled per-class accuracy minus correct-DCS accuracy",
            "repeat_pairing": (
                "Classifier repeats are paired by repeat index; each condition starts "
                "from the same classifier seed."
            ),
            "correlations": (
                "Exploratory Pearson and Spearman coefficients; generation-seed-"
                "averaged rows are the less pseudoreplicated primary view."
            ),
        },
    }
    (output_dir / "downstream_dino_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    plot_correlations(
        aggregated_rows,
        output_dir / "downstream_dino_correlations.png",
    )
    print(json.dumps({"correlations": correlations, "shifts": shift_rows}, indent=2))


if __name__ == "__main__":
    main()
