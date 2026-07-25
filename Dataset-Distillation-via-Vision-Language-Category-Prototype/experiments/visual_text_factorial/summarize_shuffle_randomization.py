import argparse
import csv
import json
import statistics
from pathlib import Path

from summarize_results import parse_log


VISUAL_MODES = ("no_visual", "prototype")


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize fixed DCS shuffle controls")
    parser.add_argument("--base-run-root", required=True)
    parser.add_argument(
        "--shuffle-run",
        action="append",
        default=[],
        metavar="SHIFT=RUN_ROOT",
        help="Additional shuffled-only run; may be supplied multiple times",
    )
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def parse_shift_runs(items):
    runs = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected SHIFT=RUN_ROOT, got {item!r}")
        shift_text, root_text = item.split("=", 1)
        shift = int(shift_text)
        if shift in runs:
            raise ValueError(f"Duplicate shuffle shift: {shift}")
        if not 1 <= shift <= 9:
            raise ValueError(f"Shuffle shift must be in [1, 9], got {shift}")
        runs[shift] = Path(root_text).resolve()
    return runs


def mean_log(run_root, generation_seed, condition):
    values = parse_log(
        Path(run_root) / "evaluation" / f"seed_{generation_seed}" / f"{condition}.log"
    )
    return statistics.fmean(values), values


def discover_generation_seeds(base_run_root):
    evaluation_root = Path(base_run_root) / "evaluation"
    seeds = sorted(
        int(path.name.split("_")[-1]) for path in evaluation_root.glob("seed_*")
    )
    if not seeds:
        raise FileNotFoundError(f"No evaluation seeds under {evaluation_root}")
    return seeds


def rank_descending(value, candidates):
    return 1 + sum(candidate > value for candidate in candidates)


def main():
    args = parse_args()
    base_run_root = Path(args.base_run_root).resolve()
    additional_runs = parse_shift_runs(args.shuffle_run)
    if 1 in additional_runs:
        raise ValueError("Shift 1 is already read from the base factorial run")
    shuffle_runs = {1: base_run_root, **additional_runs}
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generation_seeds = discover_generation_seeds(base_run_root)

    condition_rows = []
    comparison_rows = []
    seed_summaries = []
    for generation_seed in generation_seeds:
        per_visual = {}
        for visual_mode in VISUAL_MODES:
            label_mean, label_values = mean_log(
                base_run_root, generation_seed, f"{visual_mode}_label"
            )
            correct_mean, correct_values = mean_log(
                base_run_root, generation_seed, f"{visual_mode}_dcs"
            )
            shuffled_means = {}
            for shift, run_root in sorted(shuffle_runs.items()):
                shuffled_mean, shuffled_values = mean_log(
                    run_root,
                    generation_seed,
                    f"{visual_mode}_dcs_shuffled",
                )
                shuffled_means[shift] = shuffled_mean
                condition_rows.append(
                    {
                        "generation_seed": generation_seed,
                        "visual_mode": visual_mode,
                        "prompt_condition": "shuffled_dcs",
                        "shuffle_shift": shift,
                        "mean_accuracy": shuffled_mean,
                        "classifier_accuracies": shuffled_values,
                    }
                )
                comparison_rows.append(
                    {
                        "generation_seed": generation_seed,
                        "visual_mode": visual_mode,
                        "shuffle_shift": shift,
                        "correct_minus_shuffled": correct_mean - shuffled_mean,
                        "shuffled_minus_label": shuffled_mean - label_mean,
                    }
                )

            condition_rows.extend(
                [
                    {
                        "generation_seed": generation_seed,
                        "visual_mode": visual_mode,
                        "prompt_condition": "label",
                        "shuffle_shift": None,
                        "mean_accuracy": label_mean,
                        "classifier_accuracies": label_values,
                    },
                    {
                        "generation_seed": generation_seed,
                        "visual_mode": visual_mode,
                        "prompt_condition": "correct_dcs",
                        "shuffle_shift": 0,
                        "mean_accuracy": correct_mean,
                        "classifier_accuracies": correct_values,
                    },
                ]
            )
            shuffle_values = list(shuffled_means.values())
            per_visual[visual_mode] = {
                "label": label_mean,
                "correct": correct_mean,
                "shuffled": shuffled_means,
                "correct_minus_shuffle_mean": correct_mean
                - statistics.fmean(shuffle_values),
                "correct_rank_among_correct_and_shuffles": rank_descending(
                    correct_mean, [correct_mean, *shuffle_values]
                ),
            }

        interactions = {}
        for shift in sorted(shuffle_runs):
            prototype_difference = (
                per_visual["prototype"]["correct"]
                - per_visual["prototype"]["shuffled"][shift]
            )
            no_visual_difference = (
                per_visual["no_visual"]["correct"]
                - per_visual["no_visual"]["shuffled"][shift]
            )
            interactions[shift] = prototype_difference - no_visual_difference
        seed_summaries.append(
            {
                "generation_seed": generation_seed,
                "visual_modes": per_visual,
                "cluster_correspondence_interactions": interactions,
                "mean_cluster_correspondence_interaction": statistics.fmean(
                    interactions.values()
                ),
            }
        )

    aggregate = {}
    for visual_mode in VISUAL_MODES:
        values = [
            item["visual_modes"][visual_mode]["correct_minus_shuffle_mean"]
            for item in seed_summaries
        ]
        ranks = [
            item["visual_modes"][visual_mode]["correct_rank_among_correct_and_shuffles"]
            for item in seed_summaries
        ]
        aggregate[visual_mode] = {
            "correct_minus_mean_shuffled": statistics.fmean(values),
            "std_over_generation_seeds": statistics.pstdev(values)
            if len(values) > 1
            else 0.0,
            "generation_seed_values": values,
            "correct_ranks": ranks,
        }
    interaction_values = [
        item["mean_cluster_correspondence_interaction"] for item in seed_summaries
    ]
    aggregate["visual_x_cluster_correspondence"] = {
        "mean": statistics.fmean(interaction_values),
        "std_over_generation_seeds": statistics.pstdev(interaction_values)
        if len(interaction_values) > 1
        else 0.0,
        "generation_seed_values": interaction_values,
    }

    payload = {
        "base_run_root": str(base_run_root),
        "shuffle_runs": {
            str(shift): str(run_root) for shift, run_root in sorted(shuffle_runs.items())
        },
        "generation_seeds": generation_seeds,
        "condition_rows": condition_rows,
        "comparison_rows": comparison_rows,
        "seed_summaries": seed_summaries,
        "aggregate": aggregate,
        "estimand": "correct DCS accuracy minus the mean accuracy over prespecified shuffled pairings",
        "reporting_rule": "Average every prespecified shuffle; do not select the best shuffle.",
    }
    (output_dir / "shuffle_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    comparison_fields = (
        "generation_seed",
        "visual_mode",
        "shuffle_shift",
        "correct_minus_shuffled",
        "shuffled_minus_label",
    )
    with (output_dir / "shuffle_comparisons.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=comparison_fields)
        writer.writeheader()
        writer.writerows(comparison_rows)
    print(json.dumps(aggregate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
