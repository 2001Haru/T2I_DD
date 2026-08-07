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
MODE_ORDER = ("frozen",) + ROWS


def scores(path):
    matches = RESULT.findall(path.read_text(encoding="utf-8", errors="replace"))
    if not matches:
        raise ValueError(f"No completed classifier result: {path}")
    return [float(value) for value in ast.literal_eval(matches[-1])]


def paired(left, right):
    if len(left) != len(right):
        raise ValueError("Classifier repeat counts differ")
    return [a - b for a, b in zip(left, right)]


def pairwise_average(left, right):
    if len(left) != len(right):
        raise ValueError("Classifier repeat counts differ")
    return [(a + b) / 2.0 for a, b in zip(left, right)]


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


def summarize_endpoint_policy(rows):
    flattened = [value for row in rows for value in row["paired_differences"]]
    lower, upper = hierarchical_bootstrap(rows)
    return {
        "mean_difference": statistics.fmean(flattened),
        "hierarchical_bootstrap_ci_lower": lower,
        "hierarchical_bootstrap_ci_upper": upper,
        "training_generation_cells": len(rows),
        "paired_classifier_observations": len(flattened),
        "bootstrap_samples": 10000,
        "bootstrap_order": "training_seed -> generation_seed -> paired_classifier_repeat",
    }


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
                left_dcs = pairwise_average(
                    values[(training_seed, generation_seed, left, "correct")],
                    values[(training_seed, generation_seed, left, "shuffled")],
                )
                right_seed = None if right == "frozen" else training_seed
                right_dcs = pairwise_average(
                    values[(right_seed, generation_seed, right, "correct")],
                    values[(right_seed, generation_seed, right, "shuffled")],
                )
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

    performance_rows = []
    prompt_effect_rows = []
    for supervision in MODE_ORDER:
        training_seeds = (None,) if supervision == "frozen" else (0, 1)
        for training_seed in training_seeds:
            for generation_seed in generation_seeds:
                label = values[(training_seed, generation_seed, supervision, "label")]
                correct = values[(training_seed, generation_seed, supervision, "correct")]
                shuffled = values[(training_seed, generation_seed, supervision, "shuffled")]
                descriptive = pairwise_average(correct, shuffled)
                for prompt, current in (
                    ("label", label),
                    ("correct", correct),
                    ("shuffled", shuffled),
                    ("descriptive_average", descriptive),
                ):
                    performance_rows.append({
                        "training_seed": training_seed,
                        "generation_seed": generation_seed,
                        "supervision": supervision,
                        "prompt": prompt,
                        "paired_differences": current,
                    })
                for effect, differences in (
                    ("correct_minus_label", paired(correct, label)),
                    ("shuffled_minus_label", paired(shuffled, label)),
                    ("descriptive_minus_label", paired(descriptive, label)),
                    ("correct_minus_shuffled", paired(correct, shuffled)),
                ):
                    prompt_effect_rows.append({
                        "training_seed": training_seed,
                        "generation_seed": generation_seed,
                        "supervision": supervision,
                        "effect": effect,
                        "paired_differences": differences,
                    })

    performance_summary = []
    for supervision in MODE_ORDER:
        for prompt in ("label", "correct", "shuffled", "descriptive_average"):
            selected = [
                row for row in performance_rows
                if row["supervision"] == supervision and row["prompt"] == prompt
            ]
            flattened = [value for row in selected for value in row["paired_differences"]]
            lower, upper = hierarchical_bootstrap(selected)
            performance_summary.append({
                "supervision": supervision,
                "prompt": prompt,
                "mean_accuracy": statistics.fmean(flattened),
                "hierarchical_bootstrap_ci_lower": lower,
                "hierarchical_bootstrap_ci_upper": upper,
                "training_generation_cells": len(selected),
                "classifier_observations": len(flattened),
            })

    prompt_effect_summary = []
    for supervision in MODE_ORDER:
        for effect in ("correct_minus_label", "shuffled_minus_label", "descriptive_minus_label", "correct_minus_shuffled"):
            selected = [
                row for row in prompt_effect_rows
                if row["supervision"] == supervision and row["effect"] == effect
            ]
            flattened = [value for row in selected for value in row["paired_differences"]]
            lower, upper = hierarchical_bootstrap(selected)
            prompt_effect_summary.append({
                "supervision": supervision,
                "effect": effect,
                "mean_difference": statistics.fmean(flattened),
                "hierarchical_bootstrap_ci_lower": lower,
                "hierarchical_bootstrap_ci_upper": upper,
                "training_generation_cells": len(selected),
                "paired_classifier_observations": len(flattened),
            })

    # This is an end-to-end deployment-policy comparison, not a single-component
    # causal rung: both the fine-tuning supervision and inference prompt change.
    endpoint_rows = []
    for training_seed in (0, 1):
        for generation_seed in generation_seeds:
            differences = paired(
                values[(training_seed, generation_seed, "matched_ft", "correct")],
                values[(training_seed, generation_seed, "empty_ft", "label")],
            )
            endpoint_rows.append({
                "training_seed": training_seed,
                "generation_seed": generation_seed,
                "paired_differences": differences,
            })
    endpoint_contrast = summarize_endpoint_policy(endpoint_rows)
    endpoint_contrast.update({
        "contrast": "matched_ft_correct_minus_empty_ft_label",
        "left_policy": "matched_ft + correct_dcs",
        "right_policy": "empty_ft + label",
        "estimand": "accuracy premium of the caption-intensive policy over the caption-free policy",
        "interpretation_boundary": (
            "This joint policy contrast changes training supervision and inference prompt together. "
            "It measures the deployable accuracy-cost tradeoff, not the isolated causal effect of captions."
        ),
    })

    endpoint_performance = []
    for policy, supervision, prompt in (
        ("empty_ft + label", "empty_ft", "label"),
        ("matched_ft + correct_dcs", "matched_ft", "correct"),
    ):
        selected = [
            row for row in performance_rows
            if row["supervision"] == supervision and row["prompt"] == prompt
        ]
        flattened = [value for row in selected for value in row["paired_differences"]]
        lower, upper = hierarchical_bootstrap(selected)
        endpoint_performance.append({
            "policy": policy,
            "mean_accuracy": statistics.fmean(flattened),
            "hierarchical_bootstrap_ci_lower": lower,
            "hierarchical_bootstrap_ci_upper": upper,
            "training_generation_cells": len(selected),
            "classifier_observations": len(flattened),
        })

    mechanism_summary = {
        "bootstrap_order": "training_seed -> generation_seed -> paired_classifier_repeat",
        "performance_by_supervision_and_prompt": performance_summary,
        "prompt_effects_by_supervision": prompt_effect_summary,
        "endpoint_policy_comparison": endpoint_contrast,
        "interpretation_boundary": (
            "Frozen has no fine-tuning-seed level. Other rows use two fine-tuning seeds, "
            "two generation seeds, and paired classifier repeats. Confidence intervals "
            "quantify this experiment and should not be read as cross-dataset intervals."
        ),
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
        "endpoint_policy_comparison": {
            "performance": endpoint_performance,
            "contrast": endpoint_contrast,
        },
        "mechanism_summary": mechanism_summary,
    }
    (output / "causal_ladder_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for filename, records, fields in (
        ("causal_ladder_cells.csv", cells, ("training_seed", "generation_seed", "supervision", "prompt", "mean_accuracy", "std_accuracy")),
        ("causal_ladder_contrasts.csv", contrasts, ("training_seed", "generation_seed", "contrast", "mean_difference", "std_difference")),
        ("performance_by_supervision_and_prompt.csv", performance_summary, ("supervision", "prompt", "mean_accuracy", "hierarchical_bootstrap_ci_lower", "hierarchical_bootstrap_ci_upper", "training_generation_cells", "classifier_observations")),
        ("prompt_effects_by_supervision.csv", prompt_effect_summary, ("supervision", "effect", "mean_difference", "hierarchical_bootstrap_ci_lower", "hierarchical_bootstrap_ci_upper", "training_generation_cells", "paired_classifier_observations")),
        ("endpoint_policy_performance.csv", endpoint_performance, ("policy", "mean_accuracy", "hierarchical_bootstrap_ci_lower", "hierarchical_bootstrap_ci_upper", "training_generation_cells", "classifier_observations")),
        ("endpoint_policy_contrast.csv", [endpoint_contrast], ("contrast", "left_policy", "right_policy", "mean_difference", "hierarchical_bootstrap_ci_lower", "hierarchical_bootstrap_ci_upper", "training_generation_cells", "paired_classifier_observations", "bootstrap_samples", "bootstrap_order", "estimand", "interpretation_boundary")),
    ):
        with (output / filename).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows({field: row[field] for field in fields} for row in records)
    (output / "primary_mechanism_summary.json").write_text(
        json.dumps(mechanism_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    plot_mechanism_summary(performance_summary, prompt_effect_summary, output / "causal_ladder_mechanisms.png")
    plot_endpoint_policy(endpoint_performance, endpoint_contrast, output / "endpoint_policy_comparison.png")
    print(json.dumps(aggregate, indent=2, sort_keys=True))


def plot_mechanism_summary(performance, effects, output):
    import matplotlib.pyplot as plt
    import numpy as np

    labels = [mode.replace("_ft", "").replace("_", " ") for mode in MODE_ORDER]
    x = np.arange(len(MODE_ORDER))

    def performance_metric(mode, prompt):
        return next(row for row in performance if row["supervision"] == mode and row["prompt"] == prompt)

    def effect_metric(mode, effect):
        return next(row for row in effects if row["supervision"] == mode and row["effect"] == effect)

    figure, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    width = 0.36
    for offset, prompt, title in ((-width / 2, "label", "Label"), (width / 2, "descriptive_average", "Descriptive")):
        rows = [performance_metric(mode, prompt) for mode in MODE_ORDER]
        means = np.array([row["mean_accuracy"] for row in rows])
        lower = means - np.array([row["hierarchical_bootstrap_ci_lower"] for row in rows])
        upper = np.array([row["hierarchical_bootstrap_ci_upper"] for row in rows]) - means
        axes[0].bar(x + offset, means, width, label=title, yerr=np.vstack([lower, upper]), capsize=3)
    axes[0].set_title("Causal ladder performance")
    axes[0].set_ylabel("Validation accuracy")
    axes[0].legend()

    for axis, effect, title in (
        (axes[1], "descriptive_minus_label", "Conditioning-style effect"),
        (axes[2], "correct_minus_shuffled", "Cluster-correspondence effect"),
    ):
        rows = [effect_metric(mode, effect) for mode in MODE_ORDER]
        means = np.array([row["mean_difference"] for row in rows])
        lower = means - np.array([row["hierarchical_bootstrap_ci_lower"] for row in rows])
        upper = np.array([row["hierarchical_bootstrap_ci_upper"] for row in rows]) - means
        axis.bar(x, means, yerr=np.vstack([lower, upper]), capsize=3)
        axis.axhline(0, color="black", linestyle="--", linewidth=1)
        axis.set_title(title)
        axis.set_ylabel("Paired accuracy difference")

    for axis in axes:
        axis.set_xticks(x, labels, rotation=25, ha="right")
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Text-supervision causal ladder")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_endpoint_policy(performance, contrast, output):
    import matplotlib.pyplot as plt
    import numpy as np

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    labels = [row["policy"].replace(" + ", "\n+ ") for row in performance]
    means = np.array([row["mean_accuracy"] for row in performance])
    lower = means - np.array([row["hierarchical_bootstrap_ci_lower"] for row in performance])
    upper = np.array([row["hierarchical_bootstrap_ci_upper"] for row in performance]) - means
    axes[0].bar(np.arange(len(labels)), means, yerr=np.vstack([lower, upper]), capsize=4)
    axes[0].set_xticks(np.arange(len(labels)), labels)
    axes[0].set_ylabel("Validation accuracy")
    axes[0].set_title("End-to-end policy performance")

    mean = contrast["mean_difference"]
    error = np.array([[
        mean - contrast["hierarchical_bootstrap_ci_lower"]
    ], [
        contrast["hierarchical_bootstrap_ci_upper"] - mean
    ]])
    axes[1].bar([0], [mean], yerr=error, capsize=4)
    axes[1].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[1].set_xticks([0], ["Matched+Correct\nminus Empty+Label"])
    axes[1].set_ylabel("Paired accuracy difference")
    axes[1].set_title("Caption-intensive policy premium")

    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Caption-free adaptation versus matched-caption conditioning")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
