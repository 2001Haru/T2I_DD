#!/usr/bin/env python3
"""Summarize conditioning-interface matrices with paired causal interactions."""

import argparse
import ast
import csv
import json
import random
import re
import statistics
from pathlib import Path


RESULT = re.compile(r"Best, last acc:----(\[[^\]]+\])")


def read_scores(path):
    matches = RESULT.findall(Path(path).read_text(encoding="utf-8", errors="replace"))
    if not matches:
        raise ValueError(f"No completed classifier result: {path}")
    return [float(value) for value in ast.literal_eval(matches[-1])]


def paired(left, right):
    if len(left) != len(right):
        raise ValueError("Classifier repeat counts differ")
    return [a - b for a, b in zip(left, right)]


def mean_vectors(vectors):
    if not vectors or len({len(values) for values in vectors}) != 1:
        raise ValueError("Cannot average absent or unequal classifier vectors")
    return [statistics.fmean(values) for values in zip(*vectors)]


def bootstrap(rows, samples=10000, seed=20260809):
    grouped = {}
    for row in rows:
        grouped.setdefault(str(row["training_seed"]), {}).setdefault(row["generation_seed"], []).append(row["values"])
    rng = random.Random(seed)
    training_seeds = list(grouped)
    estimates = []
    for _ in range(samples):
        draw = []
        for _ in training_seeds:
            training_seed = rng.choice(training_seeds)
            generations = grouped[training_seed]
            generation_seeds = list(generations)
            for _ in generation_seeds:
                generation_seed = rng.choice(generation_seeds)
                values = rng.choice(generations[generation_seed])
                draw.extend(rng.choice(values) for _ in values)
        estimates.append(statistics.fmean(draw))
    estimates.sort()
    return estimates[int(0.025 * samples)], estimates[min(samples - 1, int(0.975 * samples))]


def summarize(rows):
    values = [value for row in rows for value in row["values"]]
    lower, upper = bootstrap(rows)
    return {
        "mean": statistics.fmean(values),
        "hierarchical_bootstrap_ci_lower": lower,
        "hierarchical_bootstrap_ci_upper": upper,
        "training_generation_cells": len(rows),
        "paired_classifier_observations": len(values),
    }


def effect_vectors(normalized_lookup):
    """Build prompt effects before any seed averaging."""
    effects = {}
    groups = sorted({key[:7] for key in normalized_lookup}, key=str)
    for group in groups:
        label = normalized_lookup.get((*group, "label"))
        correct = normalized_lookup.get((*group, "correct"))
        shuffled = normalized_lookup.get((*group, "shuffled_s1"))
        if label is None or correct is None or shuffled is None:
            continue
        descriptive = paired(mean_vectors((correct, shuffled)), label)
        effects[(*group, "descriptive_marginal")] = descriptive
        effects[(*group, "correspondence")] = paired(correct, shuffled)
    return effects


def checkpoint_boundary_vectors(normalized_lookup):
    """Build absolute descriptive performance and correspondence vectors."""
    effects = {}
    groups = sorted({key[:7] for key in normalized_lookup}, key=str)
    for group in groups:
        correct = normalized_lookup.get((*group, "correct"))
        shuffled = normalized_lookup.get((*group, "shuffled_s1"))
        if correct is None or shuffled is None:
            continue
        effects[(*group, "descriptive_average")] = mean_vectors((correct, shuffled))
        effects[(*group, "correspondence")] = paired(correct, shuffled)
    return effects


def _effect_rows(effect_lookup, identity):
    """Return seed-level vectors matching a six-field effect identity."""
    matrix, spec, ipc, visual, supervision, effect = identity
    rows = {}
    for key, values in effect_lookup.items():
        if (key[0], key[1], key[2], key[3], key[4], key[7]) != identity:
            continue
        rows[(key[5], key[6])] = values
    return rows


def paired_effect_interaction(left, right, broadcast_right_training_seed=False):
    """Pair effect vectors by generation/repeat and, where available, training seed."""
    rows = []
    for (training_seed, generation_seed), left_values in sorted(left.items(), key=str):
        right_key = (training_seed, generation_seed)
        if right_key not in right and broadcast_right_training_seed:
            right_key = (None, generation_seed)
        if right_key not in right:
            continue
        rows.append({
            "training_seed": training_seed,
            "generation_seed": generation_seed,
            "values": paired(left_values, right[right_key]),
        })
    return rows


def visual_label(row):
    return "pure_noise" if row["visual_mode"] == "pure_noise" else f"strength_{float(row['strength']):g}"


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-index", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--matrices", nargs="+")
    parser.add_argument("--specs", nargs="+")
    parser.add_argument("--ipcs", type=int, nargs="+")
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    full_index = json.loads(Path(args.evaluation_index).read_text(encoding="utf-8"))
    index = [
        item for item in full_index
        if (not args.matrices or item["matrix"] in args.matrices)
        and (not args.specs or item["spec"] in args.specs)
        and (not args.ipcs or int(item["ipc"]) in args.ipcs)
    ]
    if not index:
        raise ValueError("No evaluation cells match the requested filters")

    cells, lookup, planned_lookup, incomplete = [], {}, {}, []
    for item in index:
        shift = int(item.get("shuffle_shift") or 1) if item["prompt"] == "shuffled" else None
        visual = visual_label(item)
        key = (
            item["matrix"], item["spec"], int(item["ipc"]), visual, item["supervision"],
            item.get("training_seed"), int(item["generation_seed"]), item["prompt"], shift,
        )
        if key in planned_lookup:
            raise RuntimeError(f"Duplicate planned evaluation cell: {key}")
        planned_lookup[key] = item
        try:
            scores = read_scores(item["evaluation_log"])
        except (FileNotFoundError, ValueError) as error:
            if not args.allow_incomplete:
                raise
            incomplete.append({
                "matrix": item["matrix"], "spec": item["spec"], "ipc": int(item["ipc"]),
                "visual": visual, "supervision": item["supervision"],
                "training_seed": item.get("training_seed"),
                "generation_seed": int(item["generation_seed"]), "prompt": item["prompt"],
                "shuffle_shift": shift, "evaluation_log": item["evaluation_log"],
                "reason": str(error),
            })
            continue
        row = {
            **item, "visual": visual, "mean_accuracy": statistics.fmean(scores),
            "std_accuracy": statistics.pstdev(scores), "classifier_accuracies": scores,
        }
        cells.append(row)
        if key in lookup:
            raise RuntimeError(f"Duplicate evaluation cell: {key}")
        lookup[key] = scores

    groups = sorted({key[:7] for key in planned_lookup}, key=str)
    normalized = []
    for group in groups:
        prefix = group
        label = lookup.get((*prefix, "label", None))
        correct = lookup.get((*prefix, "correct", None))
        planned_shifts = sorted(
            key[-1] for key in planned_lookup if key[:7] == prefix and key[7] == "shuffled"
        )
        completed_shifts = sorted(
            key[-1] for key in lookup if key[:7] == prefix and key[7] == "shuffled"
        )
        all_shifts_complete = bool(planned_shifts) and completed_shifts == planned_shifts
        shuffled = mean_vectors([
            lookup[(*prefix, "shuffled", shift)] for shift in planned_shifts
        ]) if all_shifts_complete else None
        shuffled_primary = lookup.get((*prefix, "shuffled", 1))
        metadata = {
            "matrix": prefix[0], "spec": prefix[1], "ipc": prefix[2], "visual": prefix[3],
            "supervision": prefix[4], "training_seed": prefix[5], "generation_seed": prefix[6],
            "shuffle_shifts": completed_shifts, "planned_shuffle_shifts": planned_shifts,
        }
        if label is not None:
            normalized.append({**metadata, "prompt": "label", "values": label})
        if correct is not None:
            normalized.append({**metadata, "prompt": "correct", "values": correct})
        if shuffled is not None:
            normalized.append({**metadata, "prompt": "shuffled_mean", "values": shuffled})
        if shuffled_primary is not None:
            normalized.append({**metadata, "prompt": "shuffled_s1", "values": shuffled_primary})

    performance = []
    performance_keys = sorted({
        (row["matrix"], row["spec"], row["ipc"], row["visual"], row["supervision"], row["prompt"])
        for row in normalized
    }, key=str)
    for key in performance_keys:
        rows = [
            {"training_seed": row["training_seed"], "generation_seed": row["generation_seed"], "values": row["values"]}
            for row in normalized
            if (row["matrix"], row["spec"], row["ipc"], row["visual"], row["supervision"], row["prompt"]) == key
        ]
        performance.append({
            "matrix": key[0], "spec": key[1], "ipc": key[2], "visual": key[3],
            "supervision": key[4], "prompt": key[5], **summarize(rows),
        })

    normalized_lookup = {
        (
            row["matrix"], row["spec"], row["ipc"], row["visual"], row["supervision"],
            row["training_seed"], row["generation_seed"], row["prompt"],
        ): row["values"]
        for row in normalized
    }
    contrasts = []
    contrast_specs = (
        ("correct_minus_label", "correct", "label"),
        ("shuffled_s1_minus_label", "shuffled_s1", "label"),
        ("correct_minus_shuffled_s1", "correct", "shuffled_s1"),
        ("shuffled_mean_minus_label_robustness", "shuffled_mean", "label"),
        ("correct_minus_shuffled_mean_robustness", "correct", "shuffled_mean"),
    )
    contrast_groups = sorted({key[:5] for key in normalized_lookup}, key=str)
    for group in contrast_groups:
        pairs = sorted({key[5:7] for key in normalized_lookup if key[:5] == group}, key=str)
        for name, left, right in contrast_specs:
            rows = []
            for training_seed, generation_seed in pairs:
                left_key = (*group, training_seed, generation_seed, left)
                right_key = (*group, training_seed, generation_seed, right)
                if left_key in normalized_lookup and right_key in normalized_lookup:
                    rows.append({
                        "training_seed": training_seed, "generation_seed": generation_seed,
                        "values": paired(normalized_lookup[left_key], normalized_lookup[right_key]),
                    })
            if rows:
                contrasts.append({
                    "matrix": group[0], "spec": group[1], "ipc": group[2], "visual": group[3],
                    "supervision": group[4], "contrast": name, **summarize(rows),
                })

    shift_effects = []
    raw_groups = sorted({key[:7] for key in lookup if key[7] == "shuffled"}, key=str)
    for group in raw_groups:
        correct = lookup.get((*group, "correct", None))
        label = lookup.get((*group, "label", None))
        if correct is None or label is None:
            continue
        for shift in sorted(key[-1] for key in lookup if key[:7] == group and key[7] == "shuffled"):
            shuffled = lookup[(*group, "shuffled", shift)]
            for name, values in (
                ("shuffled_minus_label", paired(shuffled, label)),
                ("correct_minus_shuffled", paired(correct, shuffled)),
            ):
                shift_effects.append({
                    "matrix": group[0], "spec": group[1], "ipc": group[2], "visual": group[3],
                    "supervision": group[4], "training_seed": group[5], "generation_seed": group[6],
                    "shuffle_shift": shift, "contrast": name,
                    "mean_paired_difference": statistics.fmean(values), "paired_differences": values,
                })

    effect_lookup = effect_vectors(normalized_lookup)
    checkpoint_boundary_lookup = checkpoint_boundary_vectors(normalized_lookup)
    formal_interactions = []
    formal_interaction_cells = []

    def record_interaction(analysis, contrast, left_identity, right_identity, rows):
        if not rows:
            return
        metadata = {
            "analysis": analysis,
            "contrast": contrast,
            "matrix_left": left_identity[0], "spec_left": left_identity[1],
            "matrix_right": right_identity[0], "spec_right": right_identity[1],
            "ipc": left_identity[2], "visual": left_identity[3],
            "reference_visual": right_identity[3], "supervision_left": left_identity[4],
            "supervision_right": right_identity[4], "effect": left_identity[5],
        }
        formal_interactions.append({**metadata, **summarize(rows)})
        for row in rows:
            formal_interaction_cells.append({
                **metadata, "training_seed": row["training_seed"],
                "generation_seed": row["generation_seed"],
                "mean_paired_interaction": statistics.fmean(row["values"]),
                "paired_interactions": row["values"],
            })

    # Strength changes in the two primitive prompt effects, always relative to 0.7.
    effect_identities = sorted({
        (key[0], key[1], key[2], key[3], key[4], key[7]) for key in effect_lookup
    }, key=str)
    for identity in effect_identities:
        matrix, spec, ipc, visual, supervision, effect = identity
        if not visual.startswith("strength_") or visual == "strength_0.7":
            continue
        reference = (matrix, spec, ipc, "strength_0.7", supervision, effect)
        rows = paired_effect_interaction(
            _effect_rows(effect_lookup, identity), _effect_rows(effect_lookup, reference)
        )
        record_interaction(
            "strength_interaction", f"{visual}_minus_strength_0.7", identity, reference, rows
        )

    # Check whether the prompt effect itself changes with checkpoint supervision.
    checkpoint_pairs = (
        ("matched_ft", "frozen"),
        ("matched_ft", "empty_ft"),
        ("matched_ft", "label_ft"),
        ("unpaired_ft", "label_ft"),
        ("matched_ft", "unpaired_ft"),
    )
    checkpoint_bases = sorted({
        (key[0], key[1], key[2], key[3], key[7]) for key in effect_lookup
    }, key=str)
    for matrix, spec, ipc, visual, effect in checkpoint_bases:
        for left_supervision, right_supervision in checkpoint_pairs:
            left = (matrix, spec, ipc, visual, left_supervision, effect)
            right = (matrix, spec, ipc, visual, right_supervision, effect)
            rows = paired_effect_interaction(
                _effect_rows(effect_lookup, left), _effect_rows(effect_lookup, right),
                broadcast_right_training_seed=right_supervision == "frozen",
            )
            record_interaction(
                "checkpoint_prompt_interaction",
                f"{left_supervision}_minus_{right_supervision}", left, right, rows,
            )

    # Statistical boundary for the causal ladder. Descriptive-average rows compare
    # absolute downstream performance under rich prompts. Correspondence rows are
    # the paired three-way interaction:
    # (Correct-Shuffled)_left - (Correct-Shuffled)_right.
    boundary_bases = sorted({
        (key[0], key[1], key[2], key[3], key[7]) for key in checkpoint_boundary_lookup
    }, key=str)
    for matrix, spec, ipc, visual, effect in boundary_bases:
        for left_supervision, right_supervision in checkpoint_pairs:
            left = (matrix, spec, ipc, visual, left_supervision, effect)
            right = (matrix, spec, ipc, visual, right_supervision, effect)
            rows = paired_effect_interaction(
                _effect_rows(checkpoint_boundary_lookup, left),
                _effect_rows(checkpoint_boundary_lookup, right),
                broadcast_right_training_seed=right_supervision == "frozen",
            )
            analysis = (
                "checkpoint_descriptive_average"
                if effect == "descriptive_average"
                else "checkpoint_correspondence_interaction"
            )
            record_interaction(
                analysis, f"{left_supervision}_minus_{right_supervision}", left, right, rows
            )

    # Cross-dataset generality at matched seeds/repeats. New matrices use the same
    # matrix label for both datasets; retain the historical C-Woof/D-Nette pairing.
    dataset_effects = {}
    identities = {
        (key[0], key[1], key[2], key[3], key[4], key[7]) for key in effect_lookup
    }
    matrices = sorted({identity[0] for identity in identities})
    dataset_matrix_pairs = [
        (matrix, matrix) for matrix in matrices
        if any(identity[0] == matrix and identity[1] == "woof" for identity in identities)
        and any(identity[0] == matrix and identity[1] == "nette" for identity in identities)
    ]
    if any(identity[0] == "C" and identity[1] == "woof" for identity in identities) and any(
        identity[0] == "D" and identity[1] == "nette" for identity in identities
    ):
        dataset_matrix_pairs.append(("C", "D"))
    for woof_matrix, nette_matrix in dataset_matrix_pairs:
        bases = sorted({
            (identity[2], identity[3], identity[4], identity[5])
            for identity in identities
            if (identity[0], identity[1]) in {
                (woof_matrix, "woof"), (nette_matrix, "nette")
            }
        }, key=str)
        for ipc, visual, supervision, effect in bases:
            woof = (woof_matrix, "woof", ipc, visual, supervision, effect)
            nette = (nette_matrix, "nette", ipc, visual, supervision, effect)
            rows = paired_effect_interaction(
                _effect_rows(effect_lookup, woof), _effect_rows(effect_lookup, nette)
            )
            if not rows:
                continue
            dataset_effects[(woof_matrix, nette_matrix, ipc, visual, supervision, effect)] = {
                (row["training_seed"], row["generation_seed"]): row["values"] for row in rows
            }
            record_interaction(
                "dataset_interaction", "woof_minus_nette", woof, nette, rows
            )

    # Difference-in-differences: does the strength response itself differ by dataset?
    for key, current in sorted(dataset_effects.items(), key=str):
        woof_matrix, nette_matrix, ipc, visual, supervision, effect = key
        if visual == "strength_0.7" or not visual.startswith("strength_"):
            continue
        reference = dataset_effects.get((
            woof_matrix, nette_matrix, ipc, "strength_0.7", supervision, effect
        ), {})
        rows = paired_effect_interaction(current, reference)
        matrix_label = f"{woof_matrix}-{nette_matrix}"
        left = (matrix_label, "woof-nette", ipc, visual, supervision, effect)
        right = (matrix_label, "woof-nette", ipc, "strength_0.7", supervision, effect)
        record_interaction(
            "dataset_by_strength_interaction",
            f"(woof-nette)_{visual}_minus_(woof-nette)_strength_0.7",
            left, right, rows,
        )

    payload = {
        "format_version": 2,
        "partial_summary": args.allow_incomplete,
        "filters": {"matrices": args.matrices, "specs": args.specs, "ipcs": args.ipcs},
        "coverage": {
            "planned_evaluation_cells": len(index), "completed_evaluation_cells": len(cells),
            "incomplete_evaluation_cells": len(incomplete),
            "completion_fraction": len(cells) / len(index),
            "complete_primary_triplets": sum(
                all(key in lookup for key in (
                    (*group, "label", None), (*group, "correct", None), (*group, "shuffled", 1)
                ))
                for group in groups
            ),
            "planned_primary_triplets": len(groups),
        },
        "estimands": {
            "prompt_marginal_primary": "shuffle shift 1-label at every visual setting",
            "correspondence_primary": "correct-shuffle shift 1 at every visual setting",
            "prompt_marginal_robustness": "mean(shuffled shifts)-label, with shifts averaged before bootstrap",
            "correspondence_robustness": "correct-mean(shuffled shifts), with shifts averaged before bootstrap",
            "correct_utility": "correct-label",
            "descriptive_marginal": "mean(correct, shuffle shift 1)-label, paired before averaging",
            "descriptive_average": "mean(correct, shuffle shift 1), paired before checkpoint comparison",
            "correspondence": "correct-shuffle shift 1, paired before averaging",
            "checkpoint_correspondence_interaction": (
                "(correct-shuffle shift 1)_left checkpoint minus the same effect at the right checkpoint"
            ),
        },
        "bootstrap_order": "training seed -> generation seed -> paired classifier repeat",
        "formal_interaction_bootstrap_unit": (
            "paired training seed -> paired generation seed -> paired classifier repeat; frozen controls "
            "are broadcast across checkpoint training seeds but retain generation/repeat pairing"
        ),
        "performance": performance, "contrasts": contrasts,
        "formal_interactions": formal_interactions,
        "interpretation_boundary": (
            "Strength changes prototype corruption and effective denoising horizon. Pure-noise is a distinct "
            "text-to-image interface; without a visual source cluster, its Correct-Shuffled contrast measures "
            "caption-allocation/noise-pairing sensitivity rather than cross-modal correspondence. Shuffle shifts "
            "are randomization realizations, not independent experimental units."
        ),
    }
    (output / "conditioning_interface_matrix_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_csv(output / "cells.csv", cells, (
        "matrix", "spec", "ipc", "visual", "supervision", "training_seed", "generation_seed",
        "prompt", "shuffle_shift", "mean_accuracy", "std_accuracy", "classifier_accuracies",
        "source", "evaluation_log",
    ))
    write_csv(output / "performance.csv", performance, (
        "matrix", "spec", "ipc", "visual", "supervision", "prompt", "mean",
        "hierarchical_bootstrap_ci_lower", "hierarchical_bootstrap_ci_upper",
        "training_generation_cells", "paired_classifier_observations",
    ))
    write_csv(output / "contrasts.csv", contrasts, (
        "matrix", "spec", "ipc", "visual", "supervision", "contrast", "mean",
        "hierarchical_bootstrap_ci_lower", "hierarchical_bootstrap_ci_upper",
        "training_generation_cells", "paired_classifier_observations",
    ))
    write_csv(output / "shuffle_shift_effects.csv", shift_effects, (
        "matrix", "spec", "ipc", "visual", "supervision", "training_seed", "generation_seed",
        "shuffle_shift", "contrast", "mean_paired_difference", "paired_differences",
    ))
    formal_fields = (
        "analysis", "contrast", "matrix_left", "spec_left", "matrix_right", "spec_right",
        "ipc", "visual", "reference_visual", "supervision_left", "supervision_right", "effect",
        "mean", "hierarchical_bootstrap_ci_lower", "hierarchical_bootstrap_ci_upper",
        "training_generation_cells", "paired_classifier_observations",
    )
    write_csv(output / "formal_interactions.csv", formal_interactions, formal_fields)
    write_csv(output / "formal_interaction_cells.csv", formal_interaction_cells, (
        *formal_fields[:12], "training_seed", "generation_seed", "mean_paired_interaction",
        "paired_interactions",
    ))
    checkpoint_boundaries = [
        row for row in formal_interactions
        if row["analysis"] in {
            "checkpoint_descriptive_average", "checkpoint_correspondence_interaction"
        }
    ]
    write_csv(
        output / "checkpoint_statistical_boundaries.csv", checkpoint_boundaries, formal_fields
    )
    write_csv(output / "incomplete_cells.csv", incomplete, (
        "matrix", "spec", "ipc", "visual", "supervision", "training_seed", "generation_seed",
        "prompt", "shuffle_shift", "evaluation_log", "reason",
    ))
    plot(performance, contrasts, output / "conditioning_interface_matrix_summary.png")
    plot_formal_interactions(
        formal_interactions, output / "conditioning_interface_formal_interactions.png"
    )
    plot_checkpoint_boundaries(
        checkpoint_boundaries, output / "checkpoint_statistical_boundaries.png"
    )
    print(json.dumps({
        "performance_rows": len(performance), "contrast_rows": len(contrasts),
        "formal_interaction_rows": len(formal_interactions),
    }, indent=2))


def plot(performance, contrasts, destination):
    import matplotlib.pyplot as plt

    matrices = sorted({row["matrix"] for row in contrasts})
    figure, axes = plt.subplots(
        1, len(matrices), figsize=(max(6, 6 * len(matrices)), 5), squeeze=False
    )
    for axis, matrix in zip(axes[0], matrices):
        rows = [
            row for row in contrasts
            if row["matrix"] == matrix and row["contrast"] in {
                "correct_minus_label", "shuffled_s1_minus_label", "correct_minus_shuffled_s1"
            }
        ]
        for key, selected in _group(
            rows, lambda row: (
                row["spec"], row["supervision"], row["ipc"], row["contrast"]
            )
        ).items():
            selected = sorted(selected, key=lambda row: _visual_order(row["visual"]))
            axis.plot(
                [_visual_order(row["visual"]) for row in selected], [row["mean"] for row in selected],
                marker="o", label=f"{key[0]} {key[1]} IPC{key[2]} {key[3]}",
            )
        axis.axhline(0, color="black", linestyle="--", linewidth=1)
        axis.set_title(f"Matrix {matrix}")
        axis.set_xlabel("Visual interface (pure noise=-1; otherwise strength)")
        axis.set_ylabel("Paired accuracy difference")
        axis.grid(alpha=0.25)
        if rows:
            axis.legend(fontsize=7)
    figure.suptitle("Prompt utility across conditioning interfaces")
    figure.tight_layout()
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def plot_formal_interactions(rows, destination):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(18, 5))
    panels = (
        ("strength_interaction", "Strength minus 0.7", None),
        ("checkpoint_prompt_interaction", "Checkpoint x prompt", None),
        ("dataset_interaction", "Woof minus Nette", None),
    )
    for axis, (analysis, title, required_spec) in zip(axes, panels):
        selected = [
            row for row in rows
            if row["analysis"] == analysis
            and (required_spec is None or row["spec_left"] == required_spec)
            and (analysis != "strength_interaction" or row["supervision_left"] == "matched_ft")
        ]
        if analysis == "strength_interaction":
            key_fn = lambda row: (
                row["matrix_left"], row["spec_left"], row["ipc"],
                row["supervision_left"], row["effect"],
            )
        elif analysis == "checkpoint_prompt_interaction":
            key_fn = lambda row: (
                row["matrix_left"], row["spec_left"], row["ipc"],
                row["contrast"], row["effect"],
            )
        else:
            key_fn = lambda row: (
                row["matrix_left"], row["supervision_left"], row["ipc"], row["effect"]
            )
        grouped = _group(selected, key_fn)
        categories = sorted({row["visual"] for row in selected}, key=_visual_order)
        positions = {category: index for index, category in enumerate(categories)}
        for key, group in grouped.items():
            group = sorted(group, key=lambda row: _visual_order(row["visual"]))
            strength_rows = [row for row in group if row["visual"] != "pure_noise"]
            pure_rows = [row for row in group if row["visual"] == "pure_noise"]
            label = " / ".join(str(value) for value in key)
            color = None
            if strength_rows:
                artist = _errorbar_rows(axis, strength_rows, positions, label=label, marker="o")
                color = artist[0].get_color()
            if pure_rows:
                _errorbar_rows(
                    axis, pure_rows, positions, label=label if not strength_rows else None,
                    marker="s", linestyle="none", color=color,
                )
        axis.axhline(0, color="black", linestyle="--", linewidth=1)
        axis.set_title(title)
        axis.set_xlabel("Visual interface")
        axis.set_ylabel("Paired difference-in-differences")
        axis.set_xticks(range(len(categories)), [_visual_tick(category) for category in categories])
        axis.grid(alpha=0.25)
        if selected:
            axis.legend(fontsize=6)
    figure.suptitle("Formal prompt-interface interaction tests")
    figure.tight_layout()
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def plot_checkpoint_boundaries(rows, destination):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(14, 5), squeeze=False)
    panels = (
        ("checkpoint_descriptive_average", "Checkpoint effect on descriptive average"),
        ("checkpoint_correspondence_interaction", "Checkpoint x cluster correspondence"),
    )
    for axis, (analysis, title) in zip(axes[0], panels):
        selected = [row for row in rows if row["analysis"] == analysis]
        grouped = _group(
            selected,
            lambda row: (
                row["matrix_left"], row["spec_left"], row["ipc"], row["contrast"]
            ),
        )
        categories = sorted({row["visual"] for row in selected}, key=_visual_order)
        positions = {category: index for index, category in enumerate(categories)}
        for key, group in grouped.items():
            group = sorted(group, key=lambda row: _visual_order(row["visual"]))
            _errorbar_rows(
                axis, group, positions, marker="o",
                label=f"{key[0]} {key[1]} IPC{key[2]} {key[3]}",
            )
        axis.axhline(0, color="black", linestyle="--", linewidth=1)
        axis.set_title(title)
        axis.set_xlabel("Visual interface")
        axis.set_ylabel("Paired checkpoint difference")
        axis.set_xticks(range(len(categories)), [_visual_tick(value) for value in categories])
        axis.grid(alpha=0.25)
        if selected:
            axis.legend(fontsize=6)
    figure.suptitle("Checkpoint statistical boundaries")
    figure.tight_layout()
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def _errorbar_rows(axis, rows, positions, **kwargs):
    x = [positions[row["visual"]] for row in rows]
    y = [row["mean"] for row in rows]
    lower = [row["hierarchical_bootstrap_ci_lower"] for row in rows]
    upper = [row["hierarchical_bootstrap_ci_upper"] for row in rows]
    return axis.errorbar(
        x, y,
        yerr=(
            [center - bound for center, bound in zip(y, lower)],
            [bound - center for center, bound in zip(y, upper)],
        ),
        capsize=3,
        **kwargs,
    )


def _visual_tick(value):
    return "pure noise" if value == "pure_noise" else value.replace("strength_", "")


def _group(rows, key_fn):
    grouped = {}
    for row in rows:
        grouped.setdefault(key_fn(row), []).append(row)
    return grouped


def _visual_order(value):
    return -1.0 if value == "pure_noise" else float(value.split("_", 1)[1])


if __name__ == "__main__":
    main()
