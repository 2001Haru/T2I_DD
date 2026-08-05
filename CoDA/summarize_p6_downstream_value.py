"""Summarize the P6 visual-injection by prompt downstream factorial."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REGIMES = ("i0g0", "i1g0", "i0g1", "i1g1")
PROMPTS = ("label", "matched_label", "correct", "shuffled")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trained-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--specs", nargs="+", required=True)
    parser.add_argument("--generation-seeds", nargs="+", type=int, required=True)
    return parser.parse_args()


def load_results(root, specs, generation_seeds):
    results = {}
    for spec in specs:
        for seed in generation_seeds:
            for regime in REGIMES:
                for prompt in PROMPTS:
                    condition = f"{regime}_{prompt}"
                    path = (
                        root / spec / f"seed_{seed}" / f"{condition}-resnet_ap"
                        / "per_class_accuracy_all_seeds.json"
                    )
                    if not path.is_file():
                        raise FileNotFoundError(f"Missing completed P6 classifier: {path}")
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    if len(payload.get("overall_top1", [])) != 2:
                        raise ValueError(f"Expected two classifier runs in {path}")
                    results[(spec, seed, regime, prompt)] = payload
    return results


def run_scores(payload):
    return {
        int(run["training_seed"]): float(run["overall_top1"])
        for run in payload["runs"]
    }


def class_scores(payload):
    result = {}
    for run in payload["runs"]:
        training_seed = int(run["training_seed"])
        for row in run["classes"]:
            result[(training_seed, row["class_id"])] = float(row["accuracy"])
    return result


def summarize_values(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "positive_fraction": float(np.mean(values > 0)),
        "observations": int(len(values)),
        "values": values.tolist(),
    }


def paired_contrast(results, spec, generation_seeds, terms, class_id=None):
    values = []
    for generation_seed in generation_seeds:
        score_maps = []
        for coefficient, regime, prompt in terms:
            payload = results[(spec, generation_seed, regime, prompt)]
            scores = run_scores(payload) if class_id is None else class_scores(payload)
            score_maps.append((coefficient, scores))
        training_seeds = sorted(
            set.intersection(*(set(scores) if class_id is None else {key[0] for key in scores}
                               for _, scores in score_maps))
        )
        for training_seed in training_seeds:
            if class_id is None:
                value = sum(coefficient * scores[training_seed] for coefficient, scores in score_maps)
            else:
                value = sum(
                    coefficient * scores[(training_seed, class_id)]
                    for coefficient, scores in score_maps
                )
            values.append(value)
    return values


def contrast_definitions():
    definitions = {}
    for regime in REGIMES:
        definitions[f"{regime}_matched_minus_label"] = (
            (1, regime, "matched_label"), (-1, regime, "label")
        )
        definitions[f"{regime}_correct_minus_label"] = (
            (1, regime, "correct"), (-1, regime, "label")
        )
        definitions[f"{regime}_correct_minus_matched"] = (
            (1, regime, "correct"), (-1, regime, "matched_label")
        )
        definitions[f"{regime}_shuffled_minus_correct"] = (
            (1, regime, "shuffled"), (-1, regime, "correct")
        )
    definitions.update(
        {
            "init_x_correct_g0": (
                (1, "i1g0", "correct"), (-1, "i1g0", "label"),
                (-1, "i0g0", "correct"), (1, "i0g0", "label"),
            ),
            "init_x_correct_g1": (
                (1, "i1g1", "correct"), (-1, "i1g1", "label"),
                (-1, "i0g1", "correct"), (1, "i0g1", "label"),
            ),
            "guidance_x_correct_i0": (
                (1, "i0g1", "correct"), (-1, "i0g1", "label"),
                (-1, "i0g0", "correct"), (1, "i0g0", "label"),
            ),
            "guidance_x_correct_i1": (
                (1, "i1g1", "correct"), (-1, "i1g1", "label"),
                (-1, "i1g0", "correct"), (1, "i1g0", "label"),
            ),
            "init_x_guidance_x_correct": (
                (1, "i1g1", "correct"), (-1, "i1g1", "label"),
                (-1, "i1g0", "correct"), (1, "i1g0", "label"),
                (-1, "i0g1", "correct"), (1, "i0g1", "label"),
                (1, "i0g0", "correct"), (-1, "i0g0", "label"),
            ),
            "init_x_guidance_x_shuffle": (
                (1, "i1g1", "shuffled"), (-1, "i1g1", "correct"),
                (-1, "i1g0", "shuffled"), (1, "i1g0", "correct"),
                (-1, "i0g1", "shuffled"), (1, "i0g1", "correct"),
                (1, "i0g0", "shuffled"), (-1, "i0g0", "correct"),
            ),
            "init_x_dcs_content_g0": (
                (1, "i1g0", "correct"), (-1, "i1g0", "matched_label"),
                (-1, "i0g0", "correct"), (1, "i0g0", "matched_label"),
            ),
            "init_x_dcs_content_g1": (
                (1, "i1g1", "correct"), (-1, "i1g1", "matched_label"),
                (-1, "i0g1", "correct"), (1, "i0g1", "matched_label"),
            ),
            "guidance_x_dcs_content_i0": (
                (1, "i0g1", "correct"), (-1, "i0g1", "matched_label"),
                (-1, "i0g0", "correct"), (1, "i0g0", "matched_label"),
            ),
            "guidance_x_dcs_content_i1": (
                (1, "i1g1", "correct"), (-1, "i1g1", "matched_label"),
                (-1, "i1g0", "correct"), (1, "i1g0", "matched_label"),
            ),
            "init_x_guidance_x_dcs_content": (
                (1, "i1g1", "correct"), (-1, "i1g1", "matched_label"),
                (-1, "i1g0", "correct"), (1, "i1g0", "matched_label"),
                (-1, "i0g1", "correct"), (1, "i0g1", "matched_label"),
                (1, "i0g0", "correct"), (-1, "i0g0", "matched_label"),
            ),
        }
    )
    return definitions


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_prompt_effects(summary, output_path, specs):
    figure, axes = plt.subplots(1, len(specs), figsize=(6 * len(specs), 5), squeeze=False)
    x = np.arange(len(REGIMES))
    width = 0.25
    for axis, spec in zip(axes[0], specs):
        matched = [summary[spec][f"{regime}_matched_minus_label"]["mean"] for regime in REGIMES]
        correct = [summary[spec][f"{regime}_correct_minus_matched"]["mean"] for regime in REGIMES]
        shuffled = [summary[spec][f"{regime}_shuffled_minus_correct"]["mean"] for regime in REGIMES]
        axis.bar(x - width, matched, width, label="Matched - Raw label")
        axis.bar(x, correct, width, label="Correct - Matched")
        axis.bar(x + width, shuffled, width, label="Shuffled - Correct")
        axis.axhline(0, color="black", linestyle="--", linewidth=1)
        axis.set_xticks(x, REGIMES)
        axis.set_title(spec)
        axis.set_ylabel("Paired downstream accuracy difference")
        axis.legend()
    figure.suptitle("P6: prompt value across visual-injection regimes")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main():
    args = parse_args()
    root = Path(args.trained_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = load_results(root, args.specs, args.generation_seeds)
    definitions = contrast_definitions()

    cells = defaultdict(dict)
    for spec in args.specs:
        for regime in REGIMES:
            for prompt in PROMPTS:
                values = []
                for seed in args.generation_seeds:
                    values.extend(run_scores(results[(spec, seed, regime, prompt)]).values())
                cells[spec][f"{regime}_{prompt}"] = summarize_values(values)
    for regime in REGIMES:
        for prompt in PROMPTS:
            values = [
                value
                for spec in args.specs
                for value in cells[spec][f"{regime}_{prompt}"]["values"]
            ]
            cells["combined"][f"{regime}_{prompt}"] = summarize_values(values)

    contrasts = defaultdict(dict)
    contrast_rows = []
    for spec in args.specs:
        for name, terms in definitions.items():
            item = summarize_values(paired_contrast(results, spec, args.generation_seeds, terms))
            contrasts[spec][name] = item
            contrast_rows.append({"spec": spec, "contrast": name, **item, "values": json.dumps(item["values"])})
    for name in definitions:
        values = [
            value
            for spec in args.specs
            for value in contrasts[spec][name]["values"]
        ]
        item = summarize_values(values)
        contrasts["combined"][name] = item
        contrast_rows.append(
            {"spec": "combined", "contrast": name, **item, "values": json.dumps(item["values"])}
        )

    per_class_rows = []
    for spec in args.specs:
        spec_payload = next(payload for (key_spec, _, _, _), payload in results.items() if key_spec == spec)
        spec_classes = {
            row["class_id"]: (row["local_label"], row["class_name"])
            for row in spec_payload["runs"][0]["classes"]
        }
        for class_id, (local_label, class_name) in spec_classes.items():
            row = {"spec": spec, "local_label": local_label, "class_id": class_id, "class_name": class_name}
            for name, terms in definitions.items():
                values = paired_contrast(results, spec, args.generation_seeds, terms, class_id=class_id)
                row[name] = float(np.mean(values))
            per_class_rows.append(row)

    payload = {
        "format_version": 1,
        "generation_seeds": args.generation_seeds,
        "classifier_runs_per_cell": 2 * len(args.generation_seeds),
        "cells": dict(cells),
        "paired_contrasts": dict(contrasts),
        "interpretation_boundary": (
            "P6 measures downstream set value for complete IPC=10 datasets. Clusters excluded "
            "from P4/P5 receive the same label-generated neutral filler across all prompt "
            "conditions within each visual regime and seed."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_csv(output_dir / "paired_contrasts.csv", contrast_rows)
    write_csv(output_dir / "per_class_contrasts.csv", per_class_rows)
    plot_prompt_effects(contrasts, output_dir / "p6_downstream_value.png", args.specs)
    print(f"Saved P6 downstream summary to {output_dir}")


if __name__ == "__main__":
    main()
