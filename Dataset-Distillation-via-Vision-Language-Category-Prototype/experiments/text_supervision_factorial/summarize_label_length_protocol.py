#!/usr/bin/env python3
"""Summarize padded-77 versus token-max variable-length raw-Label conditioning."""

import argparse
import ast
import csv
import json
import random
import re
import statistics
from pathlib import Path


RESULT = re.compile(r"Best, last acc:----(\[[^\]]+\])")
PROMPTS = ("label", "raw_label_tokenmax_var")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-index", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    return parser.parse_args()


def scores(path):
    matches = RESULT.findall(Path(path).read_text(encoding="utf-8", errors="replace"))
    if not matches:
        raise ValueError(f"No completed classifier result: {path}")
    return [float(value) for value in ast.literal_eval(matches[-1])]


def percentile(values, probability):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(probability * len(ordered)))]


def bootstrap_paired(paired, samples, seed=20260814):
    generation_seeds = sorted(paired)
    observed = [value for generation in generation_seeds for value in paired[generation]]
    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        values = []
        for _ in generation_seeds:
            generation = rng.choice(generation_seeds)
            differences = paired[generation]
            values.extend(rng.choice(differences) for _ in differences)
        estimates.append(statistics.fmean(values))
    return {
        "mean_difference": statistics.fmean(observed),
        "bootstrap_ci95_lower": percentile(estimates, 0.025),
        "bootstrap_ci95_upper": percentile(estimates, 0.975),
        "generation_cells": len(generation_seeds),
        "paired_classifier_observations": len(observed),
        "bootstrap_order": "shared generation seed -> paired classifier repeat",
    }


def per_class_repeats(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    classes = payload["class_names"]
    return [{
        class_name: float(value)
        for class_name, value in zip(classes, repeat["per_class_accuracy"])
    } for repeat in payload["repeats"]]


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    index_path = Path(args.evaluation_index).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    by_prompt = {prompt: {} for prompt in PROMPTS}
    per_class = {prompt: {} for prompt in PROMPTS}
    performance = []
    for prompt in PROMPTS:
        selected = [row for row in index if row["prompt"] == prompt]
        for row in selected:
            by_prompt[prompt][int(row["generation_seed"])] = scores(row["evaluation_log"])
            per_class[prompt][int(row["generation_seed"])] = per_class_repeats(
                row["per_class_output"]
            )
        flat = [value for values in by_prompt[prompt].values() for value in values]
        performance.append({
            "prompt": prompt,
            "conditioning_protocol": (
                "raw_label_pad77" if prompt == "label" else "raw_label_tokenmax_var"
            ),
            "mean_accuracy": statistics.fmean(flat),
            "generation_cells": len(by_prompt[prompt]),
            "classifier_observations": len(flat),
        })

    variable_prompt = "raw_label_tokenmax_var"
    generations = sorted(set(by_prompt["label"]) & set(by_prompt[variable_prompt]))
    paired = {}
    for seed in generations:
        padded = by_prompt["label"][seed]
        variable = by_prompt[variable_prompt][seed]
        if len(padded) != len(variable):
            raise ValueError(f"Classifier repeat mismatch at generation seed {seed}")
        paired[seed] = [right - left for right, left in zip(variable, padded)]
    primary = bootstrap_paired(paired, args.bootstrap_samples)
    primary["contrast"] = "raw_label_tokenmax_var_minus_raw_label_pad77"
    primary["population"] = "all_classes"

    sequence_rows = []
    run_root = index_path.parent
    protocol_rows = {}
    for prompt in PROMPTS:
        lengths = []
        for seed in generations:
            records_path = (
                run_root / "synthetic" / f"seed_{seed}" / f"matched_ft_{prompt}"
                / "prompt_records.json"
            )
            records = json.loads(records_path.read_text(encoding="utf-8"))
            lengths.extend(int(row["conditioning_sequence_length"]) for row in records)
            if prompt == variable_prompt:
                for row in records:
                    protocol_rows.setdefault(row["synset"], {
                        "synset": row["synset"],
                        "raw_label": row["prompt"],
                        "positive_whitespace_words": row["positive_whitespace_words"],
                        "negative_whitespace_words": row["negative_whitespace_words"],
                        "positive_clip_tokens": row["positive_clip_tokens"],
                        "negative_clip_tokens": row["negative_clip_tokens"],
                        "tokenmax_shared_length": row["tokenmax_shared_length"],
                        "official_whitespace_selected_branch": row[
                            "official_whitespace_selected_branch"
                        ],
                        "tokenmax_selected_branch": row["tokenmax_selected_branch"],
                        "official_branch_disagrees_with_tokenmax": row[
                            "official_branch_disagrees_with_tokenmax"
                        ],
                        "official_whitespace_would_shape_mismatch": row[
                            "official_whitespace_would_shape_mismatch"
                        ],
                    })
        sequence_rows.append({
            "prompt": prompt,
            "sequence_length_mean": statistics.fmean(lengths),
            "sequence_length_min": min(lengths),
            "sequence_length_max": max(lengths),
            "records": len(lengths),
        })

    mismatch_classes = {
        synset for synset, row in protocol_rows.items()
        if row["official_whitespace_would_shape_mismatch"]
    }
    all_classes = set(protocol_rows)
    populations = {
        "all_classes": all_classes,
        "heuristic_compatible_classes": all_classes - mismatch_classes,
        "heuristic_mismatch_classes": mismatch_classes,
    }
    primary["classes"] = len(all_classes)
    population_contrasts = [primary]
    for population, classes in populations.items():
        if population == "all_classes" or not classes:
            continue
        population_paired = {}
        for generation in generations:
            padded_repeats = per_class["label"][generation]
            variable_repeats = per_class[variable_prompt][generation]
            if len(padded_repeats) != len(variable_repeats):
                raise ValueError(f"Per-class repeat mismatch at generation seed {generation}")
            population_paired[generation] = [
                statistics.fmean(variable[class_name] - padded[class_name] for class_name in classes)
                for padded, variable in zip(padded_repeats, variable_repeats)
            ]
        result = bootstrap_paired(
            population_paired, args.bootstrap_samples,
            seed=20260814 + len(population_contrasts),
        )
        result.update({
            "contrast": "raw_label_tokenmax_var_minus_raw_label_pad77",
            "population": population,
            "classes": len(classes),
        })
        population_contrasts.append(result)

    write_csv(output / "performance.csv", performance)
    write_csv(output / "paired_contrast.csv", population_contrasts)
    write_csv(output / "conditioning_lengths.csv", sequence_rows)
    write_csv(output / "label_length_by_class.csv", list(protocol_rows.values()))
    report_lines = [
        "# Raw-label conditioning-length report",
        "",
        "- Positive condition: exact ImageNet class string.",
        "- Negative condition: `cartoon, anime, painting`.",
        "- Variable arm: both branches padded to the larger actual CLIP-token length.",
        "- Official whitespace heuristic is audited but never used for generation.",
        "",
        "## Paired downstream contrasts",
        "",
        "| Population | Classes | Mean | 95% CI |",
        "|---|---:|---:|---:|",
    ]
    for row in population_contrasts:
        report_lines.append(
            f"| {row['population']} | {row.get('classes', len(all_classes))} | "
            f"{row['mean_difference']:.3f} | "
            f"[{row['bootstrap_ci95_lower']:.3f}, {row['bootstrap_ci95_upper']:.3f}] |"
        )
    report_lines.extend([
        "",
        "## Protocol audit",
        "",
        f"- Classes: {len(all_classes)}",
        f"- Official whitespace/token ordering disagreements: "
        f"{sum(bool(row['official_branch_disagrees_with_tokenmax']) for row in protocol_rows.values())}",
        f"- Official whitespace shape mismatches: {len(mismatch_classes)}",
        "- Mismatch synsets: " + (", ".join(sorted(mismatch_classes)) or "none"),
        "",
    ])
    (output / "REPORT.md").write_text("\n".join(report_lines), encoding="utf-8")
    (output / "summary.json").write_text(json.dumps({
        "format_version": 2,
        "performance": performance,
        "primary": primary,
        "population_sensitivity": population_contrasts,
        "conditioning_lengths": sequence_rows,
        "official_heuristic_audit": {
            "ordering_disagreement_classes": sum(
                bool(row["official_branch_disagrees_with_tokenmax"])
                for row in protocol_rows.values()
            ),
            "shape_mismatch_classes": len(mismatch_classes),
            "shape_mismatch_synsets": sorted(mismatch_classes),
        },
        "interpretation_boundary": (
            "Both arms use identical text, visual prototypes, image seeds, generation seeds, "
            "and classifier repeats. Only the valid shared SD1.5 conditioning length differs. "
            "The official whitespace heuristic is not a generation arm because it is undefined "
            "for labels where it selects the token-shorter CFG branch."
        ),
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
