#!/usr/bin/env python3
"""Summarize B-0 conditioning-content and sequence-length controls."""

import argparse
import ast
import csv
import hashlib
import json
import random
import re
import statistics
from pathlib import Path

import matplotlib.pyplot as plt


RESULT = re.compile(r"Best, last acc:----(\[[^\]]+\])")
FAMILIES = ("label_ft", "matched_ft", "sparse_m4_ft")
MATCHED_PROMPTS = (
    "label", "first_sentence", "correct_t77", "correct", "label_pad_dcs",
    "correct_t77_pad_dcs", "correct_head_pad_dcs",
)
CONTROL_PROMPTS = ("label", "correct_t77", "correct", "correct_head_pad_dcs")
FAMILY_PROMPTS = {
    "label_ft": CONTROL_PROMPTS,
    "matched_ft": MATCHED_PROMPTS,
    "sparse_m4_ft": CONTROL_PROMPTS,
}
CONTRASTS = (
    ("first_sentence_minus_label", "first_sentence", "label"),
    ("correct_t77_minus_label", "correct_t77", "label"),
    ("full_minus_t77", "correct", "correct_t77"),
    ("label_pad_dcs_minus_label", "label_pad_dcs", "label"),
    ("full_minus_label_pad_dcs", "correct", "label_pad_dcs"),
    ("t77_pad_dcs_minus_t77", "correct_t77_pad_dcs", "correct_t77"),
    ("full_minus_t77_pad_dcs", "correct", "correct_t77_pad_dcs"),
    ("full_minus_head_pad_dcs", "correct", "correct_head_pad_dcs"),
    ("head_pad_dcs_minus_t77_pad_dcs", "correct_head_pad_dcs", "correct_t77_pad_dcs"),
    ("full_minus_first_sentence", "correct", "first_sentence"),
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-index", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--equivalence-margin", type=float, default=1.0)
    return parser.parse_args()


def scores(path):
    matches = RESULT.findall(Path(path).read_text(encoding="utf-8", errors="replace"))
    if not matches:
        raise ValueError(f"No completed classifier result: {path}")
    return [float(value) for value in ast.literal_eval(matches[-1])]


def percentile(values, probability):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(probability * len(ordered)))]


def bootstrap_generation(rows, value_key, samples, seed):
    grouped = {row["generation_seed"]: row[value_key] for row in rows}
    generations = sorted(grouped)
    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        values = []
        for _ in generations:
            generation = rng.choice(generations)
            repeats = grouped[generation]
            values.extend(rng.choice(repeats) for _ in repeats)
        estimates.append(statistics.fmean(values))
    return (
        percentile(estimates, 0.025), percentile(estimates, 0.975),
        percentile(estimates, 0.05), percentile(estimates, 0.95),
    )


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def conditioning_audit(index_path, rows):
    run_root = Path(index_path).resolve().parent
    audit = []
    for family in FAMILIES:
        supervision = "sparse_ft" if family == "sparse_m4_ft" else family
        for generation_seed in sorted({
            row["generation_seed"] for row in rows
            if row["checkpoint_family"] == family
        }):
            for prompt in FAMILY_PROMPTS[family]:
                path = (
                    run_root / "synthetic" / family / f"seed_{generation_seed}"
                    / f"{supervision}_{prompt}" / "prompt_records.json"
                )
                records = json.loads(path.read_text(encoding="utf-8"))
                lengths = [int(row["conditioning_sequence_length"]) for row in records]
                reference = [int(row["reference_dcs_chunks"]) for row in records]
                audit.append({
                    "checkpoint_family": family,
                    "generation_seed": generation_seed,
                    "prompt": prompt,
                    "records": len(records),
                    "sequence_length_mean": statistics.fmean(lengths),
                    "sequence_length_min": min(lengths),
                    "sequence_length_max": max(lengths),
                    "reference_dcs_chunks_mean": statistics.fmean(reference),
                    "length_matches_reference_fraction": statistics.fmean(
                        length == chunks * 77 for length, chunks in zip(lengths, reference)
                    ),
                })
    return audit


def caption_strata_audit(index_path, rows):
    """Audit per-slot chunk matching and short-caption identity without refitting a classifier."""
    run_root = Path(index_path).resolve().parent
    output = []
    for family in FAMILIES:
        supervision = "sparse_ft" if family == "sparse_m4_ft" else family
        for seed in sorted({row["generation_seed"] for row in rows if row["checkpoint_family"] == family}):
            records_by_prompt = {}
            for prompt in FAMILY_PROMPTS[family]:
                condition_root = run_root / "synthetic" / family / f"seed_{seed}" / f"{supervision}_{prompt}"
                records = json.loads((condition_root / "prompt_records.json").read_text(encoding="utf-8"))
                records_by_prompt[prompt] = {
                    (row["synset"], int(row["image_index"])): (row, condition_root)
                    for row in records
                }
            keys = set(records_by_prompt["correct"])
            if any(set(records) != keys for records in records_by_prompt.values()):
                raise RuntimeError(f"Prompt-record slots differ for {family} seed {seed}")
            for key in sorted(keys):
                reference_chunks = {
                    int(records_by_prompt[prompt][key][0]["reference_dcs_chunks"])
                    for prompt in FAMILY_PROMPTS[family]
                }
                if len(reference_chunks) != 1:
                    raise RuntimeError(f"Reference chunk count drift for {family} seed {seed} {key}")
                chunks = reference_chunks.pop()
                for prompt in FAMILY_PROMPTS[family]:
                    record = records_by_prompt[prompt][key][0]
                    expected = chunks if prompt in {
                        "correct", "label_pad_dcs", "correct_t77_pad_dcs",
                        "correct_head_pad_dcs",
                    } else 1
                    if int(record["conditioning_chunks"]) != expected:
                        raise RuntimeError(
                            f"Per-slot chunk mismatch for {family} seed {seed} {key} {prompt}: "
                            f"{record['conditioning_chunks']} != {expected}"
                        )
                if not {"correct_t77", "correct", "correct_head_pad_dcs"}.issubset(records_by_prompt):
                    continue
                hashes = {}
                hash_prompts = ["correct_t77", "correct", "correct_head_pad_dcs"]
                if "correct_t77_pad_dcs" in records_by_prompt:
                    hash_prompts.append("correct_t77_pad_dcs")
                if "label_pad_dcs" in records_by_prompt:
                    hash_prompts.extend(["label", "label_pad_dcs"])
                for prompt in hash_prompts:
                    _, condition_root = records_by_prompt[prompt][key]
                    hashes[prompt] = sha256(condition_root / key[0] / f"image_{key[1]:05d}.png")
                row = {
                    "checkpoint_family": family,
                    "generation_seed": seed,
                    "synset": key[0],
                    "image_index": key[1],
                    "reference_dcs_chunks": chunks,
                    "caption_stratum": "long" if chunks > 1 else "short",
                    "d_equals_g_png": hashes["correct"] == hashes["correct_head_pad_dcs"],
                    "d_equals_c_png": hashes["correct"] == hashes["correct_t77"],
                    "g_equals_c_png": hashes["correct_head_pad_dcs"] == hashes["correct_t77"],
                    "d_equals_f_png": "",
                    "g_equals_f_png": "",
                    "f_equals_c_png": "",
                    "e_equals_a_png": "",
                }
                if "correct_t77_pad_dcs" in hashes:
                    row.update({
                        "d_equals_f_png": hashes["correct"] == hashes["correct_t77_pad_dcs"],
                        "g_equals_f_png": hashes["correct_head_pad_dcs"] == hashes["correct_t77_pad_dcs"],
                        "f_equals_c_png": hashes["correct_t77_pad_dcs"] == hashes["correct_t77"],
                    })
                if "label_pad_dcs" in hashes:
                    row["e_equals_a_png"] = hashes["label_pad_dcs"] == hashes["label"]
                output.append(row)
    return output


GUIDANCE_METRICS = (
    "epsilon_cond_l2", "epsilon_cond_rms",
    "epsilon_uncond_l2", "epsilon_uncond_rms",
    "epsilon_residual_l2", "epsilon_residual_rms",
)


def guidance_branch_audit(index_path, rows):
    run_root = Path(index_path).resolve().parent
    raw, aggregates = [], []
    for family in FAMILIES:
        supervision = "sparse_ft" if family == "sparse_m4_ft" else family
        seeds = sorted({
            row["generation_seed"] for row in rows if row["checkpoint_family"] == family
        })
        for seed in seeds:
            for prompt in FAMILY_PROMPTS[family]:
                path = (
                    run_root / "synthetic" / family / f"seed_{seed}"
                    / f"{supervision}_{prompt}" / "prompt_records.json"
                )
                for record in json.loads(path.read_text(encoding="utf-8")):
                    for diagnostic in record.get("guidance_diagnostics", []):
                        raw.append({
                            "checkpoint_family": family,
                            "generation_seed": seed,
                            "prompt": prompt,
                            "synset": record["synset"],
                            "image_index": int(record["image_index"]),
                            "reference_dcs_chunks": int(record["reference_dcs_chunks"]),
                            **diagnostic,
                        })
    if not raw:
        return [], []
    groups = {}
    for row in raw:
        key = (
            row["checkpoint_family"], row["prompt"],
            row["requested_timestep"], row["actual_timestep"],
        )
        groups.setdefault(key, []).append(row)
    for (family, prompt, requested, actual), selected in sorted(groups.items()):
        aggregates.append({
            "checkpoint_family": family,
            "prompt": prompt,
            "requested_timestep": requested,
            "actual_timestep": actual,
            "absolute_timestep_error": abs(actual - requested),
            "exact_requested_timestep_fraction": statistics.fmean(
                row["exact_requested_timestep"] for row in selected
            ),
            "observations": len(selected),
            **{
                f"{metric}_mean": statistics.fmean(row[metric] for row in selected)
                for metric in GUIDANCE_METRICS
            },
        })

    lookup = {
        (
            row["checkpoint_family"], row["generation_seed"], row["prompt"],
            row["synset"], row["image_index"], row["requested_timestep"],
        ): row
        for row in raw
    }
    definitions = (
        ("e_minus_a", "matched_ft", "label_pad_dcs", "label"),
        ("d_minus_g", None, "correct", "correct_head_pad_dcs"),
        ("g_minus_f", "matched_ft", "correct_head_pad_dcs", "correct_t77_pad_dcs"),
        ("d_minus_f", "matched_ft", "correct", "correct_t77_pad_dcs"),
    )
    contrast_rows = []
    for name, required_family, left, right in definitions:
        families = (required_family,) if required_family else FAMILIES
        for family in families:
            selected_by_timestep = {}
            for key, left_row in lookup.items():
                row_family, seed, prompt, synset, image_index, requested = key
                if row_family != family or prompt != left or left_row["reference_dcs_chunks"] <= 1:
                    continue
                right_row = lookup.get((family, seed, right, synset, image_index, requested))
                if right_row is None:
                    raise RuntimeError(f"Missing guidance pair for {name}: {key}")
                if left_row["actual_timestep"] != right_row["actual_timestep"]:
                    raise RuntimeError(f"Actual guidance timestep drift for {name}: {key}")
                selected_by_timestep.setdefault(requested, []).append((left_row, right_row))
            for requested, selected in sorted(selected_by_timestep.items()):
                for metric in GUIDANCE_METRICS:
                    values = [
                        left_row[metric] - right_row[metric]
                        for left_row, right_row in selected
                    ]
                    contrast_rows.append({
                        "checkpoint_family": family,
                        "contrast": name,
                        "caption_stratum": "long",
                        "metric": metric,
                        "requested_timestep": requested,
                        "actual_timestep": selected[0][0]["actual_timestep"],
                        "mean_difference": statistics.fmean(values),
                        "paired_observations": len(values),
                    })
    return aggregates, contrast_rows


def main():
    args = parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    raw = json.loads(Path(args.evaluation_index).read_text(encoding="utf-8"))
    cells, lookup = [], {}
    for row in raw:
        values = scores(row["evaluation_log"])
        cell = {**row, "scores": values, "mean_accuracy": statistics.fmean(values)}
        cells.append(cell)
        lookup[(row["checkpoint_family"], row["generation_seed"], row["prompt"])] = values
    expected = {
        (family, seed, prompt)
        for family in FAMILIES
        for seed in sorted({
            row["generation_seed"] for row in cells if row["checkpoint_family"] == family
        })
        for prompt in FAMILY_PROMPTS[family]
    }
    missing = sorted(expected - set(lookup))
    if missing:
        raise RuntimeError(f"Incomplete B-0 matrix; missing {missing}")

    performance = []
    for family in FAMILIES:
        for prompt in FAMILY_PROMPTS[family]:
            selected = [
                row for row in cells
                if row["checkpoint_family"] == family and row["prompt"] == prompt
            ]
            lower, upper, _, _ = bootstrap_generation(
                selected, "scores", args.bootstrap_samples, 20260901
            )
            values = [value for row in selected for value in row["scores"]]
            performance.append({
                "checkpoint_family": family, "prompt": prompt,
                "mean_accuracy": statistics.fmean(values),
                "bootstrap_ci95_lower": lower, "bootstrap_ci95_upper": upper,
                "generation_cells": len(selected), "classifier_observations": len(values),
            })

    contrasts = []
    for family_index, family in enumerate(FAMILIES):
        for contrast_index, (name, left, right) in enumerate(CONTRASTS):
            if left not in FAMILY_PROMPTS[family] or right not in FAMILY_PROMPTS[family]:
                continue
            paired = []
            for generation_seed in sorted({
                row["generation_seed"] for row in cells
                if row["checkpoint_family"] == family
            }):
                a = lookup[(family, generation_seed, left)]
                b = lookup[(family, generation_seed, right)]
                paired.append({
                    "generation_seed": generation_seed,
                    "differences": [x - y for x, y in zip(a, b)],
                })
            lower, upper, lower90, upper90 = bootstrap_generation(
                paired, "differences", args.bootstrap_samples,
                20260910 + family_index * 20 + contrast_index,
            )
            values = [value for row in paired for value in row["differences"]]
            contrasts.append({
                "checkpoint_family": family, "contrast": name,
                "mean_difference": statistics.fmean(values),
                "bootstrap_ci95_lower": lower, "bootstrap_ci95_upper": upper,
                "bootstrap_ci90_lower": lower90, "bootstrap_ci90_upper": upper90,
                "equivalent_within_1pt_by_90pct_ci": (
                    lower90 >= -args.equivalence_margin and upper90 <= args.equivalence_margin
                ),
                "noninferior_within_1pt_by_95pct_ci": lower >= -args.equivalence_margin,
                "generation_cells": len(paired), "paired_classifier_observations": len(values),
            })

        if family != "matched_ft":
            continue
        paired = []
        for generation_seed in sorted({
            row["generation_seed"] for row in cells
            if row["checkpoint_family"] == family
        }):
            f = lookup[(family, generation_seed, "correct_t77_pad_dcs")]
            c = lookup[(family, generation_seed, "correct_t77")]
            e = lookup[(family, generation_seed, "label_pad_dcs")]
            a = lookup[(family, generation_seed, "label")]
            paired.append({
                "generation_seed": generation_seed,
                "differences": [
                    (f_value - c_value) - (e_value - a_value)
                    for f_value, c_value, e_value, a_value in zip(f, c, e, a)
                ],
            })
        lower, upper, lower90, upper90 = bootstrap_generation(
            paired, "differences", args.bootstrap_samples, 20260990 + family_index,
        )
        values = [value for row in paired for value in row["differences"]]
        contrasts.append({
            "checkpoint_family": family,
            "contrast": "length_x_chunk1_content_interaction",
            "mean_difference": statistics.fmean(values),
            "bootstrap_ci95_lower": lower, "bootstrap_ci95_upper": upper,
            "bootstrap_ci90_lower": lower90, "bootstrap_ci90_upper": upper90,
            "equivalent_within_1pt_by_90pct_ci": (
                lower90 >= -args.equivalence_margin and upper90 <= args.equivalence_margin
            ),
            "noninferior_within_1pt_by_95pct_ci": lower >= -args.equivalence_margin,
            "generation_cells": len(paired), "paired_classifier_observations": len(values),
        })

    audit = conditioning_audit(args.evaluation_index, cells)
    strata = caption_strata_audit(args.evaluation_index, cells)
    guidance_audit, guidance_contrasts = guidance_branch_audit(args.evaluation_index, cells)
    for family in FAMILIES:
        family_audit = [row for row in audit if row["checkpoint_family"] == family]
        for seed in sorted({row["generation_seed"] for row in family_audit}):
            by_prompt = {
                row["prompt"]: row for row in family_audit if row["generation_seed"] == seed
            }
            if by_prompt["label"]["sequence_length_max"] != 77:
                raise RuntimeError("Label condition is not one 77-position block")
            if "first_sentence" in by_prompt and by_prompt["first_sentence"]["sequence_length_max"] != 77:
                raise RuntimeError("First-sentence condition is not one 77-position block")
            if by_prompt["correct_t77"]["sequence_length_max"] != 77:
                raise RuntimeError("T77 condition is not one 77-position block")
            if by_prompt["correct"]["length_matches_reference_fraction"] != 1.0:
                raise RuntimeError("Full DCS does not match its reference chunk count")
            if "label_pad_dcs" in by_prompt and by_prompt["label_pad_dcs"]["length_matches_reference_fraction"] != 1.0:
                raise RuntimeError("Padded Label does not match its reference DCS chunk count")
            if "correct_t77_pad_dcs" in by_prompt and by_prompt["correct_t77_pad_dcs"]["length_matches_reference_fraction"] != 1.0:
                raise RuntimeError("Padded T77 DCS does not match its reference DCS chunk count")
            if by_prompt["correct_head_pad_dcs"]["length_matches_reference_fraction"] != 1.0:
                raise RuntimeError("Raw-head DCS control does not match its reference chunk count")

    write_csv(output / "conditioning_length_audit.csv", audit)
    write_csv(output / "caption_length_strata.csv", strata)
    strata_summary = []
    for family in FAMILIES:
        selected = [row for row in strata if row["checkpoint_family"] == family]
        for stratum in ("all", "short", "long"):
            subset = selected if stratum == "all" else [
                row for row in selected if row["caption_stratum"] == stratum
            ]
            if not subset:
                continue
            strata_summary.append({
                "checkpoint_family": family, "caption_stratum": stratum,
                "slots": len(subset),
                "slot_fraction": len(subset) / len(selected),
                "d_equals_g_png_fraction": statistics.fmean(row["d_equals_g_png"] for row in subset),
                "d_equals_c_png_fraction": statistics.fmean(row["d_equals_c_png"] for row in subset),
                "g_equals_c_png_fraction": statistics.fmean(row["g_equals_c_png"] for row in subset),
                "d_equals_f_png_fraction": (
                    statistics.fmean(row["d_equals_f_png"] for row in subset)
                    if all(isinstance(row["d_equals_f_png"], bool) for row in subset) else ""
                ),
                "g_equals_f_png_fraction": (
                    statistics.fmean(row["g_equals_f_png"] for row in subset)
                    if all(isinstance(row["g_equals_f_png"], bool) for row in subset) else ""
                ),
                "f_equals_c_png_fraction": (
                    statistics.fmean(row["f_equals_c_png"] for row in subset)
                    if all(isinstance(row["f_equals_c_png"], bool) for row in subset) else ""
                ),
                "e_equals_a_png_fraction": (
                    statistics.fmean(row["e_equals_a_png"] for row in subset)
                    if all(isinstance(row["e_equals_a_png"], bool) for row in subset) else ""
                ),
            })
    for row in strata_summary:
        if row["caption_stratum"] != "short":
            continue
        required = ("d_equals_g_png_fraction", "d_equals_c_png_fraction", "g_equals_c_png_fraction")
        if any(row[key] != 1.0 for key in required):
            raise RuntimeError(
                f"Short-caption c/d/g outputs are not byte-identical for {row['checkpoint_family']}"
            )
        if row["checkpoint_family"] == "matched_ft" and any(
            row[key] != 1.0
            for key in (
                "d_equals_f_png_fraction", "g_equals_f_png_fraction",
                "f_equals_c_png_fraction", "e_equals_a_png_fraction",
            )
        ):
            raise RuntimeError(
                "Short-caption c/d/f/g and a/e outputs are not byte-identical for matched_ft"
            )

    long_fraction_by_family = {
        family: next(
            row["slot_fraction"] for row in strata_summary
            if row["checkpoint_family"] == family and row["caption_stratum"] == "long"
        )
        for family in FAMILIES
    }
    tail_contrasts = {
        "label_pad_dcs_minus_label", "full_minus_t77_pad_dcs", "full_minus_head_pad_dcs",
        "head_pad_dcs_minus_t77_pad_dcs",
        "length_x_chunk1_content_interaction",
    }
    for row in contrasts:
        is_tail_contrast = row["contrast"] in tail_contrasts
        row["estimand_population"] = (
            "long_slot_partial_population" if is_tail_contrast else "full_population"
        )
        row["long_caption_slot_fraction"] = (
            long_fraction_by_family[row["checkpoint_family"]] if is_tail_contrast else ""
        )
    write_csv(output / "performance.csv", performance)
    write_csv(output / "paired_contrasts.csv", contrasts)
    write_csv(output / "caption_length_strata_summary.csv", strata_summary)
    write_csv(output / "guidance_branch_norms.csv", guidance_audit)
    write_csv(output / "guidance_branch_contrasts.csv", guidance_contrasts)
    summary = {
        "format_version": 3,
        "performance": performance,
        "paired_contrasts": contrasts,
        "conditioning_length_audit": audit,
        "caption_length_strata_summary": strata_summary,
        "guidance_branch_norms": guidance_audit,
        "guidance_branch_contrasts": guidance_contrasts,
        "estimand_boundary": (
            "Downstream c-a is a full-population one-chunk contrast. Downstream e-a, d-g, g-f, "
            "d-f, and (f-c)-(e-a) are set-level effects whose intervention occurs only on long "
            "caption slots; short slots are verified byte-identical for c/d/f/g and a/e. These "
            "are already causal partial-population interventions with short slots held fixed. A "
            "separate long-only accuracy cannot be recovered from the same classifier fit because "
            "dataset training is nonlinear and set-valued; constructing a long-only dataset would "
            "change class balance and define a different estimand."
        ),
        "bootstrap_order": "generation seed -> shared paired classifier-repeat draw",
        "interpretation": {
            "full_minus_t77_near_zero": "extra DCS chunks add no downstream value",
            "label_pad_dcs_minus_label_positive": "extra padding-derived KV positions have structural value",
            "full_minus_label_pad_dcs_positive": "DCS content beyond length alone has value",
            "length_x_chunk1_content_interaction": (
                "difference-in-differences: (T77 DCS + empty tail - T77 DCS) - "
                "(Label + empty tail - Label)"
            ),
            "full_minus_head_pad_dcs_positive": (
                "real DCS tail content has value after exactly matching the raw first block, "
                "negative branch, positions, and chunk count"
            ),
            "head_pad_dcs_minus_t77_pad_dcs": (
                "isolates the raw-slice versus tokenizer-truncated first-block boundary convention"
            ),
            "full_minus_first_sentence_near_zero": "the first DCS sentence is sufficient",
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    x = list(range(len(MATCHED_PROMPTS)))
    for family in FAMILIES:
        rows = [
            next(row for row in performance if row["checkpoint_family"] == family and row["prompt"] == prompt)
            for prompt in MATCHED_PROMPTS
            if prompt in FAMILY_PROMPTS[family]
        ]
        family_x = [
            index for index, prompt in enumerate(MATCHED_PROMPTS)
            if prompt in FAMILY_PROMPTS[family]
        ]
        axes[0].errorbar(
            family_x, [row["mean_accuracy"] for row in rows],
            yerr=[
                [row["mean_accuracy"] - row["bootstrap_ci95_lower"] for row in rows],
                [row["bootstrap_ci95_upper"] - row["mean_accuracy"] for row in rows],
            ], marker="o", capsize=4, label=family,
        )
    axes[0].set_xticks(x, MATCHED_PROMPTS, rotation=18)
    axes[0].set_ylabel("Validation accuracy")
    axes[0].set_title("Content and sequence-length controls")
    axes[0].legend()
    key_names = (
        "correct_t77_minus_label", "label_pad_dcs_minus_label",
        "length_x_chunk1_content_interaction", "full_minus_head_pad_dcs",
        "head_pad_dcs_minus_t77_pad_dcs",
    )
    positions = list(range(len(key_names)))
    width = 0.24
    for family_index, family in enumerate(FAMILIES):
        available_names = {
            row["contrast"] for row in contrasts if row["checkpoint_family"] == family
        }
        if not set(key_names).issubset(available_names):
            continue
        rows = [
            next(row for row in contrasts if row["checkpoint_family"] == family and row["contrast"] == name)
            for name in key_names
        ]
        axes[1].bar(
            [position + (family_index - 1) * width for position in positions],
            [row["mean_difference"] for row in rows], width=width, label=family,
        )
    axes[1].set_xticks(
        positions,
        ("c-a\nchunk-1 content", "e-a\nextra positions", "(f-c)-(e-a)\ninteraction",
         "d-g\nreal tail", "g-f\nboundary"),
    )
    axes[1].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[1].set_ylabel("Paired accuracy difference")
    axes[1].set_title("Semantic content versus extra KV positions")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output / "b0_conditioning_length.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
