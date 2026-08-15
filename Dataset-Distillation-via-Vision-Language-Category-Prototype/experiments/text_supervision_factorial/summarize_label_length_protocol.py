#!/usr/bin/env python3
"""Summarize the six-cell Label/DCS conditioning-layout experiment."""

import argparse
import ast
import csv
import hashlib
import json
import random
import re
import statistics
from pathlib import Path


RESULT = re.compile(r"Best, last acc:----(\[[^\]]+\])")
MATRIX_PROMPTS = (
    "label", "label_pad_dcs", "label_wrapped_dcs",
    "correct_t77", "correct", "correct_tokenmax_var",
)
OPTIONAL_PROMPTS = ("raw_label_tokenmax_var",)
DISPLAY = {
    "label": "a_pad77",
    "label_pad_dcs": "e_raw_empty",
    "label_wrapped_dcs": "e_wrapped",
    "correct_t77": "c_t77",
    "correct": "d_pad77_chunks",
    "correct_tokenmax_var": "d_variable_official",
    "raw_label_tokenmax_var": "a_variable_auxiliary",
}
CONTRASTS = (
    ("c_minus_a", "label", "correct_t77"),
    ("e_minus_a", "label", "label_pad_dcs"),
    ("e_wrapped_minus_a", "label", "label_wrapped_dcs"),
    ("e_wrapped_minus_e", "label_pad_dcs", "label_wrapped_dcs"),
    ("d_pad_minus_c", "correct_t77", "correct"),
    ("d_var_minus_c", "correct_t77", "correct_tokenmax_var"),
    ("d_var_minus_d_pad", "correct", "correct_tokenmax_var"),
)


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


def bootstrap_values(cells, samples, seed):
    generations = sorted(cells)
    observed = [value for generation in generations for value in cells[generation]]
    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        values = []
        for _ in generations:
            generation = rng.choice(generations)
            repeats = cells[generation]
            values.extend(rng.choice(repeats) for _ in repeats)
        estimates.append(statistics.fmean(values))
    return {
        "mean_accuracy": statistics.fmean(observed),
        "bootstrap_ci95_lower": percentile(estimates, 0.025),
        "bootstrap_ci95_upper": percentile(estimates, 0.975),
        "generation_cells": len(generations),
        "classifier_observations": len(observed),
    }


def bootstrap_paired(left, right, samples, seed):
    generations = sorted(set(left) & set(right))
    if set(left) != set(right):
        raise ValueError("Generation seeds differ across paired arms")
    paired = {}
    for generation in generations:
        if len(left[generation]) != len(right[generation]):
            raise ValueError(f"Classifier-repeat mismatch at generation seed {generation}")
        paired[generation] = [
            right_value - left_value
            for left_value, right_value in zip(left[generation], right[generation])
        ]
    result = bootstrap_values(paired, samples, seed)
    result["mean_difference"] = result.pop("mean_accuracy")
    result["paired_classifier_observations"] = result.pop("classifier_observations")
    result["bootstrap_order"] = "shared generation seed -> paired classifier repeat"
    return result


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def record_key(row):
    return row["synset"], int(row["image_index"])


def integrity_audit(index):
    rows = []
    checkpoint_identities = []
    by_seed_prompt = {}
    for entry in index:
        prompt = entry.get("prompt")
        if prompt not in MATRIX_PROMPTS:
            continue
        synthetic = Path(entry["synthetic_dir"])
        records_path = synthetic / "prompt_records.json"
        manifest_path = synthetic / "manifest.json"
        if not records_path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError(records_path if not records_path.is_file() else manifest_path)
        records = json.loads(records_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        checkpoint_identities.append(manifest["checkpoint"])
        by_seed_prompt[(int(entry["generation_seed"]), prompt)] = (
            synthetic, {record_key(row): row for row in records}
        )

    if any(identity != checkpoint_identities[0] for identity in checkpoint_identities):
        raise RuntimeError("Checkpoint identity differs across six paired arms")

    for generation in sorted({seed for seed, _ in by_seed_prompt}):
        available = {prompt for seed, prompt in by_seed_prompt if seed == generation}
        if not set(MATRIX_PROMPTS).issubset(available):
            continue
        roots = {prompt: by_seed_prompt[(generation, prompt)][0] for prompt in MATRIX_PROMPTS}
        records = {prompt: by_seed_prompt[(generation, prompt)][1] for prompt in MATRIX_PROMPTS}
        keys = set.intersection(*(set(records[prompt]) for prompt in MATRIX_PROMPTS))
        for key in sorted(keys):
            reference_chunks = int(records["correct"][key]["conditioning_chunks"])
            if int(records["label_pad_dcs"][key]["conditioning_chunks"]) != reference_chunks:
                raise RuntimeError(f"e chunk-count mismatch at seed={generation}, key={key}")
            if int(records["label_wrapped_dcs"][key]["conditioning_chunks"]) != reference_chunks:
                raise RuntimeError(f"e_wrapped chunk-count mismatch at seed={generation}, key={key}")
            relative = Path(key[0]) / f"image_{key[1]:05d}.png"
            hashes = {prompt: sha256(roots[prompt] / relative) for prompt in MATRIX_PROMPTS}
            short = reference_chunks == 1
            if short and not (
                hashes["label"] == hashes["label_pad_dcs"] == hashes["label_wrapped_dcs"]
            ):
                raise RuntimeError(f"Short-slot a/e/e_wrapped identity failed: seed={generation}, key={key}")
            rows.append({
                "generation_seed": generation,
                "synset": key[0],
                "image_index": key[1],
                "reference_chunks": reference_chunks,
                "caption_stratum": "short" if short else "long",
                "e_equals_a_png": hashes["label_pad_dcs"] == hashes["label"],
                "e_wrapped_equals_a_png": hashes["label_wrapped_dcs"] == hashes["label"],
                "e_wrapped_equals_e_png": hashes["label_wrapped_dcs"] == hashes["label_pad_dcs"],
                "d_var_equals_d_pad_png": hashes["correct_tokenmax_var"] == hashes["correct"],
            })
    return rows, checkpoint_identities[0]


def main():
    args = parse_args()
    index_path = Path(args.evaluation_index).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    present = {row.get("prompt") for row in index}
    missing = set(MATRIX_PROMPTS) - present
    if missing:
        raise RuntimeError(f"Incomplete six-cell matrix: {sorted(missing)}")
    prompts = list(MATRIX_PROMPTS) + [prompt for prompt in OPTIONAL_PROMPTS if prompt in present]

    cells = {prompt: {} for prompt in prompts}
    for row in index:
        prompt = row.get("prompt")
        if prompt in cells:
            cells[prompt][int(row["generation_seed"])] = scores(row["evaluation_log"])

    performance = []
    for offset, prompt in enumerate(prompts):
        result = bootstrap_values(cells[prompt], args.bootstrap_samples, 20260815 + offset)
        performance.append({
            "prompt": prompt,
            "cell": DISPLAY[prompt],
            "matrix_role": "primary" if prompt in MATRIX_PROMPTS else "auxiliary_prior_control",
            **result,
        })

    contrasts = []
    for offset, (name, left, right) in enumerate(CONTRASTS):
        result = bootstrap_paired(cells[left], cells[right], args.bootstrap_samples, 20260900 + offset)
        result.update({"contrast": name, "left": DISPLAY[left], "right": DISPLAY[right]})
        contrasts.append(result)

    integrity, checkpoint_identity = integrity_audit(index)
    strata = []
    for stratum in ("all", "short", "long"):
        subset = integrity if stratum == "all" else [row for row in integrity if row["caption_stratum"] == stratum]
        if subset:
            strata.append({
                "caption_stratum": stratum,
                "slots": len(subset),
                "slot_fraction": len(subset) / len(integrity),
                "e_equals_a_png_fraction": statistics.fmean(row["e_equals_a_png"] for row in subset),
                "e_wrapped_equals_a_png_fraction": statistics.fmean(row["e_wrapped_equals_a_png"] for row in subset),
                "e_wrapped_equals_e_png_fraction": statistics.fmean(row["e_wrapped_equals_e_png"] for row in subset),
                "d_var_equals_d_pad_png_fraction": statistics.fmean(row["d_var_equals_d_pad_png"] for row in subset),
            })

    write_csv(output / "six_cell_performance.csv", performance)
    write_csv(output / "six_cell_paired_contrasts.csv", contrasts)
    write_csv(output / "six_cell_integrity.csv", integrity)
    write_csv(output / "six_cell_integrity_summary.csv", strata)
    summary = {
        "format_version": 1,
        "matrix": {
            "a_pad77": "raw Label in one padded 77-position block",
            "e_raw_empty": "Label plus raw/padding-derived empty tail blocks",
            "e_wrapped": "Label plus independently BOS/EOS-wrapped empty tail blocks",
            "c_t77": "Correct DCS truncated/padded to one 77-position block",
            "d_pad77_chunks": "full Correct DCS rounded to 77-position chunks",
            "d_variable_official": "full Correct DCS at exact shared CFG token length",
        },
        "performance": performance,
        "paired_contrasts": contrasts,
        "integrity_summary": strata,
        "checkpoint_identity": checkpoint_identity,
        "interpretation_boundary": (
            "All six cells share checkpoint, visual prototypes, image seeds, generation seeds, "
            "classifier repeats, CFG scale, and denoising schedule. Tail block count for e and "
            "e_wrapped follows the paired Correct-DCS caption independently per generated slot."
        ),
    }
    (output / "six_cell_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        "# Six-cell conditioning-layout control", "", "## Performance", "",
        "| Cell | Prompt | Mean | 95% paired-bootstrap CI |", "|---|---|---:|---:|",
    ]
    for row in performance:
        lines.append(
            f"| {row['cell']} | {row['prompt']} | {row['mean_accuracy']:.3f} | "
            f"[{row['bootstrap_ci95_lower']:.3f}, {row['bootstrap_ci95_upper']:.3f}] |"
        )
    lines.extend(["", "## Paired contrasts", "", "| Contrast | Mean | 95% CI |", "|---|---:|---:|"])
    for row in contrasts:
        lines.append(
            f"| {row['contrast']} | {row['mean_difference']:.3f} | "
            f"[{row['bootstrap_ci95_lower']:.3f}, {row['bootstrap_ci95_upper']:.3f}] |"
        )
    lines.extend(["", f"Checkpoint identity: `{json.dumps(checkpoint_identity, sort_keys=True)}`", ""])
    (output / "SIX_CELL_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
