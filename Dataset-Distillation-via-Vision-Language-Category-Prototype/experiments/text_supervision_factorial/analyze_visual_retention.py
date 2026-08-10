#!/usr/bin/env python3
"""Replace raw img2img strength with measured DINO cluster retention."""

import argparse
import csv
import json
import os
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from summarize_conditioning_interface_matrix import read_scores


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-index", required=True)
    parser.add_argument("--assignment", action="append", required=True, metavar="SPEC=CSV")
    parser.add_argument("--dino-model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--random-seed", type=int, default=20260811)
    return parser.parse_args()


def read_csv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_assignments(entries):
    result = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"Expected SPEC=CSV, got {entry}")
        spec, path = entry.split("=", 1)
        result[spec] = Path(path).resolve()
    return result


def normalize(features):
    return features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)


def extract_features(paths, model_root, device, batch_size):
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel

    target = torch.device(device)
    dtype = torch.float16 if target.type == "cuda" else torch.float32
    processor = AutoImageProcessor.from_pretrained(model_root, local_files_only=True)
    model = AutoModel.from_pretrained(
        model_root, local_files_only=True, torch_dtype=dtype
    ).to(target).eval()
    output = []
    with torch.inference_mode():
        for start in range(0, len(paths), batch_size):
            images = []
            for path in paths[start:start + batch_size]:
                with Image.open(path) as image:
                    images.append(image.convert("RGB"))
            pixels = processor(images=images, return_tensors="pt")["pixel_values"].to(
                device=target, dtype=dtype
            )
            features = model(pixel_values=pixels).last_hidden_state[:, 0].float()
            output.append(torch.nn.functional.normalize(features, dim=1).cpu().numpy())
            print(f"DINO retention features: {min(start + batch_size, len(paths))}/{len(paths)}", flush=True)
    return np.concatenate(output).astype(np.float32, copy=False)


def cached_features(paths, cache_path, model_root, device, batch_size):
    paths = [str(Path(path).resolve()) for path in paths]
    cache_path = Path(cache_path)
    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as cached:
            if np.array_equal(cached["paths"].astype(str), np.asarray(paths)):
                return cached["features"].astype(np.float32, copy=False)
        raise RuntimeError(f"Feature-cache path inventory changed: {cache_path}")
    features = extract_features(paths, model_root, device, batch_size)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_name(cache_path.name + ".tmp.npz")
    np.savez_compressed(temporary, paths=np.asarray(paths), features=features)
    os.replace(temporary, cache_path)
    return features


def real_centroids(assignments, features):
    grouped = defaultdict(list)
    for row, feature in zip(assignments, features):
        grouped[(row["synset"], int(row["assigned_cluster"]))].append(feature)
    result = {}
    for key, selected in grouped.items():
        result[key] = normalize(np.mean(selected, axis=0, keepdims=True))[0]
    return result


def complete_rows(index):
    rows = []
    for row in index:
        if not row.get("synthetic_dir") or not row.get("evaluation_log"):
            continue
        synthetic = Path(row["synthetic_dir"])
        log = Path(row["evaluation_log"])
        if synthetic.is_dir() and log.is_file():
            rows.append(row)
    return rows


def prompt_records(root):
    path = Path(root) / "prompt_records.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def retention_for_cell(row, centroids, features):
    records = prompt_records(row["synthetic_dir"])
    if len(records) != len(features):
        raise ValueError(f"Prompt/image count mismatch: {row['synthetic_dir']}")
    similarities, margins, hits = [], [], []
    by_class = defaultdict(list)
    for (synset, cluster), centroid in centroids.items():
        by_class[synset].append((cluster, centroid))
    for record, feature in zip(records, features):
        synset = record["synset"]
        target = int(record["prototype_index"])
        candidates = sorted(by_class[synset])
        ids = np.asarray([item[0] for item in candidates])
        scores = np.stack([item[1] for item in candidates]) @ feature
        target_position = np.flatnonzero(ids == target)
        if len(target_position) != 1:
            raise ValueError(f"No real DINO centroid for {synset} cluster {target}")
        position = int(target_position[0])
        target_score = float(scores[position])
        similarities.append(target_score)
        margins.append(target_score - float(np.max(scores[ids != target])))
        hits.append(int(ids[int(np.argmax(scores))] == target))
    return {
        "source_centroid_cosine": statistics.fmean(similarities),
        "source_target_margin": statistics.fmean(margins),
        "source_cluster_top1": statistics.fmean(hits),
    }


def bootstrap_correlation(rows, x_key, y_key, samples, seed):
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(samples):
        selected = [rows[index] for index in rng.integers(0, len(rows), len(rows))]
        result = spearmanr(
            [row[x_key] for row in selected], [row[y_key] for row in selected]
        ).statistic
        if np.isfinite(result):
            values.append(float(result))
    values.sort()
    return values[int(0.025 * len(values))], values[min(len(values) - 1, int(0.975 * len(values)))]


def plot_mechanism(rows, correlations, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    colors = {"prototype": "tab:blue", "schedule_matched_noise": "tab:orange", "pure_noise": "tab:green"}
    for axis, utility in zip(axes, ("descriptive_marginal", "correspondence_value")):
        for visual_mode in colors:
            selected = [row for row in rows if row["visual_mode"] == visual_mode]
            if selected:
                axis.scatter(
                    [row["visual_retention_margin"] for row in selected],
                    [row[utility] for row in selected],
                    alpha=0.75, label=visual_mode, color=colors[visual_mode],
                )
        correlation = next(
            row for row in correlations
            if row["retention_metric"] == "visual_retention_margin"
            and row["utility_metric"] == utility
        )
        axis.axhline(0.0, color="black", linestyle="--", linewidth=1)
        axis.set_xlabel("Measured visual retention: DINO source-target margin")
        axis.set_ylabel(utility.replace("_", " "))
        axis.set_title(
            f"Spearman rho={correlation['spearman']:.3f} "
            f"[{correlation['bootstrap_ci_lower']:.3f}, {correlation['bootstrap_ci_upper']:.3f}]"
        )
    axes[0].legend()
    figure.suptitle("Prompt utility versus measured visual-cluster retention")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main():
    args = parse_args()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    assignments_by_spec = parse_assignments(args.assignment)
    centroids = {}
    for spec, assignment_path in assignments_by_spec.items():
        rows = read_csv(assignment_path)
        paths = [row["image_path"] for row in rows]
        features = cached_features(
            paths, output / "feature_cache" / f"real_{spec}.npz",
            args.dino_model, args.device, args.batch_size,
        )
        centroids[spec] = real_centroids(rows, features)

    index = json.loads(Path(args.evaluation_index).read_text(encoding="utf-8"))
    cells = complete_rows(index)
    retention_rows = []
    for cell_index, row in enumerate(cells):
        records = prompt_records(row["synthetic_dir"])
        paths = [
            str(Path(row["synthetic_dir"]) / record["synset"] / f"image_{int(record['image_index']):05d}.png")
            for record in records
        ]
        features = cached_features(
            paths, output / "feature_cache" / f"synthetic_{cell_index:04d}.npz",
            args.dino_model, args.device, args.batch_size,
        )
        metrics = retention_for_cell(row, centroids[row["spec"]], features)
        retention_rows.append({
            "spec": row["spec"], "visual_mode": row["visual_mode"],
            "strength": row.get("strength"), "supervision": row["supervision"],
            "training_seed": row.get("training_seed"),
            "generation_seed": int(row["generation_seed"]), "prompt": row["prompt"],
            "shuffle_shift": row.get("shuffle_shift"),
            "downstream_accuracy": statistics.fmean(read_scores(row["evaluation_log"])),
            **metrics,
        })
        print(f"Retention cell {cell_index + 1}/{len(cells)}", flush=True)
    write_csv(output / "visual_retention_per_cell.csv", retention_rows)

    grouped = defaultdict(dict)
    for row in retention_rows:
        identity = (
            row["spec"], row["visual_mode"], row["strength"], row["supervision"],
            row["training_seed"], row["generation_seed"],
        )
        key = row["prompt"] if row["prompt"] != "shuffled" else f"shuffled_{row['shuffle_shift']}"
        grouped[identity][key] = row
    mechanism_rows = []
    for identity, prompts in grouped.items():
        if not {"label", "correct", "shuffled_1"} <= set(prompts):
            continue
        shuffled = [row for key, row in prompts.items() if key.startswith("shuffled_")]
        descriptive_accuracy = statistics.fmean(
            [prompts["correct"]["downstream_accuracy"], *[row["downstream_accuracy"] for row in shuffled]]
        )
        correspondence_accuracy = prompts["correct"]["downstream_accuracy"] - statistics.fmean(
            row["downstream_accuracy"] for row in shuffled
        )
        mechanism_rows.append({
            "spec": identity[0], "visual_mode": identity[1], "strength": identity[2],
            "supervision": identity[3], "training_seed": identity[4],
            "generation_seed": identity[5],
            "visual_retention_top1": prompts["label"]["source_cluster_top1"],
            "visual_retention_margin": prompts["label"]["source_target_margin"],
            "visual_retention_cosine": prompts["label"]["source_centroid_cosine"],
            "descriptive_marginal": descriptive_accuracy - prompts["label"]["downstream_accuracy"],
            "correspondence_value": correspondence_accuracy,
        })
    write_csv(output / "retention_vs_prompt_utility.csv", mechanism_rows)

    correlations = []
    for retention in ("visual_retention_top1", "visual_retention_margin", "visual_retention_cosine"):
        for utility in ("descriptive_marginal", "correspondence_value"):
            value = float(spearmanr(
                [row[retention] for row in mechanism_rows],
                [row[utility] for row in mechanism_rows],
            ).statistic)
            lower, upper = bootstrap_correlation(
                mechanism_rows, retention, utility, args.bootstrap_samples,
                args.random_seed + len(correlations),
            )
            correlations.append({
                "retention_metric": retention, "utility_metric": utility,
                "spearman": value, "bootstrap_ci_lower": lower,
                "bootstrap_ci_upper": upper, "cells": len(mechanism_rows),
            })
    write_csv(output / "retention_utility_correlations.csv", correlations)
    plot_mechanism(mechanism_rows, correlations, output / "retention_vs_prompt_utility.png")
    (output / "summary.json").write_text(json.dumps({
        "format_version": 1,
        "retention_definition": "Label-conditioned generated-image recovery by real-image DINO centroids",
        "cells": len(retention_rows), "mechanism_cells": len(mechanism_rows),
        "correlation_bootstrap_unit": "complete training-seed x generation-seed x interface cell",
        "correlations": correlations,
        "interpretation_boundary": "Associations are diagnostic; strength and visual mode are not randomized through DINO retention itself.",
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
