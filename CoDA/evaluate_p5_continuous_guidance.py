"""Evaluate how continuous CoDA guidance changes frozen-SDXL text execution."""

import argparse
import csv
import json
import math
import os
import pickle
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_p4_feature_displacements import cosine, target_specificity
from evaluate_p4_text_execution import (
    PROBES,
    atomic_json,
    load_or_extract,
    load_prompt_records,
    paired_metrics,
    read_csv,
    summarize_recovery,
    write_csv,
)


REGIMES = ("i0g0", "i0g1", "i1g0", "i1g1")
PROMPTS = ("label", "correct", "shuffled")
EFFECT_METRICS = (
    "label_target_margin", "correct_target_margin", "delta_target", "delta_pull",
    "caption_rank_improvement", "visual_rank_drop",
)
GEOMETRY_METRICS = (
    "text_norm_log_ratio", "swap_norm_log_ratio", "target_alignment_delta",
    "target_specificity_delta", "target_residual_delta", "label_guidance_displacement_norm",
)


def parse_args():
    parser = argparse.ArgumentParser(description="P5 continuous-guidance diagnostic")
    parser.add_argument("--prepared-dir", required=True)
    parser.add_argument("--generation-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dino-model", required=True)
    parser.add_argument("--clip-model", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--permutation-samples", type=int, default=1000)
    parser.add_argument("--random-seed", type=int, default=20260804)
    return parser.parse_args()


def inventory(prepared_dir, manifest_path):
    pairs = read_csv(Path(prepared_dir) / "pair_manifest.csv")
    manifests = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    expected = {(regime, prompt) for regime in REGIMES for prompt in PROMPTS}
    datasets = {}
    by_spec_seed = defaultdict(set)
    for row in manifests:
        key = (
            row["spec"], int(row["generation_seed"]),
            row["visual_mode"], row["prompt_condition"],
        )
        if key in datasets:
            raise ValueError(f"Duplicate P5 generation entry: {key}")
        datasets[key] = Path(row["dataset_dir"])
        by_spec_seed[(key[0], key[1])].add((key[2], key[3]))
    for key, actual in by_spec_seed.items():
        if actual != expected:
            raise ValueError(f"Incomplete P5 factorial matrix for {key}: {sorted(actual)}")

    pair_lookup = defaultdict(list)
    for row in pairs:
        converted = dict(row)
        for column in (
            "visual_cluster_id", "correct_caption_cluster_id", "shuffled_caption_cluster_id"
        ):
            converted[column] = int(converted[column])
        pair_lookup[converted["spec"]].append(converted)

    samples = []
    for key in sorted(datasets):
        spec, generation_seed, regime, prompt_condition = key
        dataset_dir = datasets[key]
        prompts = load_prompt_records(dataset_dir)
        if regime.endswith("g1"):
            raw_guidance = dataset_dir / "guidance_metrics" / "guidance_metrics_raw.csv"
            if not raw_guidance.is_file():
                raise FileNotFoundError(f"P5 G1 condition lacks guidance metrics: {raw_guidance}")
        for pair in pair_lookup[spec]:
            k = pair["visual_cluster_id"]
            path = dataset_dir / pair["class_id"] / f"{k}.png"
            if not path.is_file():
                raise FileNotFoundError(f"Missing P5 generated image: {path}")
            prompt = prompts.get((pair["class_id"], k))
            if prompt is None:
                raise ValueError(f"Missing P5 prompt record for {path}")
            expected_caption = None
            if prompt_condition == "correct":
                expected_caption = pair["correct_caption_cluster_id"]
            elif prompt_condition == "shuffled":
                expected_caption = pair["shuffled_caption_cluster_id"]
            if prompt.get("caption_cluster_id") != expected_caption:
                raise ValueError(f"P5 caption source mismatch for {path}")
            samples.append(
                {
                    **pair,
                    "generation_seed": generation_seed,
                    "visual_mode": regime,
                    "prompt_condition": prompt_condition,
                    "image_seed": int(prompt["image_seed"]),
                    "path": str(path.resolve()),
                    "dataset_dir": str(dataset_dir.resolve()),
                }
            )

    paired = defaultdict(dict)
    cross_regime = defaultdict(set)
    for sample in samples:
        pair_key = (
            sample["spec"], sample["class_key"], sample["visual_cluster_id"],
            sample["generation_seed"], sample["visual_mode"],
        )
        paired[pair_key][sample["prompt_condition"]] = sample
        seed_key = pair_key[:4]
        cross_regime[seed_key].add(sample["image_seed"])
    for key, rows in paired.items():
        if set(rows) != set(PROMPTS):
            raise ValueError(f"Incomplete P5 prompt triplet for {key}")
        if len({row["image_seed"] for row in rows.values()}) != 1:
            raise ValueError(f"P5 prompt triplet does not share image seed for {key}")
    for key, seeds in cross_regime.items():
        if len(seeds) != 1:
            raise ValueError(f"P5 regimes do not share image seed for {key}: {seeds}")
    return samples, paired, datasets


def bootstrap_grouped(rows, value_key, samples, seed):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["class_key"], row["visual_cluster_id"])].append(float(row[value_key]))
    values = np.asarray([np.mean(group) for group in groups.values()], dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = np.mean(rng.choice(values, size=(samples, len(values)), replace=True), axis=1)
    return {
        "mean": float(np.mean(values)),
        "bootstrap_ci_lower": float(np.quantile(draws, 0.025)),
        "bootstrap_ci_upper": float(np.quantile(draws, 0.975)),
        "class_cluster_groups": len(values),
        "raw_observations": len(rows),
        "positive_group_fraction": float(np.mean(values > 0)),
    }


def summarize_regime_effects(rows, bootstrap_samples, seed):
    output = []
    counter = 0
    scopes = ["combined"] + sorted({row["spec"] for row in rows})
    for scope in scopes:
        scoped = rows if scope == "combined" else [row for row in rows if row["spec"] == scope]
        for encoder in sorted({row["encoder"] for row in scoped}):
            for probe in PROBES:
                for regime in REGIMES:
                    selected = [
                        row for row in scoped if row["encoder"] == encoder
                        and row["probe"] == probe and row["visual_mode"] == regime
                    ]
                    for metric in EFFECT_METRICS:
                        stats = bootstrap_grouped(selected, metric, bootstrap_samples, seed + counter)
                        counter += 1
                        output.append({
                            "scope": scope, "encoder": encoder, "probe": probe,
                            "visual_mode": regime, "metric": metric, **stats,
                        })
    return output


def guidance_interaction_rows(effect_rows):
    indexed = defaultdict(dict)
    for row in effect_rows:
        key = (
            row["encoder"], row["probe"], row["spec"], row["class_key"],
            row["class_id"], row["class_name"], row["visual_cluster_id"],
            row["caption_source_cluster_id"], row["generation_seed"], row["image_seed"],
        )
        indexed[key][row["visual_mode"]] = row
    output = []
    for key, regimes in sorted(indexed.items()):
        if set(regimes) != set(REGIMES):
            raise ValueError(f"Incomplete P5 effects for {key}: {sorted(regimes)}")
        base = dict(zip((
            "encoder", "probe", "spec", "class_key", "class_id", "class_name",
            "visual_cluster_id", "caption_source_cluster_id", "generation_seed", "image_seed",
        ), key))
        init_rows = {}
        for initialization, g0, g1 in (("i0", "i0g0", "i0g1"), ("i1", "i1g0", "i1g1")):
            record = {**base, "initialization": initialization}
            for metric in EFFECT_METRICS:
                record[f"g0_{metric}"] = float(regimes[g0][metric])
                record[f"g1_{metric}"] = float(regimes[g1][metric])
                record[f"guidance_interaction_{metric}"] = (
                    float(regimes[g1][metric]) - float(regimes[g0][metric])
                )
                record[f"three_way_{metric}"] = ""
            output.append(record)
            init_rows[initialization] = record
        three_way = {**base, "initialization": "i1_minus_i0"}
        for metric in EFFECT_METRICS:
            three_way[f"g0_{metric}"] = ""
            three_way[f"g1_{metric}"] = ""
            three_way[f"guidance_interaction_{metric}"] = ""
            column = f"guidance_interaction_{metric}"
            three_way[f"three_way_{metric}"] = (
                init_rows["i1"][column] - init_rows["i0"][column]
            )
        output.append(three_way)
    return output


def summarize_guidance_interactions(rows, bootstrap_samples, seed):
    output = []
    counter = 0
    for scope in ["combined"] + sorted({row["spec"] for row in rows}):
        scoped = rows if scope == "combined" else [row for row in rows if row["spec"] == scope]
        for encoder in sorted({row["encoder"] for row in scoped}):
            for probe in PROBES:
                for initialization in ("i0", "i1", "i1_minus_i0"):
                    selected = [
                        row for row in scoped if row["encoder"] == encoder
                        and row["probe"] == probe and row["initialization"] == initialization
                    ]
                    for metric in EFFECT_METRICS:
                        prefix = "three_way" if initialization == "i1_minus_i0" else "guidance_interaction"
                        column = f"{prefix}_{metric}"
                        stats = bootstrap_grouped(selected, column, bootstrap_samples, seed + counter)
                        counter += 1
                        output.append({
                            "scope": scope, "encoder": encoder, "probe": probe,
                            "initialization": initialization, "metric": metric,
                            "contrast": "G1_minus_G0" if initialization != "i1_minus_i0" else "I1_minus_I0_of_G1_minus_G0",
                            **stats,
                        })
    return output


def summarize_interactions_by_generation_seed(rows):
    """Keep seed-level reversals visible instead of only reporting pooled effects."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[(
            row["encoder"], row["probe"], row["generation_seed"], row["initialization"],
        )].append(row)
    output = []
    for (encoder, probe, generation_seed, initialization), selected in sorted(grouped.items()):
        prefix = "three_way" if initialization == "i1_minus_i0" else "guidance_interaction"
        for metric in EFFECT_METRICS:
            values = np.asarray([float(row[f"{prefix}_{metric}"]) for row in selected])
            output.append({
                "encoder": encoder, "probe": probe,
                "generation_seed": generation_seed, "initialization": initialization,
                "metric": metric, "mean": float(np.mean(values)),
                "std_over_class_clusters": float(np.std(values)),
                "positive_fraction": float(np.mean(values > 0)),
                "class_cluster_observations": len(values),
            })
    return output


def normalize_vector(vector):
    vector = np.asarray(vector, dtype=np.float64)
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


def geometry_rows(samples, paired, features, probes):
    feature_map = {
        sample["path"]: normalize_vector(features[index]) for index, sample in enumerate(samples)
    }
    mode_geometry = {}
    for key, prompts in paired.items():
        spec, class_key, k, generation_seed, regime = key
        label = feature_map[prompts["label"]["path"]]
        correct = feature_map[prompts["correct"]["path"]]
        shuffled = feature_map[prompts["shuffled"]["path"]]
        d_text = correct - label
        d_swap = shuffled - correct
        payload = probes["encoders"]["dino"]["classes"][class_key]
        cluster_ids = np.asarray(payload["centroid_cluster_ids"], dtype=np.int64)
        centroids = np.asarray(payload["centroids"], dtype=np.float64)
        position = np.flatnonzero(cluster_ids == k)
        if len(position) != 1:
            raise ValueError(f"Frozen DINO centroids lack {class_key} cluster {k}")
        target_centroid = normalize_vector(centroids[int(position[0])])
        alignment, specificity = target_specificity(d_text, label, cluster_ids, centroids, k)
        mode_geometry[key] = {
            "spec": spec, "class_key": class_key,
            "class_id": prompts["label"]["class_id"],
            "class_name": prompts["label"]["class_name"],
            "visual_cluster_id": k, "generation_seed": generation_seed,
            "visual_mode": regime, "image_seed": prompts["label"]["image_seed"],
            "text_norm": float(np.linalg.norm(d_text)),
            "swap_norm": float(np.linalg.norm(d_swap)),
            "target_alignment": alignment, "target_specificity": specificity,
            "target_residual": float(np.linalg.norm(target_centroid - label)),
            "label_feature": label,
        }

    output = []
    for spec, class_key, k, generation_seed in sorted({key[:4] for key in mode_geometry}):
        regimes = {
            regime: mode_geometry[(spec, class_key, k, generation_seed, regime)]
            for regime in REGIMES
        }
        for initialization, g0, g1 in (("i0", "i0g0", "i0g1"), ("i1", "i1g0", "i1g1")):
            before, after = regimes[g0], regimes[g1]
            output.append({
                **{key: before[key] for key in (
                    "spec", "class_key", "class_id", "class_name", "visual_cluster_id",
                    "generation_seed", "image_seed",
                )},
                "initialization": initialization,
                "text_norm_g0": before["text_norm"], "text_norm_g1": after["text_norm"],
                "text_norm_log_ratio": float(math.log(
                    (after["text_norm"] + 1e-8) / (before["text_norm"] + 1e-8)
                )),
                "swap_norm_g0": before["swap_norm"], "swap_norm_g1": after["swap_norm"],
                "swap_norm_log_ratio": float(math.log(
                    (after["swap_norm"] + 1e-8) / (before["swap_norm"] + 1e-8)
                )),
                "target_alignment_g0": before["target_alignment"],
                "target_alignment_g1": after["target_alignment"],
                "target_alignment_delta": after["target_alignment"] - before["target_alignment"],
                "target_specificity_g0": before["target_specificity"],
                "target_specificity_g1": after["target_specificity"],
                "target_specificity_delta": after["target_specificity"] - before["target_specificity"],
                "target_residual_g0": before["target_residual"],
                "target_residual_g1": after["target_residual"],
                "target_residual_delta": after["target_residual"] - before["target_residual"],
                "label_guidance_displacement_norm": float(np.linalg.norm(
                    after["label_feature"] - before["label_feature"]
                )),
            })
    return output


def summarize_geometry(rows, bootstrap_samples, seed):
    output = []
    counter = 0
    for scope in ["combined"] + sorted({row["spec"] for row in rows}):
        scoped = rows if scope == "combined" else [row for row in rows if row["spec"] == scope]
        for initialization in ("i0", "i1"):
            selected = [row for row in scoped if row["initialization"] == initialization]
            for metric in GEOMETRY_METRICS:
                stats = bootstrap_grouped(selected, metric, bootstrap_samples, seed + counter)
                counter += 1
                output.append({
                    "scope": scope, "initialization": initialization,
                    "metric": metric, **stats,
                })
    return output


def summarize_geometry_by_generation_seed(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["generation_seed"], row["initialization"])].append(row)
    output = []
    for (generation_seed, initialization), selected in sorted(grouped.items()):
        for metric in GEOMETRY_METRICS:
            values = np.asarray([float(row[metric]) for row in selected])
            output.append({
                "generation_seed": generation_seed, "initialization": initialization,
                "metric": metric, "mean": float(np.mean(values)),
                "std_over_class_clusters": float(np.std(values)),
                "positive_fraction": float(np.mean(values > 0)),
                "class_cluster_observations": len(values),
            })
    return output


def load_guidance_rows(datasets):
    output = []
    step_output = []
    for (spec, generation_seed, regime, prompt), dataset_dir in sorted(datasets.items()):
        if not regime.endswith("g1"):
            continue
        raw_path = dataset_dir / "guidance_metrics" / "guidance_metrics_raw.csv"
        records = read_csv(raw_path)
        for row in records:
            step_output.append({
                "spec": spec, "generation_seed": generation_seed,
                "visual_mode": regime, "initialization": regime[:2],
                "prompt_condition": prompt, "class_id": row["class_id"],
                "visual_cluster_id": int(row["sample_index"]),
                "image_seed": int(row["image_seed"]),
                "step_index": int(row["step_index"]), "timestep": int(row["timestep"]),
                "cosine": float(row["cosine_similarity"]),
                "q": float(row["q_text_over_image"]),
                "kappa": float(row["conflict_projection_ratio"]),
            })
        grouped = defaultdict(list)
        for row in records:
            grouped[(row["class_id"], int(row["sample_index"]), int(row["image_seed"]))].append(row)
        for (class_id, cluster_id, image_seed), steps in grouped.items():
            ordered = sorted(steps, key=lambda row: int(row["step_index"]))
            thirds = np.array_split(np.arange(len(ordered)), 3)
            record = {
                "spec": spec, "generation_seed": generation_seed, "visual_mode": regime,
                "initialization": regime[:2], "prompt_condition": prompt,
                "class_id": class_id, "visual_cluster_id": cluster_id, "image_seed": image_seed,
                "active_guidance_steps": len(ordered),
            }
            for name, source in (
                ("cosine", "cosine_similarity"),
                ("q", "q_text_over_image"),
                ("kappa", "conflict_projection_ratio"),
            ):
                values = np.asarray([float(row[source]) for row in ordered])
                record[f"{name}_mean"] = float(np.mean(values))
                record[f"{name}_median"] = float(np.median(values))
                if name == "cosine":
                    record["cosine_negative_fraction"] = float(np.mean(values < 0))
                for section_name, indices in zip(("early", "middle", "late"), thirds):
                    record[f"{section_name}_{name}_mean"] = float(np.mean(values[indices]))
            output.append(record)
    return output, step_output


def summarize_guidance_steps(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["visual_mode"], row["prompt_condition"], row["step_index"])].append(row)
    output = []
    for (regime, prompt, step_index), members in sorted(grouped.items()):
        output.append({
            "visual_mode": regime, "prompt_condition": prompt,
            "step_index": step_index, "timestep": members[0]["timestep"],
            "observations": len(members),
            **{
                f"{metric}_{stat}": float(function([row[metric] for row in members]))
                for metric in ("cosine", "q", "kappa")
                for stat, function in (("mean", np.mean), ("median", np.median))
            },
            "cosine_negative_fraction": float(np.mean([row["cosine"] < 0 for row in members])),
        })
    return output


def rank_values(values):
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def correlation_value(left, right, method):
    if method == "spearman":
        left, right = rank_values(left), rank_values(right)
    return float(np.corrcoef(left, right)[0, 1])


def conflict_correlations(conflicts, geometry, bootstrap_samples, permutation_samples, seed):
    geometry_index = {
        (row["spec"], row["generation_seed"], row["initialization"],
         row["class_id"], row["visual_cluster_id"]): row
        for row in geometry
    }
    merged = []
    for row in conflicts:
        key = (
            row["spec"], row["generation_seed"], row["initialization"],
            row["class_id"], row["visual_cluster_id"],
        )
        geometry_row = geometry_index[key]
        outcome = (
            geometry_row["text_norm_log_ratio"]
            if row["prompt_condition"] == "correct"
            else geometry_row["swap_norm_log_ratio"]
            if row["prompt_condition"] == "shuffled" else None
        )
        if outcome is not None:
            merged.append({
                **row,
                "class_key": geometry_row["class_key"],
                "output_log_norm_ratio": outcome,
            })

    output = []
    rng = np.random.default_rng(seed)
    for initialization in ("i0", "i1"):
        for prompt in ("correct", "shuffled"):
            selected = [
                row for row in merged if row["initialization"] == initialization
                and row["prompt_condition"] == prompt
            ]
            grouped = defaultdict(list)
            for row in selected:
                grouped[(row["class_key"], row["visual_cluster_id"])].append(row)
            selected = [
                {
                    "class_key": key[0], "visual_cluster_id": key[1],
                    **{
                        column: float(np.mean([row[column] for row in members]))
                        for column in (
                            "kappa_mean", "cosine_mean", "q_mean", "output_log_norm_ratio"
                        )
                    },
                }
                for key, members in grouped.items()
            ]
            for signal in ("kappa_mean", "cosine_mean", "q_mean"):
                matrix = np.asarray([
                    [row[signal], row["output_log_norm_ratio"]] for row in selected
                ], dtype=np.float64)
                class_indices = defaultdict(list)
                for index, row in enumerate(selected):
                    class_indices[row["class_key"]].append(index)
                for method in ("pearson", "spearman"):
                    observed = correlation_value(matrix[:, 0], matrix[:, 1], method)
                    boot = []
                    for _ in range(bootstrap_samples):
                        indices = rng.integers(0, len(matrix), len(matrix))
                        boot.append(correlation_value(matrix[indices, 0], matrix[indices, 1], method))
                    null = []
                    for _ in range(permutation_samples):
                        permuted = matrix[:, 1].copy()
                        for indices in class_indices.values():
                            permuted[indices] = rng.permutation(permuted[indices])
                        null.append(correlation_value(matrix[:, 0], permuted, method))
                    output.append({
                        "initialization": initialization, "prompt_condition": prompt,
                        "guidance_signal": signal, "outcome": "output_log_norm_ratio",
                        "correlation": method, "value": observed,
                        "bootstrap_ci_lower": float(np.nanquantile(boot, 0.025)),
                        "bootstrap_ci_upper": float(np.nanquantile(boot, 0.975)),
                        "permutation_p_two_sided": float(
                            (1 + np.sum(np.abs(null) >= abs(observed))) / (len(null) + 1)
                        ),
                        "class_cluster_groups": len(matrix),
                    })
    return output


def plot_results(output_dir, interaction_summary, geometry_summary, recovery_summary):
    figure, axes = plt.subplots(2, 2, figsize=(14, 10))
    dino = [
        row for row in interaction_summary if row["scope"] == "combined"
        and row["encoder"] == "dino" and row["initialization"] in {"i0", "i1"}
    ]
    for axis, metric in zip(axes[0], ("delta_target", "delta_pull")):
        selected = [row for row in dino if row["metric"] == metric]
        labels = [f"{row['initialization']}\n{row['probe']}" for row in selected]
        means = [row["mean"] for row in selected]
        errors = np.asarray([
            [row["mean"] - row["bootstrap_ci_lower"] for row in selected],
            [row["bootstrap_ci_upper"] - row["mean"] for row in selected],
        ])
        axis.bar(np.arange(len(selected)), means, yerr=errors, capsize=4)
        axis.set_xticks(np.arange(len(selected)), labels)
        axis.set_title(f"Guidance interaction: {metric}")
        axis.axhline(0, color="black", linestyle="--", linewidth=1)

    geometry = [row for row in geometry_summary if row["scope"] == "combined"]
    for initialization in ("i0", "i1"):
        selected = {row["metric"]: row for row in geometry if row["initialization"] == initialization}
        axes[1, 0].bar(
            [f"{initialization} text", f"{initialization} swap"],
            [selected["text_norm_log_ratio"]["mean"], selected["swap_norm_log_ratio"]["mean"]],
        )
    axes[1, 0].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[1, 0].set_title("G1/G0 DINO displacement log-ratio")

    recovery = [
        row for row in recovery_summary if row["encoder"] == "dino"
        and row["probe"] == "nearest_centroid" and row["prompt_condition"] == "label"
        and row["target_kind"] == "visual_source"
    ]
    axes[1, 1].bar(
        [row["visual_mode"] for row in recovery], [row["top1"] for row in recovery]
    )
    axes[1, 1].set_title("Label-only visual cluster Top-1")
    for axis in axes.flat:
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("P5: how continuous image guidance changes text execution")
    figure.tight_layout()
    figure.savefig(Path(output_dir) / "p5_continuous_guidance.png", dpi=180)
    plt.close(figure)


def plot_guidance_steps(output_dir, rows):
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=False)
    for column, regime in enumerate(("i0g1", "i1g1")):
        selected_regime = [row for row in rows if row["visual_mode"] == regime]
        for prompt in PROMPTS:
            selected = sorted(
                (row for row in selected_regime if row["prompt_condition"] == prompt),
                key=lambda row: row["step_index"],
            )
            axes[0, column].plot(
                [row["step_index"] for row in selected],
                [row["cosine_mean"] for row in selected], marker="o", label=prompt,
            )
            axes[1, column].plot(
                [row["step_index"] for row in selected],
                [row["kappa_mean"] for row in selected], marker="o", label=prompt,
            )
        axes[0, column].axhline(0, color="black", linestyle="--", linewidth=1)
        axes[0, column].set_title(f"{regime}: text-image cosine")
        axes[1, column].set_title(f"{regime}: cancellation kappa")
        axes[1, column].set_xlabel("Active guidance step index")
        axes[0, column].legend()
        axes[1, column].legend()
    for axis in axes.flat:
        axis.grid(alpha=0.25)
    figure.suptitle("P5 step-level text/image guidance interaction")
    figure.tight_layout()
    figure.savefig(Path(output_dir) / "p5_guidance_conflict_steps.png", dpi=180)
    plt.close(figure)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples, paired, datasets = inventory(args.prepared_dir, args.generation_manifest)
    with (Path(args.prepared_dir) / "frozen_real_image_probes.pkl").open("rb") as handle:
        probes = pickle.load(handle)
    if probes.get("training_data") != "real_images_only":
        raise RuntimeError("P5 probes are not frozen real-image-only probes")

    all_effects, all_recovery = [], []
    dino_features = None
    for encoder, model_root in (("dino", args.dino_model), ("clip", args.clip_model)):
        expected = Path(probes["encoders"][encoder]["model_root"]).resolve()
        if Path(model_root).resolve() != expected:
            raise RuntimeError(f"P5 {encoder} model differs from frozen probe model")
        features = load_or_extract(args, samples, encoder, model_root)
        effects, recovery = paired_metrics(samples, paired, features, probes, encoder)
        all_effects.extend(effects)
        all_recovery.extend(recovery)
        if encoder == "dino":
            dino_features = features

    regime_summary = summarize_regime_effects(
        all_effects, args.bootstrap_samples, args.random_seed
    )
    interaction_raw = guidance_interaction_rows(all_effects)
    interaction_summary = summarize_guidance_interactions(
        interaction_raw, args.bootstrap_samples, args.random_seed + 10000
    )
    interaction_by_seed = summarize_interactions_by_generation_seed(interaction_raw)
    recovery_per_class, recovery_summary = summarize_recovery(all_recovery)
    geometry = geometry_rows(samples, paired, dino_features, probes)
    geometry_summary = summarize_geometry(
        geometry, args.bootstrap_samples, args.random_seed + 20000
    )
    geometry_by_seed = summarize_geometry_by_generation_seed(geometry)
    conflict, conflict_steps_raw = load_guidance_rows(datasets)
    conflict_steps = summarize_guidance_steps(conflict_steps_raw)
    correlations = conflict_correlations(
        conflict, geometry, args.bootstrap_samples, args.permutation_samples,
        args.random_seed + 30000,
    )

    write_csv(output_dir / "regime_effects_raw.csv", all_effects)
    write_csv(output_dir / "regime_effects_summary.csv", regime_summary)
    write_csv(output_dir / "guidance_interactions_raw.csv", interaction_raw)
    write_csv(output_dir / "guidance_interactions_summary.csv", interaction_summary)
    write_csv(output_dir / "guidance_interactions_by_generation_seed.csv", interaction_by_seed)
    write_csv(output_dir / "cluster_recoverability_raw.csv", all_recovery)
    write_csv(output_dir / "cluster_recoverability_per_class_seed.csv", recovery_per_class)
    write_csv(output_dir / "cluster_recoverability_summary.csv", recovery_summary)
    write_csv(output_dir / "feature_displacements_raw.csv", geometry)
    write_csv(output_dir / "feature_displacements_summary.csv", geometry_summary)
    write_csv(output_dir / "feature_displacements_by_generation_seed.csv", geometry_by_seed)
    write_csv(output_dir / "guidance_conflict_per_sample.csv", conflict)
    write_csv(output_dir / "guidance_conflict_per_step_raw.csv", conflict_steps_raw)
    write_csv(output_dir / "guidance_conflict_per_step_summary.csv", conflict_steps)
    write_csv(output_dir / "guidance_conflict_output_correlations.csv", correlations)
    plot_results(output_dir, interaction_summary, geometry_summary, recovery_summary)
    plot_guidance_steps(output_dir, conflict_steps)

    primary = {}
    for initialization in ("i0", "i1", "i1_minus_i0"):
        selected = [
            row for row in interaction_summary if row["scope"] == "combined"
            and row["encoder"] == "dino" and row["initialization"] == initialization
            and row["metric"] in {"label_target_margin", "delta_target", "delta_pull"}
        ]
        primary[initialization] = {
            f"{row['probe']}_{row['metric']}": row for row in selected
        }
    primary["feature_displacements"] = {
        f"{row['initialization']}_{row['metric']}": row
        for row in geometry_summary if row["scope"] == "combined"
        and row["metric"] in {
            "text_norm_log_ratio", "swap_norm_log_ratio", "target_alignment_delta",
            "target_specificity_delta", "target_residual_delta",
        }
    }
    atomic_json(output_dir / "summary.json", {
        "format_version": 1,
        "primary": primary,
        "factorial_design": "initialization I0/I1 x continuous guidance G0/G1 x Label/Correct/Shuffled",
        "probe_training_data": "real_images_only; no generated-image refitting",
        "pairing": "same class, visual cluster, generation seed, and image seed across all 12 cells",
        "bootstrap_unit": "class_key x visual_cluster_id after averaging generation seeds",
        "guidance_definition": (
            "G1 is unprojected original CoDA continuous guidance. Recorded g_img already "
            "contains gamma and the current scheduler sigma. Metrics are measured in "
            "noise-prediction space before the final CFG multiplication; CFG subsequently "
            "multiplies both text and image conditional directions by the same coefficient, "
            "so cosine, q, and kappa are unchanged by that common scaling."
        ),
        "interpretation_boundary": (
            "Within-initialization G1-G0 contrasts isolate continuous guidance. The I1-vs-I0 "
            "three-way contrast includes their different effective denoising schedules. P5 does "
            "not test downstream training value."
        ),
    })
    print(f"P5 evaluation complete: {output_dir}")


if __name__ == "__main__":
    main()
