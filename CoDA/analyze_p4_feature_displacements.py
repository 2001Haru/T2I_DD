"""P4b: distinguish conditioning redundancy from path interference.

The diagnostic reuses generated DINO features and real-image centroids. It
does not load SDXL, extract new features, or fit anything on generated data.
"""

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


MODES = ("i0g0", "i1g0")
CONDITIONS = ("label", "correct", "shuffled")
EPSILON = 1e-8
SUMMARY_METRICS = (
    "text_norm_i0g0", "text_norm_i1g0", "text_norm_ratio", "text_norm_log_ratio",
    "swap_norm_i0g0", "swap_norm_i1g0", "swap_norm_ratio", "swap_norm_log_ratio",
    "proto_text_overlap",
    "target_alignment_i0g0", "target_alignment_i1g0", "target_alignment_delta",
    "target_specificity_i0g0", "target_specificity_i1g0", "target_specificity_delta",
    "target_residual_i0g0", "target_residual_i1g0", "target_residual_delta",
)


def parse_args():
    parser = argparse.ArgumentParser(description="P4b feature-displacement diagnostic")
    parser.add_argument("--prepared-dir", required=True)
    parser.add_argument("--generation-manifest", required=True)
    parser.add_argument("--feature-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--permutation-samples", type=int, default=1000)
    parser.add_argument("--random-seed", type=int, default=20260803)
    return parser.parse_args()


def read_csv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def atomic_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def normalize(vector):
    vector = np.asarray(vector, dtype=np.float64)
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


def cosine(left, right):
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-12:
        return float("nan")
    return float(np.dot(left, right) / denominator)


def load_prompt_records(dataset_dir):
    records = []
    for path in sorted(Path(dataset_dir).glob("prompt_records_gpu*.json")):
        records.extend(json.loads(path.read_text(encoding="utf-8")))
    if not records:
        raise FileNotFoundError(f"No prompt records in {dataset_dir}")
    return {(row["class_id"], int(row["visual_cluster_id"])): row for row in records}


def load_feature_map(cache_path):
    with np.load(cache_path, allow_pickle=False) as cached:
        paths = cached["paths"].astype(str)
        features = cached["features"].astype(np.float64, copy=False)
    if len(paths) != len(features):
        raise ValueError("DINO feature cache path/feature lengths differ")
    return {
        str(Path(path).resolve()): normalize(feature)
        for path, feature in zip(paths, features)
    }


def inventory(prepared_dir, manifest_path, feature_map):
    pairs = read_csv(Path(prepared_dir) / "pair_manifest.csv")
    manifests = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    datasets = {}
    prompt_records = {}
    for row in manifests:
        key = (
            row["spec"], int(row["generation_seed"]),
            row["visual_mode"], row["prompt_condition"],
        )
        if key in datasets:
            raise ValueError(f"Duplicate generation condition: {key}")
        datasets[key] = Path(row["dataset_dir"])
        prompt_records[key] = load_prompt_records(row["dataset_dir"])

    expected = {(mode, condition) for mode in MODES for condition in CONDITIONS}
    by_spec_seed = defaultdict(set)
    for spec, seed, mode, condition in datasets:
        by_spec_seed[(spec, seed)].add((mode, condition))
    for key, actual in by_spec_seed.items():
        if actual != expected:
            raise ValueError(f"Incomplete P4 matrix for {key}: {sorted(actual)}")

    pair_by_spec = defaultdict(list)
    for row in pairs:
        pair_by_spec[row["spec"]].append(row)
    output = []
    for spec, seed in sorted(by_spec_seed):
        for pair in pair_by_spec[spec]:
            k = int(pair["visual_cluster_id"])
            j = int(pair["shuffled_caption_cluster_id"])
            samples = {}
            image_seeds = set()
            for mode, condition in sorted(expected):
                condition_key = (spec, seed, mode, condition)
                image_path = datasets[condition_key] / pair["class_id"] / f"{k}.png"
                resolved = str(image_path.resolve())
                if resolved not in feature_map:
                    raise KeyError(f"Generated image is absent from DINO cache: {resolved}")
                prompt = prompt_records[condition_key].get((pair["class_id"], k))
                if prompt is None:
                    raise ValueError(f"Missing prompt record for {image_path}")
                image_seeds.add(int(prompt["image_seed"]))
                expected_caption = None if condition == "label" else (k if condition == "correct" else j)
                if prompt.get("caption_cluster_id") != expected_caption:
                    raise ValueError(f"Caption source mismatch for {image_path}")
                samples[(mode, condition)] = feature_map[resolved]
            if len(image_seeds) != 1:
                raise ValueError(
                    f"P4b conditions do not share image seed for {pair['class_key']}/cluster {k}/seed {seed}"
                )
            output.append(
                {
                    **pair,
                    "visual_cluster_id": k,
                    "shuffled_caption_cluster_id": j,
                    "generation_seed": seed,
                    "image_seed": next(iter(image_seeds)),
                    "features": samples,
                }
            )
    return output


def target_specificity(displacement, label_feature, cluster_ids, centroids, target):
    alignments = np.asarray([
        cosine(displacement, centroid - label_feature) for centroid in centroids
    ])
    position = np.flatnonzero(cluster_ids == target)
    if len(position) != 1:
        raise ValueError(f"Real-image centroid bundle lacks target cluster {target}")
    target_index = int(position[0])
    alternatives = np.delete(alignments, target_index)
    return float(alignments[target_index]), float(alignments[target_index] - np.nanmax(alternatives))


def calculate_rows(records, probes):
    rows = []
    vector_records = []
    for record in records:
        features = record["features"]
        label0, correct0, shuffled0 = (
            features[("i0g0", condition)] for condition in CONDITIONS
        )
        label1, correct1, shuffled1 = (
            features[("i1g0", condition)] for condition in CONDITIONS
        )
        d_text0 = correct0 - label0
        d_text1 = correct1 - label1
        d_proto = label1 - label0
        d_swap0 = shuffled0 - correct0
        d_swap1 = shuffled1 - correct1
        text_norm0, text_norm1 = np.linalg.norm(d_text0), np.linalg.norm(d_text1)
        swap_norm0, swap_norm1 = np.linalg.norm(d_swap0), np.linalg.norm(d_swap1)

        payload = probes["encoders"]["dino"]["classes"][record["class_key"]]
        cluster_ids = np.asarray(payload["centroid_cluster_ids"], dtype=np.int64)
        centroids = np.asarray(payload["centroids"], dtype=np.float64)
        k = record["visual_cluster_id"]
        position = np.flatnonzero(cluster_ids == k)
        if len(position) != 1:
            raise ValueError(f"DINO centroids lack {record['class_key']} cluster {k}")
        target_centroid = normalize(centroids[int(position[0])])
        alignment0, specificity0 = target_specificity(
            d_text0, label0, cluster_ids, centroids, k
        )
        alignment1, specificity1 = target_specificity(
            d_text1, label1, cluster_ids, centroids, k
        )
        public = {
            key: value for key, value in record.items() if key != "features"
        }
        public.update(
            {
                "text_norm_i0g0": float(text_norm0),
                "text_norm_i1g0": float(text_norm1),
                "text_norm_ratio": float((text_norm1 + EPSILON) / (text_norm0 + EPSILON)),
                "text_norm_log_ratio": float(math.log((text_norm1 + EPSILON) / (text_norm0 + EPSILON))),
                "swap_norm_i0g0": float(swap_norm0),
                "swap_norm_i1g0": float(swap_norm1),
                "swap_norm_ratio": float((swap_norm1 + EPSILON) / (swap_norm0 + EPSILON)),
                "swap_norm_log_ratio": float(math.log((swap_norm1 + EPSILON) / (swap_norm0 + EPSILON))),
                "proto_text_overlap": cosine(d_text0, d_proto),
                "target_alignment_i0g0": alignment0,
                "target_alignment_i1g0": alignment1,
                "target_alignment_delta": alignment1 - alignment0,
                "target_specificity_i0g0": specificity0,
                "target_specificity_i1g0": specificity1,
                "target_specificity_delta": specificity1 - specificity0,
                "target_residual_i0g0": float(np.linalg.norm(target_centroid - label0)),
                "target_residual_i1g0": float(np.linalg.norm(target_centroid - label1)),
                "target_residual_delta": float(
                    np.linalg.norm(target_centroid - label1) - np.linalg.norm(target_centroid - label0)
                ),
            }
        )
        rows.append(public)
        vector_records.append({**public, "d_text0": d_text0, "d_proto": d_proto})
    return rows, vector_records


def finite(values):
    return np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)


def bootstrap(values, samples, seed):
    values = finite(values)
    if not len(values):
        return float("nan"), float("nan"), float("nan"), 0
    rng = np.random.default_rng(seed)
    draws = np.mean(rng.choice(values, size=(samples, len(values)), replace=True), axis=1)
    return (
        float(np.mean(values)), float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)), int(len(values)),
    )


def aggregate_class_cluster(rows, metric):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["class_key"], row["visual_cluster_id"])].append(float(row[metric]))
    return [float(np.nanmean(values)) for values in grouped.values()]


def summarize(rows, samples, seed):
    output = []
    counter = 0
    for scope in ["combined"] + sorted({row["spec"] for row in rows}):
        scoped = rows if scope == "combined" else [row for row in rows if row["spec"] == scope]
        for metric in SUMMARY_METRICS:
            values = aggregate_class_cluster(scoped, metric)
            mean, lower, upper, groups = bootstrap(values, samples, seed + counter)
            counter += 1
            output.append(
                {
                    "scope": scope, "metric": metric, "mean": mean,
                    "bootstrap_ci_lower": lower, "bootstrap_ci_upper": upper,
                    "class_cluster_groups": groups, "raw_observations": len(scoped),
                    "positive_group_fraction": float(np.mean(finite(values) > 0)),
                    "median": float(np.nanmedian(values)),
                }
            )
    return output


def summarize_by_seed(rows, samples, seed):
    output = []
    counter = 0
    for scope in ["combined"] + sorted({row["spec"] for row in rows}):
        scoped = rows if scope == "combined" else [row for row in rows if row["spec"] == scope]
        for generation_seed in sorted({row["generation_seed"] for row in scoped}):
            selected = [row for row in scoped if row["generation_seed"] == generation_seed]
            for metric in SUMMARY_METRICS:
                values = [float(row[metric]) for row in selected]
                mean, lower, upper, groups = bootstrap(values, samples, seed + counter)
                counter += 1
                output.append(
                    {
                        "scope": scope, "generation_seed": generation_seed,
                        "metric": metric, "mean": mean,
                        "bootstrap_ci_lower": lower, "bootstrap_ci_upper": upper,
                        "class_cluster_groups": groups,
                        "positive_group_fraction": float(np.mean(finite(values) > 0)),
                    }
                )
    return output


def overlap_permutation(vector_rows, scope, samples, seed):
    selected = vector_rows if scope == "combined" else [
        row for row in vector_rows if row["spec"] == scope
    ]
    by_class = defaultdict(list)
    for row in selected:
        by_class[row["class_key"]].append(row)
    observed = np.mean(aggregate_class_cluster(selected, "proto_text_overlap"))
    rng = np.random.default_rng(seed)
    null_values = []
    for _ in range(samples):
        group_cosines = defaultdict(list)
        for class_rows in by_class.values():
            by_seed = defaultdict(dict)
            for row in class_rows:
                by_seed[row["generation_seed"]][row["visual_cluster_id"]] = row
            cluster_ids = sorted({row["visual_cluster_id"] for row in class_rows})
            permutation = dict(zip(cluster_ids, rng.permutation(cluster_ids)))
            for generation_seed, seed_rows in by_seed.items():
                if set(seed_rows) != set(cluster_ids):
                    raise ValueError(
                        f"Incomplete cluster set across generation seeds in {class_rows[0]['class_key']}"
                    )
                for cluster_id, row in seed_rows.items():
                    donor = seed_rows[permutation[cluster_id]]
                    group_cosines[(row["class_key"], cluster_id)].append(
                        cosine(row["d_text0"], donor["d_proto"])
                    )
        null_values.append(np.mean([
            np.nanmean(values) for values in group_cosines.values()
        ]))
    null_values = np.asarray(null_values, dtype=np.float64)
    return {
        "scope": scope,
        "metric": "proto_text_overlap",
        "true_value": float(observed),
        "null_mean": float(np.mean(null_values)),
        "null_std": float(np.std(null_values)),
        "delta_over_null": float(observed - np.mean(null_values)),
        "null_partitions": samples,
        "null_percentile": float(np.mean(observed > null_values)),
        "permutation_p_one_sided": float((1 + np.sum(null_values >= observed)) / (samples + 1)),
    }


def average_group_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["class_key"], row["visual_cluster_id"])].append(row)
    output = []
    for key, members in grouped.items():
        output.append({
            "class_key": key[0], "visual_cluster_id": key[1], "spec": members[0]["spec"],
            **{metric: float(np.nanmean([row[metric] for row in members])) for metric in SUMMARY_METRICS},
        })
    return output


def rank_values(values):
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def correlation_rows(rows):
    grouped = average_group_rows(rows)
    output = []
    pairs = (
        ("proto_text_overlap", "text_norm_log_ratio"),
        ("proto_text_overlap", "target_alignment_delta"),
        ("proto_text_overlap", "target_residual_delta"),
        ("text_norm_log_ratio", "target_alignment_delta"),
        ("text_norm_log_ratio", "swap_norm_log_ratio"),
    )
    for scope in ["combined"] + sorted({row["spec"] for row in grouped}):
        scoped = grouped if scope == "combined" else [row for row in grouped if row["spec"] == scope]
        for left, right in pairs:
            values = np.asarray([[row[left], row[right]] for row in scoped], dtype=np.float64)
            values = values[np.all(np.isfinite(values), axis=1)]
            pearson = float(np.corrcoef(values[:, 0], values[:, 1])[0, 1])
            spearman = float(np.corrcoef(rank_values(values[:, 0]), rank_values(values[:, 1]))[0, 1])
            output.append(
                {"scope": scope, "x": left, "y": right, "pearson_r": pearson,
                 "spearman_rho": spearman, "class_cluster_groups": len(values)}
            )
    return output


def plot_results(output_dir, rows, summary, null_rows):
    lookup = {(row["scope"], row["metric"]): row for row in summary}
    figure, axes = plt.subplots(2, 2, figsize=(14, 10))
    metrics = ("text_norm_log_ratio", "swap_norm_log_ratio")
    selected = [lookup[("combined", metric)] for metric in metrics]
    means = [row["mean"] for row in selected]
    errors = np.asarray([
        [row["mean"] - row["bootstrap_ci_lower"] for row in selected],
        [row["bootstrap_ci_upper"] - row["mean"] for row in selected],
    ])
    axes[0, 0].bar(["Correct vs Label", "Shuffled vs Correct"], means, yerr=errors, capsize=4)
    axes[0, 0].set_title("Text displacement attenuation")
    axes[0, 0].set_ylabel("log(||d I1G0|| / ||d I0G0||)")

    null = next(row for row in null_rows if row["scope"] == "combined")
    axes[0, 1].bar(["matched", "within-class permuted"], [null["true_value"], null["null_mean"]])
    axes[0, 1].set_title("Prototype-text directional overlap")
    axes[0, 1].set_ylabel("Mean cosine")

    selected = [lookup[("combined", metric)] for metric in (
        "target_alignment_delta", "target_specificity_delta", "target_residual_delta"
    )]
    means = [row["mean"] for row in selected]
    errors = np.asarray([
        [row["mean"] - row["bootstrap_ci_lower"] for row in selected],
        [row["bootstrap_ci_upper"] - row["mean"] for row in selected],
    ])
    axes[1, 0].bar(["alignment", "specificity", "residual distance"], means, yerr=errors, capsize=4)
    axes[1, 0].set_title("Change in cluster-target geometry")
    axes[1, 0].set_ylabel("I1G0 - I0G0")

    grouped = average_group_rows(rows)
    colors = {"imageA": "tab:blue", "imageB": "tab:orange", "imageC": "tab:green"}
    for spec in sorted(colors):
        spec_rows = [row for row in grouped if row["spec"] == spec]
        axes[1, 1].scatter(
            [row["proto_text_overlap"] for row in spec_rows],
            [row["text_norm_log_ratio"] for row in spec_rows],
            alpha=0.65, label=spec, color=colors[spec],
        )
    axes[1, 1].set_title("Does overlap predict attenuation?")
    axes[1, 1].set_xlabel("Prototype-text cosine")
    axes[1, 1].set_ylabel("Correct-text log norm ratio")
    axes[1, 1].legend()
    for axis in axes.flat:
        axis.axhline(0, color="black", linestyle="--", linewidth=1)
        axis.grid(alpha=0.2)
    figure.suptitle("P4b: conditioning redundancy or generation-path interference?")
    figure.tight_layout()
    figure.savefig(Path(output_dir) / "p4b_feature_displacements.png", dpi=180)
    plt.close(figure)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_map = load_feature_map(args.feature_cache)
    records = inventory(args.prepared_dir, args.generation_manifest, feature_map)
    with (Path(args.prepared_dir) / "frozen_real_image_probes.pkl").open("rb") as handle:
        probes = pickle.load(handle)
    if probes.get("training_data") != "real_images_only":
        raise RuntimeError("P4b centroids are not marked real_images_only")
    rows, vector_rows = calculate_rows(records, probes)
    summary = summarize(rows, args.bootstrap_samples, args.random_seed)
    by_seed = summarize_by_seed(rows, args.bootstrap_samples, args.random_seed + 10000)
    null_rows = [
        overlap_permutation(vector_rows, scope, args.permutation_samples, args.random_seed + index)
        for index, scope in enumerate(["combined"] + sorted({row["spec"] for row in rows}))
    ]
    correlations = correlation_rows(rows)
    write_csv(output_dir / "feature_displacements_raw.csv", rows)
    write_csv(output_dir / "feature_displacements_summary.csv", summary)
    write_csv(output_dir / "feature_displacements_by_seed.csv", by_seed)
    write_csv(output_dir / "prototype_text_overlap_null.csv", null_rows)
    write_csv(output_dir / "feature_displacement_correlations.csv", correlations)
    plot_results(output_dir, rows, summary, null_rows)

    lookup = {(row["scope"], row["metric"]): row for row in summary}
    combined_null = next(row for row in null_rows if row["scope"] == "combined")
    primary = {
        metric: lookup[("combined", metric)] for metric in (
            "text_norm_log_ratio", "swap_norm_log_ratio", "proto_text_overlap",
            "target_alignment_delta", "target_specificity_delta", "target_residual_delta",
        )
    }
    atomic_json(
        output_dir / "summary.json",
        {
            "format_version": 1,
            "feature_space": "L2-normalized cached DINO image features",
            "primary": primary,
            "prototype_text_overlap_permutation": combined_null,
            "bootstrap_unit": "class_key x visual_cluster_id after averaging generation seeds",
            "permutation_null": (
                "within-class cluster permutation of d_proto donors, shared across generation seeds"
            ),
            "interpretation": {
                "redundancy": (
                    "Correct-text attenuation with preserved swap response, overlap above null, "
                    "and stronger attenuation at higher overlap"
                ),
                "path_suppression": (
                    "Both Correct-text and swap displacement norms attenuate without matched "
                    "prototype-text overlap"
                ),
                "off_axis_redirection": (
                    "Displacement norm is preserved while target alignment or specificity falls"
                ),
            },
            "interpretation_boundary": (
                "I1G0 changes both initialization content and the effective denoising schedule; "
                "path suppression cannot yet be attributed to prototype content alone."
            ),
        },
    )
    print(f"P4b feature-displacement analysis complete: {output_dir}")


if __name__ == "__main__":
    main()
