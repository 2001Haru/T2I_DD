import argparse
import csv
import gc
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from diagnose_cross_modal_recombination import (
    PAIR_METRICS,
    dataset_feature_index,
    feature_index,
    normalized_dot,
    pearson,
    projection_metrics,
    rankdata,
    record_image_path,
    spearman,
)
from diagnose_dino_coverage import (
    DinoExtractor,
    discover_datasets,
    feature_cache_path,
    load_or_extract_features,
)
from diagnostic_common import atomic_write_json, load_json, parse_shift_runs


EPSILON = 1e-12
EXTRA_PAIR_METRICS = (
    "unit_caption_pull_projection",
    "caption_source_similarity_gain",
    "visual_target_similarity_change",
    "caption_displacement_norm",
    "off_axis_fraction",
    "visual_cluster_size",
    "caption_source_cluster_size",
    "log_visual_cluster_size",
    "visual_cluster_size_percentile",
)
ALL_PAIR_METRICS = (*PAIR_METRICS, *EXTRA_PAIR_METRICS)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Re-evaluate shuffled-caption geometry with DINO anchors built from "
            "the nearest real members of each VAE cluster"
        )
    )
    parser.add_argument("--base-run-root", required=True)
    parser.add_argument(
        "--shuffle-run",
        action="append",
        default=[],
        metavar="SHIFT=RUN_ROOT",
    )
    parser.add_argument("--cluster-assignments", required=True)
    parser.add_argument("--dino-model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--feature-cache-dir")
    parser.add_argument("--downstream-csv")
    parser.add_argument(
        "--anchor-k",
        action="append",
        type=int,
        default=[],
        help="Nearest-member count; repeat for sensitivity analysis",
    )
    parser.add_argument("--heldout-start", type=int, default=10)
    parser.add_argument("--heldout-end", type=int, default=30)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--resume", action="store_true")
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


def normalize(vector):
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm <= EPSILON:
        raise ValueError("Cannot normalize a zero feature vector")
    return vector / norm


def mean_anchor(features):
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2 or len(features) == 0:
        raise ValueError("Anchor features must be a non-empty matrix")
    return normalize(features.mean(axis=0))


def grouped_assignments(rows):
    if not rows:
        raise ValueError("Cluster assignment CSV is empty")
    fields = set(rows[0])
    if "assigned_cluster" in fields:
        cluster_field = "assigned_cluster"
    elif "cluster_index" in fields:
        cluster_field = "cluster_index"
    else:
        raise KeyError(
            "Cluster assignment CSV must contain 'assigned_cluster' "
            f"(current audit schema) or 'cluster_index'; found {sorted(fields)}"
        )

    grouped = defaultdict(list)
    for row in rows:
        normalized = {
            **row,
            "cluster_index": int(float(row[cluster_field])),
            "center_rmse": float(row["center_rmse"]),
            "image_path": str(Path(row["image_path"]).resolve()),
        }
        grouped[(row["synset"], normalized["cluster_index"])].append(normalized)
    for selected in grouped.values():
        selected.sort(key=lambda row: row["center_rmse"])
    return grouped


def cluster_size_percentiles(grouped):
    by_class = defaultdict(list)
    for (synset, cluster_index), rows in grouped.items():
        by_class[synset].append((cluster_index, len(rows)))
    output = {}
    for synset, values in by_class.items():
        sizes = [size for _, size in values]
        ranks = rankdata(sizes)
        denominator = max(len(values) - 1, 1)
        for (cluster_index, _), rank in zip(values, ranks):
            output[(synset, cluster_index)] = (rank - 1.0) / denominator
    return output


def class_occupancy(grouped):
    by_class = defaultdict(list)
    for (synset, _), rows in grouped.items():
        by_class[synset].append(len(rows))
    output = {}
    for synset, sizes in by_class.items():
        mean_size = statistics.fmean(sizes)
        output[synset] = {
            "minimum_cluster_size": min(sizes),
            "maximum_cluster_size": max(sizes),
            "mean_cluster_size": mean_size,
            "cluster_size_cv": (
                statistics.pstdev(sizes) / max(mean_size, EPSILON)
            ),
            "minimum_cluster_fraction": min(sizes) / sum(sizes),
            "maximum_to_minimum_cluster_size": max(sizes) / min(sizes),
        }
    return output


def required_real_members(grouped, max_anchor_k, heldout_start, heldout_end):
    required = []
    seen = set()
    for key in sorted(grouped):
        rows = grouped[key]
        if len(rows) < heldout_end:
            raise RuntimeError(
                f"Cluster {key} has {len(rows)} members, fewer than "
                f"heldout_end={heldout_end}"
            )
        selected = [
            *rows[:max_anchor_k],
            *rows[heldout_start - 1 : heldout_end],
        ]
        for row in selected:
            path = str(Path(row["image_path"]).resolve())
            if path not in seen:
                required.append(row)
                seen.add(path)
    return required


def build_anchors(grouped, feature_by_path, anchor_ks):
    anchors = {}
    for key, rows in grouped.items():
        for anchor_k in anchor_ks:
            features = [
                feature_by_path[str(Path(row["image_path"]).resolve())]
                for row in rows[:anchor_k]
            ]
            anchors[(anchor_k, *key)] = mean_anchor(features)
    return anchors


def evaluate_heldout_member(feature, own_index, class_anchors):
    indices = sorted(class_anchors)
    similarities = np.asarray(
        [normalized_dot(feature, class_anchors[index]) for index in indices],
        dtype=np.float64,
    )
    own_position = indices.index(own_index)
    own_similarity = float(similarities[own_position])
    other_similarities = np.delete(similarities, own_position)
    best_other_similarity = float(other_similarities.max())
    order = np.argsort(-similarities)
    retrieval_rank = int(np.where(order == own_position)[0][0]) + 1
    return {
        "own_anchor_similarity": own_similarity,
        "best_other_anchor_similarity": best_other_similarity,
        "own_anchor_margin": own_similarity - best_other_similarity,
        "retrieval_rank": retrieval_rank,
        "retrieval_correct": float(retrieval_rank == 1),
        "positive_own_margin": float(own_similarity > best_other_similarity),
    }


def validate_anchors(
    grouped,
    anchors,
    feature_by_path,
    anchor_ks,
    heldout_start,
    heldout_end,
):
    classes = sorted({synset for synset, _ in grouped})
    rows = []
    for anchor_k in anchor_ks:
        for synset in classes:
            class_anchors = {
                cluster_index: anchors[(anchor_k, synset, cluster_index)]
                for candidate_synset, cluster_index in grouped
                if candidate_synset == synset
            }
            for cluster_index in sorted(class_anchors):
                members = grouped[(synset, cluster_index)]
                heldout = members[heldout_start - 1 : heldout_end]
                for member_rank, member in enumerate(
                    heldout, start=heldout_start
                ):
                    path = str(Path(member["image_path"]).resolve())
                    metrics = evaluate_heldout_member(
                        feature_by_path[path],
                        cluster_index,
                        class_anchors,
                    )
                    rows.append(
                        {
                            "anchor_k": anchor_k,
                            "synset": synset,
                            "cluster_index": cluster_index,
                            "cluster_size": len(members),
                            "member_rank": member_rank,
                            "center_rmse": member["center_rmse"],
                            "image_path": path,
                            **metrics,
                        }
                    )
    return rows


def aggregate_validation(rows, group_fields):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in group_fields)].append(row)
    output = []
    for key, selected in grouped.items():
        output.append(
            {
                **dict(zip(group_fields, key)),
                "observations": len(selected),
                "retrieval_accuracy": statistics.fmean(
                    row["retrieval_correct"] for row in selected
                ),
                "mean_retrieval_rank": statistics.fmean(
                    row["retrieval_rank"] for row in selected
                ),
                "mean_own_anchor_similarity": statistics.fmean(
                    row["own_anchor_similarity"] for row in selected
                ),
                "mean_best_other_anchor_similarity": statistics.fmean(
                    row["best_other_anchor_similarity"] for row in selected
                ),
                "mean_own_anchor_margin": statistics.fmean(
                    row["own_anchor_margin"] for row in selected
                ),
                "positive_own_margin_fraction": statistics.fmean(
                    row["positive_own_margin"] for row in selected
                ),
            }
        )
    return sorted(
        output,
        key=lambda row: tuple(str(row[field]) for field in group_fields),
    )


def aggregate_pairs(rows, group_fields):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in group_fields)].append(row)
    output = []
    for key, selected in grouped.items():
        result = {**dict(zip(group_fields, key)), "observations": len(selected)}
        for field in ALL_PAIR_METRICS:
            result[field] = statistics.fmean(
                float(row[field]) for row in selected
            )
        output.append(result)
    return sorted(
        output,
        key=lambda row: tuple(str(row[field]) for field in group_fields),
    )


def intuitive_pair_metrics(
    metrics,
    visual_anchor,
    caption_anchor,
    shuffled_feature,
    correct_feature,
):
    caption_delta = np.asarray(shuffled_feature) - np.asarray(correct_feature)
    displacement_norm = float(np.linalg.norm(caption_delta))
    pair_distance = float(metrics["pair_euclidean_distance"])
    return {
        "unit_caption_pull_projection": (
            float(metrics["projection_from_correct"]) / max(pair_distance, EPSILON)
        ),
        "caption_source_similarity_gain": (
            normalized_dot(shuffled_feature, caption_anchor)
            - normalized_dot(correct_feature, caption_anchor)
        ),
        "visual_target_similarity_change": (
            normalized_dot(shuffled_feature, visual_anchor)
            - normalized_dot(correct_feature, visual_anchor)
        ),
        "caption_displacement_norm": displacement_norm,
        "off_axis_fraction": (
            float(metrics["residual_from_correct"])
            / max(displacement_norm, EPSILON)
        ),
    }


def build_pair_rows(
    datasets,
    prototypes,
    grouped,
    anchors,
    anchor_ks,
    dataset_features,
):
    correct_datasets = {
        int(dataset["generation_seed"]): dataset
        for dataset in datasets
        if dataset["visual_mode"] == "prototype"
        and dataset["prompt_condition"] == "correct_dcs"
    }
    shuffled_datasets = [
        dataset
        for dataset in datasets
        if dataset["visual_mode"] == "prototype"
        and dataset["prompt_condition"] == "shuffled_dcs"
    ]
    if not correct_datasets or not shuffled_datasets:
        raise RuntimeError("Missing prototype correct/shuffled DCS datasets")

    size_percentiles = cluster_size_percentiles(grouped)
    rows = []
    for shuffled in shuffled_datasets:
        generation_seed = int(shuffled["generation_seed"])
        correct = correct_datasets[generation_seed]
        correct_records = {
            (record["synset"], int(record["image_index"])): record
            for record in correct["manifest"]["prompt_records"]
        }
        correct_features = dataset_features[correct["dataset_key"]]
        shuffled_features = dataset_features[shuffled["dataset_key"]]
        shift = int(shuffled["shuffle_shift"])
        for record in shuffled["manifest"]["prompt_records"]:
            synset = record["synset"]
            image_index = int(record["image_index"])
            visual_index = int(record["prototype_index"])
            caption_index = int(record["prompt_source_index"])
            count = len(prototypes[synset])
            if caption_index != (visual_index + shift) % count:
                raise RuntimeError(
                    f"Unexpected permutation for {synset}/{image_index}: "
                    f"{visual_index}->{caption_index}, shift={shift}"
                )
            correct_record = correct_records[(synset, image_index)]
            if int(correct_record["prototype_index"]) != visual_index:
                raise RuntimeError("Correct/shuffled prototype index mismatch")
            if int(correct_record["prompt_source_index"]) != visual_index:
                raise RuntimeError("Correct DCS is not matched to its prototype")

            shuffled_path = record_image_path(shuffled["root"], record).resolve()
            correct_path = record_image_path(
                correct["root"], correct_record
            ).resolve()
            shuffled_feature = shuffled_features[str(shuffled_path)]
            correct_feature = correct_features[str(correct_path)]
            visual_size = len(grouped[(synset, visual_index)])
            caption_size = len(grouped[(synset, caption_index)])

            for anchor_k in anchor_ks:
                visual_anchor = anchors[(anchor_k, synset, visual_index)]
                caption_anchor = anchors[(anchor_k, synset, caption_index)]
                metrics = projection_metrics(
                    visual_anchor,
                    caption_anchor,
                    shuffled_feature,
                    correct_feature,
                )
                intuitive = intuitive_pair_metrics(
                    metrics,
                    visual_anchor,
                    caption_anchor,
                    shuffled_feature,
                    correct_feature,
                )
                rows.append(
                    {
                        "anchor_k": anchor_k,
                        "generation_seed": generation_seed,
                        "shuffle_shift": shift,
                        "synset": synset,
                        "image_index": image_index,
                        "visual_cluster_index": visual_index,
                        "caption_source_cluster_index": caption_index,
                        "caption": record["prompt"],
                        "shuffled_image": str(shuffled_path),
                        "correct_image": str(correct_path),
                        **metrics,
                        **intuitive,
                        "visual_cluster_size": visual_size,
                        "caption_source_cluster_size": caption_size,
                        "log_visual_cluster_size": math.log(visual_size),
                        "visual_cluster_size_percentile": size_percentiles[
                            (synset, visual_index)
                        ],
                    }
                )
    return rows


def join_downstream(class_rows, downstream_csv):
    downstream = {
        (
            int(float(row["generation_seed"])),
            row["visual_mode"],
            int(float(row["shuffle_shift"])),
            row["synset"],
        ): float(row["downstream_gain"])
        for row in read_csv(downstream_csv)
    }
    output = []
    for row in class_rows:
        key = (
            int(row["generation_seed"]),
            "prototype",
            int(row["shuffle_shift"]),
            row["synset"],
        )
        if key not in downstream:
            raise KeyError(f"Missing downstream result for {key}")
        output.append({**row, "downstream_gain": downstream[key]})
    return output


def summarize_class_hypotheses(downstream_rows):
    grouped = defaultdict(list)
    for row in downstream_rows:
        grouped[(int(row["anchor_k"]), row["synset"])].append(row)
    output = []
    occupancy_fields = (
        "minimum_cluster_size",
        "maximum_cluster_size",
        "mean_cluster_size",
        "cluster_size_cv",
        "minimum_cluster_fraction",
        "maximum_to_minimum_cluster_size",
    )
    geometry_fields = (
        "pair_cosine_distance",
        "unit_caption_pull_projection",
        "caption_source_similarity_gain",
        "visual_target_similarity_change",
        "off_axis_fraction",
    )
    for (anchor_k, synset), selected in sorted(grouped.items()):
        result = {
            "anchor_k": anchor_k,
            "synset": synset,
            "generation_shift_observations": len(selected),
            "mean_downstream_gain": statistics.fmean(
                row["downstream_gain"] for row in selected
            ),
        }
        for field in occupancy_fields:
            values = {float(row[field]) for row in selected}
            if len(values) != 1:
                raise RuntimeError(
                    f"Class occupancy changed within {anchor_k}/{synset}: "
                    f"{field}={values}"
                )
            result[field] = values.pop()
        for field in geometry_fields:
            result[field] = statistics.fmean(
                float(row[field]) for row in selected
            )
        output.append(result)
    return output


def correlation(left, right):
    return {
        "pearson": pearson(left, right),
        "spearman": spearman(left, right),
    }


def relationship_summary(
    pair_rows,
    downstream_rows,
    class_hypothesis_rows,
    validation_summary,
):
    output = {"by_anchor_k": {}}
    anchor_ks = sorted({int(row["anchor_k"]) for row in pair_rows})
    for anchor_k in anchor_ks:
        selected = [
            row for row in pair_rows if int(row["anchor_k"]) == anchor_k
        ]
        cluster_rows = aggregate_pairs(
            selected,
            (
                "anchor_k",
                "synset",
                "visual_cluster_index",
            ),
        )
        size = [row["log_visual_cluster_size"] for row in cluster_rows]
        entry = {
            "anchor_validation": next(
                row
                for row in validation_summary
                if int(row["anchor_k"]) == anchor_k
            ),
            "caption_pull": {
                "mean_unit_projection": statistics.fmean(
                    row["unit_caption_pull_projection"] for row in selected
                ),
                "mean_caption_source_similarity_gain": statistics.fmean(
                    row["caption_source_similarity_gain"] for row in selected
                ),
                "mean_visual_target_similarity_change": statistics.fmean(
                    row["visual_target_similarity_change"] for row in selected
                ),
                "positive_projection_fraction": statistics.fmean(
                    row["positive_caption_pull"] for row in selected
                ),
            },
            "cluster_size_response": {},
        }
        for field in (
            "unit_caption_pull_projection",
            "caption_source_similarity_gain",
            "visual_target_similarity_change",
            "off_axis_fraction",
        ):
            entry["cluster_size_response"][field] = correlation(
                size,
                [row[field] for row in cluster_rows],
            )

        selected_downstream = [
            row
            for row in class_hypothesis_rows
            if int(row["anchor_k"]) == anchor_k
        ]
        if selected_downstream:
            gains = [
                row["mean_downstream_gain"] for row in selected_downstream
            ]
            entry["downstream_response"] = {
                field: correlation(
                    [row[field] for row in selected_downstream],
                    gains,
                )
                for field in (
                    "pair_cosine_distance",
                    "unit_caption_pull_projection",
                    "caption_source_similarity_gain",
                    "visual_target_similarity_change",
                    "off_axis_fraction",
                )
            }
            entry["occupancy_downstream_response"] = {
                field: correlation(
                    [row[field] for row in selected_downstream],
                    gains,
                )
                for field in (
                    "minimum_cluster_size",
                    "minimum_cluster_fraction",
                    "cluster_size_cv",
                    "maximum_to_minimum_cluster_size",
                )
            }
            entry["downstream_response"]["independent_classes"] = len(
                selected_downstream
            )
            entry["downstream_response"][
                "generation_shift_rows_before_class_aggregation"
            ] = len(
                [
                    row
                    for row in downstream_rows
                    if int(row["anchor_k"]) == anchor_k
                ]
            )
        output["by_anchor_k"][str(anchor_k)] = entry
    return output


def plot_anchor_validation(rows, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as pyplot

    anchor_ks = [int(row["anchor_k"]) for row in rows]
    figure, axes = pyplot.subplots(1, 3, figsize=(15, 4.5))
    axes[0].plot(
        anchor_ks,
        [row["retrieval_accuracy"] for row in rows],
        marker="o",
    )
    axes[0].axhline(0.1, linestyle="--", color="black", label="chance")
    axes[0].set_ylabel("Held-out 10-way retrieval accuracy")
    axes[0].legend()

    axes[1].plot(
        anchor_ks,
        [row["mean_own_anchor_margin"] for row in rows],
        marker="o",
    )
    axes[1].axhline(0.0, linestyle="--", color="black")
    axes[1].set_ylabel("Own similarity - best other similarity")

    axes[2].plot(
        anchor_ks,
        [row["mean_own_anchor_similarity"] for row in rows],
        marker="o",
        label="own",
    )
    axes[2].plot(
        anchor_ks,
        [row["mean_best_other_anchor_similarity"] for row in rows],
        marker="o",
        label="best other",
    )
    axes[2].set_ylabel("Mean DINO cosine similarity")
    axes[2].legend()

    for axis in axes:
        axis.set_xlabel("Nearest real members per anchor (K)")
        axis.grid(alpha=0.25)
    figure.suptitle("Are nearest-member DINO cluster anchors valid?")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    pyplot.close(figure)


def plot_pair_geometry(pair_rows, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as pyplot

    anchor_ks = sorted({int(row["anchor_k"]) for row in pair_rows})
    shifts = sorted({int(row["shuffle_shift"]) for row in pair_rows})
    figure, axes = pyplot.subplots(
        1, len(anchor_ks), figsize=(5.5 * len(anchor_ks), 4.8), squeeze=False
    )
    for axis, anchor_k in zip(axes[0], anchor_ks):
        for shift in shifts:
            selected = [
                row
                for row in pair_rows
                if int(row["anchor_k"]) == anchor_k
                and int(row["shuffle_shift"]) == shift
            ]
            axis.scatter(
                [row["pair_cosine_distance"] for row in selected],
                [row["unit_caption_pull_projection"] for row in selected],
                alpha=0.55,
                label=f"shift {shift}",
            )
        axis.axhline(0.0, linestyle="--", color="black")
        axis.set_xlabel("Real-anchor source-target cosine distance")
        axis.set_ylabel("Unit caption-pull projection")
        axis.set_title(f"K={anchor_k}")
        axis.grid(alpha=0.25)
    axes[0][-1].legend()
    figure.suptitle("Does shuffled text pull outputs toward its real-member anchor?")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    pyplot.close(figure)


def plot_size_response(pair_rows, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as pyplot

    cluster_rows = aggregate_pairs(
        pair_rows,
        ("anchor_k", "synset", "visual_cluster_index"),
    )
    anchor_ks = sorted({int(row["anchor_k"]) for row in cluster_rows})
    figure, axes = pyplot.subplots(
        1, len(anchor_ks), figsize=(5.5 * len(anchor_ks), 4.8), squeeze=False
    )
    for axis, anchor_k in zip(axes[0], anchor_ks):
        selected = [
            row for row in cluster_rows if int(row["anchor_k"]) == anchor_k
        ]
        x_values = np.asarray(
            [row["log_visual_cluster_size"] for row in selected]
        )
        y_values = np.asarray(
            [row["unit_caption_pull_projection"] for row in selected]
        )
        axis.scatter(x_values, y_values, alpha=0.65)
        if np.std(x_values) > EPSILON:
            slope, intercept = np.polyfit(x_values, y_values, 1)
            order = np.argsort(x_values)
            axis.plot(
                x_values[order],
                intercept + slope * x_values[order],
                linestyle="--",
            )
        axis.axhline(0.0, linestyle="--", color="black")
        axis.set_xlabel("log assigned cluster size")
        axis.set_ylabel("Mean unit caption-pull projection")
        axis.set_title(f"K={anchor_k}")
        axis.grid(alpha=0.25)
    figure.suptitle("Is caption-induced movement size dependent?")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    pyplot.close(figure)


def main():
    args = parse_args()
    anchor_ks = sorted(set(args.anchor_k or [3, 5, 9]))
    if min(anchor_ks) <= 0:
        raise ValueError("anchor-k values must be positive")
    if args.heldout_start <= max(anchor_ks):
        raise ValueError("heldout-start must be greater than every anchor-k")
    if args.heldout_end < args.heldout_start:
        raise ValueError("heldout-end must be >= heldout-start")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = (
        Path(args.feature_cache_dir).resolve()
        if args.feature_cache_dir
        else output_dir / "feature_cache"
    )
    cache_dir.mkdir(parents=True, exist_ok=True)

    shuffle_runs = parse_shift_runs(args.shuffle_run)
    datasets = discover_datasets(
        Path(args.base_run_root).resolve(), shuffle_runs
    )
    relevant_datasets = [
        dataset
        for dataset in datasets
        if dataset["visual_mode"] == "prototype"
        and dataset["prompt_condition"] in ("correct_dcs", "shuffled_dcs")
    ]
    prototype_paths = {
        str(Path(dataset["manifest"]["prototype_path"]).resolve())
        for dataset in relevant_datasets
    }
    if len(prototype_paths) != 1:
        raise RuntimeError(f"Runs use different prototype files: {prototype_paths}")
    prototype_path = Path(prototype_paths.pop())
    prototypes = load_json(prototype_path)

    assignment_rows = read_csv(args.cluster_assignments)
    grouped = grouped_assignments(assignment_rows)
    expected_clusters = {
        (synset, cluster_index)
        for synset, values in prototypes.items()
        for cluster_index in range(len(values))
    }
    if set(grouped) != expected_clusters:
        missing = sorted(expected_clusters - set(grouped))
        extra = sorted(set(grouped) - expected_clusters)
        raise RuntimeError(
            f"Assignment/prototype cluster mismatch; missing={missing}, extra={extra}"
        )

    real_members = required_real_members(
        grouped,
        max(anchor_ks),
        args.heldout_start,
        args.heldout_end,
    )
    real_paths = [Path(row["image_path"]) for row in real_members]
    missing_paths = [str(path) for path in real_paths if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError(f"Missing real member images: {missing_paths[:3]}")

    extractor = DinoExtractor(args.dino_model, args.device)
    real_features, _, real_cached_paths = load_or_extract_features(
        extractor,
        real_paths,
        [row["synset"] for row in real_members],
        feature_cache_path(
            cache_dir,
            (
                "real_cluster_anchor_members_"
                f"k{'-'.join(str(value) for value in anchor_ks)}_"
                f"h{args.heldout_start}-{args.heldout_end}"
            ),
        ),
        args.dino_model,
        args.batch_size,
        args.resume,
    )
    real_feature_index = feature_index(real_cached_paths, real_features)
    anchors = build_anchors(grouped, real_feature_index, anchor_ks)
    validation_rows = validate_anchors(
        grouped,
        anchors,
        real_feature_index,
        anchor_ks,
        args.heldout_start,
        args.heldout_end,
    )
    validation_by_class = aggregate_validation(
        validation_rows, ("anchor_k", "synset")
    )
    validation_summary = aggregate_validation(validation_rows, ("anchor_k",))

    dataset_features = {}
    for dataset in relevant_datasets:
        dataset_features[dataset["dataset_key"]] = dataset_feature_index(
            extractor,
            dataset,
            cache_dir,
            args.dino_model,
            args.batch_size,
            args.resume,
        )

    pair_rows = build_pair_rows(
        relevant_datasets,
        prototypes,
        grouped,
        anchors,
        anchor_ks,
        dataset_features,
    )
    del extractor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    pair_class_rows = aggregate_pairs(
        pair_rows,
        ("anchor_k", "generation_seed", "shuffle_shift", "synset"),
    )
    occupancy = class_occupancy(grouped)
    pair_class_rows = [
        {**row, **occupancy[row["synset"]]} for row in pair_class_rows
    ]
    pair_shift_rows = aggregate_pairs(
        pair_rows, ("anchor_k", "shuffle_shift")
    )
    downstream_rows = []
    class_hypothesis_rows = []
    if args.downstream_csv:
        downstream_rows = join_downstream(pair_class_rows, args.downstream_csv)
        class_hypothesis_rows = summarize_class_hypotheses(downstream_rows)

    relationships = relationship_summary(
        pair_rows,
        downstream_rows,
        class_hypothesis_rows,
        validation_summary,
    )
    summary = {
        "schema_version": 1,
        "base_run_root": str(Path(args.base_run_root).resolve()),
        "shuffle_runs": {
            str(key): str(value) for key, value in sorted(shuffle_runs.items())
        },
        "prototype_path": str(prototype_path),
        "cluster_assignments": str(
            Path(args.cluster_assignments).resolve()
        ),
        "dino_model": str(Path(args.dino_model).resolve()),
        "anchor_ks": anchor_ks,
        "anchor_definition": (
            "L2-normalized mean of L2-normalized DINOv2 CLS features from the "
            "K real images nearest to each stored VAE center"
        ),
        "heldout_member_ranks": [
            args.heldout_start,
            args.heldout_end,
        ],
        "feature_definition": "L2-normalized DINOv2 CLS token",
        "geometry": {
            "unit_caption_pull_projection": (
                "dot(shuffled-correct, unit(caption-anchor - visual-anchor))"
            ),
            "caption_source_similarity_gain": (
                "cos(shuffled, caption-anchor) - cos(correct, caption-anchor)"
            ),
            "visual_target_similarity_change": (
                "cos(shuffled, visual-anchor) - cos(correct, visual-anchor)"
            ),
            "off_axis_fraction": (
                "orthogonal displacement / total shuffled-correct displacement"
            ),
        },
        "caveats": [
            (
                "Held-out members are assigned by VAE geometry, while validation "
                "is measured in DINO space; low retrieval means the proposed "
                "semantic cluster axis is not supported."
            ),
            (
                "Anchor-K sensitivity is part of the result. A mechanism should "
                "not be claimed if its sign changes across K=3,5,9."
            ),
            (
                "Cluster-size correlations are descriptive because observations "
                "within a class share generated datasets and are not independent "
                "downstream experiments."
            ),
            "DINO geometry is a diagnostic representation, not a causal objective.",
        ],
        "validation_summary": validation_summary,
        "relationships": relationships,
    }

    write_csv(output_dir / "anchor_validation_per_member.csv", validation_rows)
    write_csv(
        output_dir / "anchor_validation_per_class.csv",
        validation_by_class,
    )
    write_csv(output_dir / "anchor_validation_summary.csv", validation_summary)
    write_csv(output_dir / "real_anchor_recombination_per_image.csv", pair_rows)
    write_csv(
        output_dir / "real_anchor_recombination_per_class.csv",
        pair_class_rows,
    )
    write_csv(
        output_dir / "real_anchor_recombination_per_shift.csv",
        pair_shift_rows,
    )
    if downstream_rows:
        write_csv(
            output_dir / "real_anchor_recombination_vs_downstream.csv",
            downstream_rows,
        )
        write_csv(
            output_dir / "real_anchor_class_hypothesis_summary.csv",
            class_hypothesis_rows,
        )
    atomic_write_json(
        output_dir / "real_anchor_recombination_summary.json", summary
    )
    plot_anchor_validation(
        validation_summary,
        output_dir / "real_anchor_validation.png",
    )
    plot_pair_geometry(
        pair_rows,
        output_dir / "real_anchor_pair_geometry.png",
    )
    plot_size_response(
        pair_rows,
        output_dir / "real_anchor_size_response.png",
    )
    print(json.dumps(relationships, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
