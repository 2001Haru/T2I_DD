"""Test whether replayed CoDA clusters are captured by language and DCS.

The diagnostic reuses P1 assignments, cached CLIP image features, and existing
per-image LLaVA captions. It does not call a VLM or generate synthetic images.
"""

import argparse
import csv
import gc
import glob
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from dcs_caption import _read_jsonl, select_dcs_captions
from diagnose_cluster_recoverability import (
    atomic_json,
    class_seed,
    evaluate_class,
    one_sided_p,
    summarize_encoder,
    write_csv,
)


def parse_args():
    parser = argparse.ArgumentParser(description="P2/P3 language-cluster diagnostic")
    parser.add_argument("--specs", nargs="+", default=["imageA", "imageB", "imageC"])
    parser.add_argument("--p1-run-dir", required=True)
    parser.add_argument("--caption-cache-root", default="./results/dcs_caption_cache")
    parser.add_argument("--caption-cache-name", default="vlcp_dcs_class_aware")
    parser.add_argument("--clip-model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--null-partitions", type=int, default=100)
    parser.add_argument("--linear-null-partitions", type=int, default=20)
    parser.add_argument("--retrieval-null-partitions", type=int, default=1000)
    parser.add_argument("--random-seed", type=int, default=20260803)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def normalize_path(path):
    return os.path.realpath(os.path.abspath(path))


def load_p1_assignments(p1_run_dir, specs):
    path = Path(p1_run_dir) / "assignments.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing P1 assignments: {path}")
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["spec"] not in specs:
                continue
            included = row["included_in_original_partition"].lower() == "true"
            if not included or int(row["cluster_id"]) < 0:
                continue
            rows.append(
                {
                    "sample_index": int(row["sample_index"]),
                    "class_key": row["class_key"],
                    "spec": row["spec"],
                    "class_id": row["class_id"],
                    "class_name": row["class_name"],
                    "cluster_id": int(row["cluster_id"]),
                    "path": normalize_path(row["path"]),
                }
            )
    if not rows:
        raise ValueError("No included P1 assignments were found")
    return rows


def load_clip_image_features(p1_run_dir):
    path = Path(p1_run_dir) / "feature_cache" / "clip.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Missing P1 CLIP image cache: {path}")
    with np.load(path, allow_pickle=False) as cached:
        paths = [normalize_path(value) for value in cached["paths"].astype(str)]
        features = cached["features"].astype(np.float32, copy=False)
    if len(paths) != len(features):
        raise ValueError("P1 CLIP cache path/feature count differs")
    return {path: features[index] for index, path in enumerate(paths)}


def load_caption_shards(args):
    captions = {}
    shard_counts = {}
    for spec in args.specs:
        directory = Path(args.caption_cache_root) / spec / args.caption_cache_name
        shards = sorted(glob.glob(str(directory / "captions.rank*.jsonl")))
        if not shards:
            raise FileNotFoundError(f"No caption shards found under {directory}")
        count = 0
        for shard in shards:
            for row in _read_jsonl(shard):
                path = normalize_path(row["image_path"])
                caption = row["caption"].strip()
                previous = captions.get(path)
                if previous is not None and previous != caption:
                    raise ValueError(f"Conflicting captions for {path}")
                captions[path] = caption
                count += 1
        shard_counts[spec] = {"shards": len(shards), "rows": count}
    return captions, shard_counts


def text_cache_signature(texts, model_root):
    digest = hashlib.sha256()
    digest.update(b"p2p3_clip_text_features_v1")
    digest.update(str(Path(model_root).resolve()).encode("utf-8"))
    for text in texts:
        digest.update(text.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


class ClipTextEncoder:
    def __init__(self, model_root, device):
        from transformers import AutoTokenizer, CLIPTextModelWithProjection

        self.device = torch.device(device)
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model_root, local_files_only=True)
        self.model = CLIPTextModelWithProjection.from_pretrained(
            model_root, local_files_only=True, torch_dtype=self.dtype
        ).to(self.device).eval()

    @torch.inference_mode()
    def encode(self, texts, batch_size):
        output = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            tokens = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            ).to(self.device)
            encoded = self.model(**tokens).text_embeds
            encoded = torch.nn.functional.normalize(encoded.float(), dim=1)
            output.append(encoded.cpu().numpy())
            print(
                f"CLIP text features: {min(start + batch_size, len(texts))}/{len(texts)}",
                flush=True,
            )
        return np.concatenate(output).astype(np.float32, copy=False)


def load_or_encode_texts(args, texts):
    unique_texts = list(dict.fromkeys(texts))
    cache_dir = Path(args.output_dir) / "feature_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "clip_text.npz"
    signature = text_cache_signature(unique_texts, args.clip_model)
    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as cached:
            if str(cached["signature"].item()) != signature:
                raise RuntimeError(
                    f"Text cache inputs changed: {cache_path}. Use a new run ID."
                )
            cached_texts = cached["texts"].astype(str).tolist()
            if cached_texts != unique_texts:
                raise RuntimeError(f"Text cache order changed: {cache_path}")
            unique_features = cached["features"].astype(np.float32, copy=False)
        print(f"Reusing CLIP text cache: {cache_path}")
    else:
        encoder = ClipTextEncoder(args.clip_model, args.device)
        unique_features = encoder.encode(unique_texts, args.batch_size)
        del encoder
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        temporary = cache_path.with_name(cache_path.name + ".tmp.npz")
        np.savez_compressed(
            temporary,
            texts=np.asarray(unique_texts),
            features=unique_features,
            signature=np.asarray(signature),
        )
        os.replace(temporary, cache_path)
    lookup = {text: unique_features[index] for index, text in enumerate(unique_texts)}
    return np.stack([lookup[text] for text in texts])


def build_caption_records(assignments, captions, image_features):
    records = []
    coverage = defaultdict(lambda: {"assigned": 0, "captioned": 0})
    cluster_coverage = {}
    for row in assignments:
        coverage[row["spec"]]["assigned"] += 1
        key = (row["class_key"], row["cluster_id"])
        if key not in cluster_coverage:
            cluster_coverage[key] = {
                "spec": row["spec"],
                "class_id": row["class_id"],
                "class_name": row["class_name"],
                "class_key": row["class_key"],
                "cluster_id": row["cluster_id"],
                "assigned_images": 0,
                "captioned_images": 0,
            }
        cluster_coverage[key]["assigned_images"] += 1
        caption = captions.get(row["path"])
        image_feature = image_features.get(row["path"])
        if caption is None or image_feature is None:
            continue
        coverage[row["spec"]]["captioned"] += 1
        cluster_coverage[key]["captioned_images"] += 1
        records.append({**row, "caption": caption, "image_feature": image_feature})
    if not records:
        raise ValueError("No overlap among P1 assignments, captions, and CLIP images")
    for values in coverage.values():
        values["caption_fraction"] = values["captioned"] / values["assigned"]
    coverage_rows = []
    for row in cluster_coverage.values():
        coverage_rows.append(
            {
                **row,
                "caption_fraction": row["captioned_images"] / row["assigned_images"],
            }
        )
    coverage_rows.sort(key=lambda row: (row["spec"], row["class_id"], row["cluster_id"]))
    return records, dict(coverage), coverage_rows


def build_visual_centroids(assignments, image_features):
    members = defaultdict(list)
    for row in assignments:
        feature = image_features.get(row["path"])
        if feature is not None:
            members[(row["class_key"], row["cluster_id"])].append(feature)
    centroids = {}
    for key, values in members.items():
        centroid = np.mean(np.stack(values), axis=0)
        centroid /= max(float(np.linalg.norm(centroid)), 1e-12)
        centroids[key] = centroid
    return centroids


def evaluate_text_recoverability(args, records, text_features):
    indices_by_class = defaultdict(list)
    for index, row in enumerate(records):
        indices_by_class[row["class_key"]].append(index)
    tasks = []
    for class_key in sorted(indices_by_class):
        indices = np.asarray(indices_by_class[class_key], dtype=np.int64)
        first = records[int(indices[0])]
        tasks.append(
            (
                class_key,
                first["spec"],
                first["class_id"],
                first["class_name"],
                text_features[indices],
                np.asarray([records[index]["cluster_id"] for index in indices]),
                args.folds,
                args.null_partitions,
                args.linear_null_partitions,
                args.random_seed,
                args.ridge_alpha,
            )
        )
    if args.jobs == 1:
        results = [evaluate_class(*task) for task in tasks]
    else:
        from joblib import Parallel, delayed, parallel_backend

        with parallel_backend("loky", inner_max_num_threads=1):
            results = Parallel(n_jobs=args.jobs, verbose=10)(
                delayed(evaluate_class)(*task) for task in tasks
            )
    return results


def retrieval_metrics(similarity, targets=None):
    count = similarity.shape[0]
    targets = np.arange(count) if targets is None else np.asarray(targets)
    row_order = np.argsort(-similarity, axis=1)
    row_ranks = np.asarray(
        [np.flatnonzero(row_order[index] == targets[index])[0] + 1 for index in range(count)]
    )
    inverse = np.empty(count, dtype=np.int64)
    inverse[targets] = np.arange(count)
    column_order = np.argsort(-similarity, axis=0)
    column_ranks = np.asarray(
        [np.flatnonzero(column_order[:, index] == inverse[index])[0] + 1 for index in range(count)]
    )
    selected = similarity[np.arange(count), targets]
    unselected = similarity.copy()
    unselected[np.arange(count), targets] = np.nan
    return {
        "text_to_image_top1": float(np.mean(row_ranks == 1)),
        "text_to_image_mrr": float(np.mean(1.0 / row_ranks)),
        "image_to_text_top1": float(np.mean(column_ranks == 1)),
        "image_to_text_mrr": float(np.mean(1.0 / column_ranks)),
        "bidirectional_top1": float(
            0.5 * (np.mean(row_ranks == 1) + np.mean(column_ranks == 1))
        ),
        "bidirectional_mrr": float(
            0.5 * (np.mean(1.0 / row_ranks) + np.mean(1.0 / column_ranks))
        ),
        "paired_similarity": float(np.mean(selected)),
        "unpaired_similarity": float(np.nanmean(unselected)),
        "diagonal_margin": float(np.mean(selected) - np.nanmean(unselected)),
    }


def build_dcs_summaries(args, records, visual_centroids):
    rows_by_class = defaultdict(list)
    for row in records:
        rows_by_class[row["class_key"]].append(row)
    summaries = []
    payload = {"metadata": {"threshold": args.threshold, "top_k": args.top_k}, "classes": {}}
    for class_key in sorted(rows_by_class):
        rows = rows_by_class[class_key]
        captions = [row["caption"] for row in rows]
        labels = np.asarray([row["cluster_id"] for row in rows], dtype=np.int64)
        selected, common, diagnostics = select_dcs_captions(
            captions,
            labels,
            rows[0]["class_name"],
            args.threshold,
            args.top_k,
        )
        class_payload = {
            "spec": rows[0]["spec"],
            "class_id": rows[0]["class_id"],
            "class_name": rows[0]["class_name"],
            "class_common_words": common,
            "clusters": {},
        }
        for cluster_id in sorted(selected):
            member_indices = np.flatnonzero(labels == cluster_id)
            selected_index = diagnostics[cluster_id]["selected_member_index"]
            entry = {
                "class_key": class_key,
                "spec": rows[0]["spec"],
                "class_id": rows[0]["class_id"],
                "class_name": rows[0]["class_name"],
                "cluster_id": int(cluster_id),
                "caption": selected[cluster_id],
                "source_path": rows[selected_index]["path"],
                "captioned_members": int(len(member_indices)),
                "visual_centroid": visual_centroids[(class_key, int(cluster_id))].copy(),
            }
            summaries.append(entry)
            class_payload["clusters"][str(cluster_id)] = {
                "caption": entry["caption"],
                "source_path": entry["source_path"],
                "captioned_members": entry["captioned_members"],
                "selection": diagnostics[cluster_id],
            }
        payload["classes"][class_key] = class_payload
    return summaries, payload


def evaluate_correspondence(args, summaries, summary_text_features):
    by_class = defaultdict(list)
    for index, row in enumerate(summaries):
        by_class[row["class_key"]].append(index)
    rng = np.random.default_rng(args.random_seed + 5000)
    class_results = []
    matrix_rows = []
    for class_key in sorted(by_class):
        indices = by_class[class_key]
        rows = [summaries[index] for index in indices]
        if len(rows) < 2:
            raise ValueError(
                f"Fewer than two captioned replay clusters for correspondence: {class_key}"
            )
        order = np.argsort([row["cluster_id"] for row in rows])
        indices = [indices[index] for index in order]
        rows = [summaries[index] for index in indices]
        text = summary_text_features[np.asarray(indices)]
        visual = np.stack([row["visual_centroid"] for row in rows])
        similarity = text @ visual.T
        observed = retrieval_metrics(similarity)
        null = []
        for _ in range(args.retrieval_null_partitions):
            null.append(retrieval_metrics(similarity, rng.permutation(len(rows))))
        first = rows[0]
        class_results.append(
            {
                "class_key": class_key,
                "spec": first["spec"],
                "class_id": first["class_id"],
                "class_name": first["class_name"],
                "clusters": len(rows),
                "observed": observed,
                "null": null,
            }
        )
        for row_index, source in enumerate(rows):
            for column_index, target in enumerate(rows):
                matrix_rows.append(
                    {
                        "class_key": class_key,
                        "spec": first["spec"],
                        "class_id": first["class_id"],
                        "source_cluster_id": source["cluster_id"],
                        "target_cluster_id": target["cluster_id"],
                        "is_correct_pair": row_index == column_index,
                        "cosine_similarity": float(similarity[row_index, column_index]),
                    }
                )
    metric_names = list(class_results[0]["observed"])
    aggregate_rows = []
    per_class_rows = []
    scopes = [("combined", class_results)]
    for spec in sorted({row["spec"] for row in class_results}):
        scopes.append((spec, [row for row in class_results if row["spec"] == spec]))
    for result in class_results:
        for metric in metric_names:
            null_values = [row[metric] for row in result["null"]]
            per_class_rows.append(
                {
                    "scope": "class",
                    "spec": result["spec"],
                    "class_id": result["class_id"],
                    "class_name": result["class_name"],
                    "class_key": result["class_key"],
                    "clusters": result["clusters"],
                    "metric": metric,
                    "true_value": result["observed"][metric],
                    "null_mean": float(np.mean(null_values)),
                    "null_std": float(np.std(null_values, ddof=1)),
                    "delta_over_null": float(result["observed"][metric] - np.mean(null_values)),
                    "permutation_p_one_sided": one_sided_p(result["observed"][metric], null_values),
                }
            )
    for scope, selected in scopes:
        for metric in metric_names:
            true_value = float(np.mean([row["observed"][metric] for row in selected]))
            null_values = [
                float(np.mean([row["null"][index][metric] for row in selected]))
                for index in range(args.retrieval_null_partitions)
            ]
            aggregate_rows.append(
                {
                    "scope": scope,
                    "classes": len(selected),
                    "metric": metric,
                    "true_value": true_value,
                    "null_mean": float(np.mean(null_values)),
                    "null_std": float(np.std(null_values, ddof=1)),
                    "delta_over_null": float(true_value - np.mean(null_values)),
                    "null_percentile": float(np.mean(np.asarray(null_values) < true_value)),
                    "permutation_p_one_sided": one_sided_p(true_value, null_values),
                    "null_partitions": args.retrieval_null_partitions,
                }
            )
    return class_results, per_class_rows, aggregate_rows, matrix_rows


def plot_results(output_dir, text_per_class, text_aggregate, retrieval_per_class, retrieval_aggregate):
    text_primary = next(
        row for row in text_aggregate
        if row["scope"] == "combined"
        and row["classifier"] == "nearest_centroid"
        and row["metric"] == "macro_f1"
    )
    retrieval_primary = next(
        row for row in retrieval_aggregate
        if row["scope"] == "combined" and row["metric"] == "bidirectional_mrr"
    )
    class_keys = sorted({row["class_key"] for row in text_per_class})
    text_lookup = {
        row["class_key"]: row["delta_over_null"]
        for row in text_per_class
        if row["classifier"] == "nearest_centroid" and row["metric"] == "macro_f1"
    }
    retrieval_lookup = {
        row["class_key"]: row["delta_over_null"]
        for row in retrieval_per_class if row["metric"] == "diagonal_margin"
    }
    figure, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes[0, 0].bar(
        ["caption clusters", "matched random"],
        [text_primary["true_value"], text_primary["null_mean"]],
        color=["#2878b5", "#f28e2b"],
    )
    axes[0, 0].set_ylabel("Macro-F1")
    axes[0, 0].set_title("Caption-only cluster-ID recoverability")
    axes[0, 1].bar(
        ["correct DCS", "permuted DCS"],
        [retrieval_primary["true_value"], retrieval_primary["null_mean"]],
        color=["#2878b5", "#f28e2b"],
    )
    axes[0, 1].set_ylabel("Bidirectional MRR")
    axes[0, 1].set_title("DCS summary-to-visual-centroid correspondence")
    positions = np.arange(len(class_keys))
    axes[1, 0].bar(positions, [text_lookup[key] for key in class_keys])
    axes[1, 0].axhline(0, color="black", linewidth=1)
    axes[1, 0].set_title("Per-class caption Macro-F1 gain")
    axes[1, 1].bar(positions, [retrieval_lookup[key] for key in class_keys])
    axes[1, 1].axhline(0, color="black", linewidth=1)
    axes[1, 1].set_title("Per-class DCS correspondence-margin gain")
    for axis in axes[1]:
        axis.set_xticks(positions, class_keys, rotation=80, ha="right", fontsize=7)
    for axis in axes.flat:
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(Path(output_dir) / "language_cluster_alignment.png", dpi=180)
    plt.close(figure)


def main():
    args = parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between zero and one")
    if args.top_k < 1:
        raise ValueError("--top-k must be positive")
    if args.null_partitions < 1 or args.retrieval_null_partitions < 1:
        raise ValueError("Null partition counts must be positive")
    if not 1 <= args.linear_null_partitions <= args.null_partitions:
        raise ValueError("--linear-null-partitions must be within --null-partitions")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    assignments = load_p1_assignments(args.p1_run_dir, args.specs)
    image_features = load_clip_image_features(args.p1_run_dir)
    captions, shard_counts = load_caption_shards(args)
    records, coverage, coverage_rows = build_caption_records(
        assignments, captions, image_features
    )
    visual_centroids = build_visual_centroids(assignments, image_features)

    summaries, dcs_payload = build_dcs_summaries(args, records, visual_centroids)
    all_texts = [row["caption"] for row in records] + [row["caption"] for row in summaries]
    all_features = load_or_encode_texts(args, all_texts)
    caption_features = all_features[: len(records)]
    summary_features = all_features[len(records) :]

    text_results = evaluate_text_recoverability(args, records, caption_features)
    text_per_class, text_aggregate, text_null = summarize_encoder(
        "clip_text", text_results, args.null_partitions
    )
    _, retrieval_per_class, retrieval_aggregate, matrix_rows = evaluate_correspondence(
        args, summaries, summary_features
    )

    write_csv(output_dir / "text_recoverability_per_class.csv", text_per_class)
    write_csv(output_dir / "text_recoverability_aggregate.csv", text_aggregate)
    write_csv(output_dir / "text_recoverability_null.csv", text_null)
    write_csv(output_dir / "caption_coverage_by_cluster.csv", coverage_rows)
    write_csv(output_dir / "dcs_correspondence_per_class.csv", retrieval_per_class)
    write_csv(output_dir / "dcs_correspondence_aggregate.csv", retrieval_aggregate)
    write_csv(output_dir / "dcs_similarity_matrices.csv", matrix_rows)
    atomic_json(output_dir / "replayed_dcs_summaries.json", dcs_payload)
    plot_results(
        output_dir,
        text_per_class,
        text_aggregate,
        retrieval_per_class,
        retrieval_aggregate,
    )

    text_primary = next(
        row for row in text_aggregate
        if row["scope"] == "combined"
        and row["classifier"] == "nearest_centroid"
        and row["metric"] == "macro_f1"
    )
    retrieval_primary = next(
        row for row in retrieval_aggregate
        if row["scope"] == "combined" and row["metric"] == "bidirectional_mrr"
    )
    text_pass = text_primary["delta_over_null"] > 0 and text_primary["permutation_p_one_sided"] <= 0.05
    retrieval_pass = retrieval_primary["delta_over_null"] > 0 and retrieval_primary["permutation_p_one_sided"] <= 0.05
    if text_pass and retrieval_pass:
        decision = "language_recoverable_and_dcs_correspondent"
    elif text_pass:
        decision = "language_recoverable_but_dcs_not_correspondent"
    elif retrieval_pass:
        decision = "dcs_correspondent_without_per_image_recoverability"
    else:
        decision = "language_alignment_not_detected"
    summary = {
        "format_version": 1,
        "decision": decision,
        "text_recoverability_primary": text_primary,
        "dcs_correspondence_primary": retrieval_primary,
        "coverage": coverage,
        "caption_shards": shard_counts,
        "records": len(records),
        "dcs_summaries": len(summaries),
        "interpretation_boundary": (
            "This tests whether existing LLaVA captions and replayed DCS summaries "
            "encode replay-cluster identity on the path intersection reported in "
            "caption_coverage_by_cluster.csv. It does not test whether the diffusion "
            "model executes the text or whether downstream training improves."
        ),
    }
    atomic_json(output_dir / "summary.json", summary)
    atomic_json(output_dir / "complete.json", {"status": "complete"})
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
