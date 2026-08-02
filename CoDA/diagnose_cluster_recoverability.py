"""Diagnose whether CoDA SDXL-VAE partitions transfer to independent encoders.

CoDA persists the final representatives but not the final post-processing
``points_mask``. This script replays StandardScaler -> UMAP -> HDBSCAN -> CoDA
post-processing in one fixed environment and freezes the resulting masks.
Matching against historical representatives is reported as an audit because
library-version changes can prevent exact historical reconstruction. DINOv2
and CLIP are evaluated against exact-occupancy random partitions.
"""

import argparse
import csv
import gc
import hashlib
import json
import os
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import hdbscan
from PIL import Image
from scipy.optimize import linear_sum_assignment
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from umap import UMAP

from dcs_caption import _flat_features, _load_pickle, load_class_info
from postprocess import hdbscan_post


METRICS = ("top1", "macro_f1", "balanced_accuracy")
CLASSIFIERS = ("linear_probe", "nearest_centroid")
MINIMUM_CV_CLUSTER_SIZE = 2


def parse_args():
    parser = argparse.ArgumentParser(
        description="P1: cross-representation recoverability of CoDA partitions"
    )
    parser.add_argument("--specs", nargs="+", default=["imageA", "imageB", "imageC"])
    parser.add_argument("--misc-dir", default="./misc")
    parser.add_argument("--cluster-root", default="./results/clusterfile")
    parser.add_argument("--features-cache-name", default="original_features_cache.pkl")
    parser.add_argument(
        "--saved-clusters-base-name",
        default="10_n_85_s_55_saved_clusters.pkl",
    )
    parser.add_argument("--ipc", type=int, default=10)
    parser.add_argument("--n-neighbors", type=int, default=85)
    parser.add_argument("--min-cluster-size", type=int, default=55)
    parser.add_argument("--min-samples", type=int, default=3)
    parser.add_argument("--num-seed-candidates", type=int, default=3)
    parser.add_argument("--representative-match-rmse", type=float, default=1e-6)
    parser.add_argument("--strict-representative-match", action="store_true")
    parser.add_argument("--nclass", type=int, default=10)
    parser.add_argument("--phase", type=int, default=0)
    parser.add_argument("--dino-model", required=True)
    parser.add_argument("--clip-model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--null-partitions", type=int, default=100)
    parser.add_argument("--linear-null-partitions", type=int, default=20)
    parser.add_argument("--random-seed", type=int, default=20260802)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path, rows, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        if not rows:
            raise ValueError(f"Cannot infer CSV columns for empty output: {path}")
        fieldnames = list(rows[0])
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def center_path(cluster_dir, base_name, chunk_id):
    base, extension = os.path.splitext(base_name)
    return cluster_dir / f"{base}_{chunk_id}{extension}"


def partition_signature(args):
    digest = hashlib.sha256()
    digest.update(b"p1_replayed_coda_partition_v3")
    configuration = {
        "specs": args.specs,
        "features_cache_name": args.features_cache_name,
        "saved_clusters_base_name": args.saved_clusters_base_name,
        "ipc": args.ipc,
        "nclass": args.nclass,
        "phase": args.phase,
        "n_neighbors": args.n_neighbors,
        "min_cluster_size": args.min_cluster_size,
        "min_samples": args.min_samples,
        "num_seed_candidates": args.num_seed_candidates,
        "representative_match_rmse": args.representative_match_rmse,
        "strict_representative_match": args.strict_representative_match,
    }
    digest.update(json.dumps(configuration, sort_keys=True).encode("utf-8"))
    artifact_paths = []
    for spec in args.specs:
        cluster_dir = Path(args.cluster_root) / spec
        artifact_paths.extend(sorted(cluster_dir.glob(f"{args.features_cache_name}_*")))
        base, extension = os.path.splitext(args.saved_clusters_base_name)
        artifact_paths.extend(sorted(cluster_dir.glob(f"{base}_*{extension}")))
    for path in artifact_paths:
        stat = path.stat()
        digest.update(str(path.resolve()).encode("utf-8"))
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
    return digest.hexdigest()


def load_or_reconstruct_partition(args, output_dir):
    cache_path = output_dir / "partition_cache.pkl"
    signature = partition_signature(args)
    if cache_path.is_file():
        with cache_path.open("rb") as handle:
            cached = pickle.load(handle)
        if cached.get("signature") != signature:
            raise RuntimeError(
                f"Partition cache inputs changed: {cache_path}. Use a new run ID."
            )
        print(f"Reusing reconstructed CoDA partition: {cache_path}")
        return cached["samples"], cached["class_metadata"]

    samples, class_metadata = load_partition(args)
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(
            {
                "signature": signature,
                "samples": samples,
                "class_metadata": class_metadata,
            },
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    os.replace(temporary, cache_path)
    return samples, class_metadata


def reconstruct_class_partition(args, features, saved_centers, class_key):
    args.IPC = args.ipc
    args.cluster_detial = False
    args.cluster_logger = False
    scaler = StandardScaler()
    processed = UMAP(
        n_components=50,
        n_neighbors=args.n_neighbors,
        min_dist=0.0,
        random_state=42,
    ).fit_transform(scaler.fit_transform(features))
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        prediction_data=True,
    )
    labels = clusterer.fit_predict(processed)
    cluster_count = len(np.unique(labels)) - (1 if -1 in labels else 0)
    if cluster_count == 0:
        raise ValueError(f"HDBSCAN reconstructed no clusters for {class_key}")

    initial_centers = []
    initial_indices = {}
    for cluster_id in range(cluster_count):
        mask = labels == cluster_id
        global_indices = np.flatnonzero(mask)
        local_index = int(np.argmax(clusterer.probabilities_[mask]))
        global_index = int(global_indices[local_index])
        initial_indices[cluster_id] = global_index
        initial_centers.append(processed[global_index])

    # hdbscan_post mutates next_new_id, so the main argparse namespace is safe:
    # each class resets it from the newly reconstructed HDBSCAN cluster count.
    final_clusters = hdbscan_post(
        args,
        cluster_count,
        initial_centers,
        labels,
        processed,
        "",
        clusterer,
    )
    if len(final_clusters) != args.ipc:
        raise ValueError(
            f"Expected {args.ipc} reconstructed clusters for {class_key}, "
            f"found {len(final_clusters)}"
        )

    reconstructed = []
    for cluster_id, cluster_info in final_clusters.items():
        mask = np.asarray(cluster_info["points_mask"], dtype=bool)
        if not np.any(mask):
            raise ValueError(f"Empty reconstructed cluster {cluster_id} in {class_key}")
        if cluster_info.get("origin") == "hdbscan_initial":
            representative_index = initial_indices[cluster_id]
        else:
            member_indices = np.flatnonzero(mask)
            center = np.asarray(cluster_info["center"])
            local_index = int(
                np.argmin(np.sum((processed[member_indices] - center) ** 2, axis=1))
            )
            representative_index = int(member_indices[local_index])
        reconstructed.append(
            {
                "internal_cluster_id": int(cluster_id),
                "mask": mask,
                "representative_index": representative_index,
                "representative": features[representative_index],
                "origin": cluster_info.get("origin", "unknown"),
            }
        )

    reconstructed_centers = np.stack(
        [cluster["representative"] for cluster in reconstructed]
    )
    mean_squared = np.mean(
        (reconstructed_centers[:, None, :] - saved_centers[None, :, :]) ** 2,
        axis=2,
    )
    rows, columns = linear_sum_assignment(mean_squared)
    matched_rmse = np.sqrt(mean_squared[rows, columns])
    maximum_match_rmse = float(matched_rmse.max())
    representative_match_passed = (
        maximum_match_rmse <= args.representative_match_rmse
    )
    if not representative_match_passed:
        message = (
            f"Replayed representatives differ from historical centers for "
            f"{class_key}: max RMSE={maximum_match_rmse:.8g}, exact-replay "
            f"tolerance={args.representative_match_rmse:.8g}. The replayed "
            "partition remains valid for P1 but is not an exact recovery of the "
            "historical points_mask."
        )
        if args.strict_representative_match:
            raise RuntimeError(message)
        print(f"WARNING: {message}", flush=True)

    reconstructed_to_saved = {
        int(row): int(column) for row, column in zip(rows, columns)
    }
    assignments = np.full(len(features), -1, dtype=np.int64)
    origins = {}
    representative_indices = {}
    for reconstructed_index, cluster in enumerate(reconstructed):
        saved_index = reconstructed_to_saved[reconstructed_index]
        if np.any(assignments[cluster["mask"]] >= 0):
            raise RuntimeError(f"Overlapping reconstructed masks in {class_key}")
        assignments[cluster["mask"]] = saved_index
        origins[saved_index] = cluster["origin"]
        representative_indices[saved_index] = cluster["representative_index"]

    counts = np.bincount(assignments[assignments >= 0], minlength=args.ipc)
    if np.any(counts == 0):
        raise RuntimeError(
            f"Empty original CoDA clusters after reconstruction for {class_key}: "
            f"{np.flatnonzero(counts == 0).tolist()}"
        )
    return assignments, {
        "cluster_counts": counts.tolist(),
        "assigned_images": int(np.sum(assignments >= 0)),
        "unassigned_images": int(np.sum(assignments < 0)),
        "coverage_fraction": float(np.mean(assignments >= 0)),
        "initial_hdbscan_clusters": int(cluster_count),
        "maximum_representative_match_rmse": maximum_match_rmse,
        "representative_match_passed": representative_match_passed,
        "representative_match_rmse": matched_rmse.tolist(),
        "cluster_origins": [origins[index] for index in range(args.ipc)],
        "representative_source_indices": [
            representative_indices[index] for index in range(args.ipc)
        ],
    }


def load_partition(args):
    samples = []
    class_metadata = []
    for spec in args.specs:
        cluster_dir = Path(args.cluster_root) / spec
        selected, class_names = load_class_info(
            spec, args.misc_dir, nclass=args.nclass, phase=args.phase
        )
        chunks = {}
        centers_by_chunk = {}
        for local_label, class_id in enumerate(selected):
            chunk_id = local_label // 10
            if chunk_id not in chunks:
                feature_path = cluster_dir / f"{args.features_cache_name}_{chunk_id}"
                if not feature_path.is_file():
                    raise FileNotFoundError(f"Missing SDXL-VAE feature cache: {feature_path}")
                chunks[chunk_id] = _load_pickle(feature_path)
                saved_path = center_path(
                    cluster_dir, args.saved_clusters_base_name, chunk_id
                )
                if not saved_path.is_file():
                    raise FileNotFoundError(f"Missing saved representatives: {saved_path}")
                centers_by_chunk[chunk_id] = _load_pickle(saved_path)

            chunk = chunks[chunk_id]
            features = chunk["features"].get(local_label)
            paths = chunk["paths"].get(local_label)
            centers = centers_by_chunk[chunk_id].get(local_label)
            if features is None or paths is None or centers is None:
                raise KeyError(
                    f"Incomplete artifacts for {spec} local label {local_label} ({class_id})"
                )
            features = _flat_features(features)
            centers = _flat_features(centers)
            if len(features) != len(paths):
                raise ValueError(
                    f"Feature/path mismatch for {spec}/{class_id}: "
                    f"{len(features)} vs {len(paths)}"
                )
            if len(centers) != args.ipc:
                raise ValueError(
                    f"Expected {args.ipc} representatives for {spec}/{class_id}, "
                    f"found {len(centers)}"
                )

            distances = (
                np.sum(features * features, axis=1, keepdims=True)
                + np.sum(centers * centers, axis=1)[None, :]
                - 2.0 * features @ centers.T
            )
            class_key = f"{spec}:{class_id}"
            voronoi_assignments = np.argmin(distances, axis=1).astype(np.int64)
            print(f"Reconstructing original CoDA masks: {class_key}", flush=True)
            assignments, reconstruction = reconstruct_class_partition(
                args, features, centers, class_key
            )
            counts = np.asarray(reconstruction["cluster_counts"], dtype=np.int64)
            print(
                f"  replay frozen; historical_match="
                f"{reconstruction['representative_match_passed']}; coverage="
                f"{reconstruction['assigned_images']}/{len(paths)}, "
                f"cluster sizes={counts.tolist()}",
                flush=True,
            )
            class_metadata.append(
                {
                    "class_key": class_key,
                    "spec": spec,
                    "class_id": class_id,
                    "class_name": class_names[class_id],
                    "local_label": local_label,
                    "images": len(paths),
                    "evaluated_images": reconstruction["assigned_images"],
                    "unassigned_images": reconstruction["unassigned_images"],
                    "coverage_fraction": reconstruction["coverage_fraction"],
                    "cluster_counts": counts.tolist(),
                    "voronoi_cluster_counts": np.bincount(
                        voronoi_assignments, minlength=args.ipc
                    ).tolist(),
                    "minimum_cluster_size": int(counts.min()),
                    "maximum_cluster_size": int(counts.max()),
                    "initial_hdbscan_clusters": reconstruction[
                        "initial_hdbscan_clusters"
                    ],
                    "maximum_representative_match_rmse": reconstruction[
                        "maximum_representative_match_rmse"
                    ],
                    "representative_match_passed": reconstruction[
                        "representative_match_passed"
                    ],
                    "representative_match_rmse": reconstruction[
                        "representative_match_rmse"
                    ],
                    "cluster_origins": reconstruction["cluster_origins"],
                    "representative_source_indices": reconstruction[
                        "representative_source_indices"
                    ],
                }
            )
            for index, image_path in enumerate(paths):
                cluster_id = int(assignments[index])
                samples.append(
                    {
                        "sample_index": len(samples),
                        "class_key": class_key,
                        "spec": spec,
                        "class_id": class_id,
                        "class_name": class_names[class_id],
                        "local_label": local_label,
                        "cluster_id": cluster_id,
                        "included_in_original_partition": cluster_id >= 0,
                        "voronoi_cluster_id": int(voronoi_assignments[index]),
                        "distance_sq_to_representative": float(
                            max(distances[index, voronoi_assignments[index]], 0.0)
                        ),
                        "path": str(Path(image_path).resolve()),
                    }
                )
    return samples, class_metadata


def inventory_signature(samples, model_root, encoder_name):
    digest = hashlib.sha256()
    digest.update(b"p1_cluster_recoverability_features_v1")
    digest.update(str(Path(model_root).resolve()).encode("utf-8"))
    digest.update(encoder_name.encode("ascii"))
    for sample in samples:
        path = Path(sample["path"])
        if not path.is_file():
            raise FileNotFoundError(f"Missing source image: {path}")
        stat = path.stat()
        digest.update(str(path).encode("utf-8"))
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
    return digest.hexdigest()


class IndependentEncoder:
    def __init__(self, name, model_root, device):
        from transformers import (
            AutoImageProcessor,
            AutoModel,
            AutoProcessor,
            CLIPVisionModelWithProjection,
        )

        self.name = name
        self.model_root = model_root
        self.device = torch.device(device)
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        if name == "dino":
            self.processor = AutoImageProcessor.from_pretrained(
                model_root, local_files_only=True
            )
            self.model = AutoModel.from_pretrained(
                model_root, local_files_only=True, torch_dtype=self.dtype
            )
        elif name == "clip":
            self.processor = AutoProcessor.from_pretrained(
                model_root, local_files_only=True
            )
            self.model = CLIPVisionModelWithProjection.from_pretrained(
                model_root, local_files_only=True, torch_dtype=self.dtype
            )
        else:
            raise ValueError(f"Unknown encoder: {name}")
        self.model = self.model.to(self.device).eval()

    @torch.inference_mode()
    def extract(self, paths, batch_size):
        batches = []
        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start : start + batch_size]
            images = []
            for path in batch_paths:
                with Image.open(path) as image:
                    images.append(image.convert("RGB"))
            inputs = self.processor(images=images, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(
                device=self.device, dtype=self.dtype
            )
            output = self.model(pixel_values=pixel_values)
            if self.name == "clip":
                features = output.image_embeds
            else:
                features = output.last_hidden_state[:, 0]
            features = torch.nn.functional.normalize(features.float(), dim=1)
            batches.append(features.cpu().numpy())
            print(
                f"{self.name.upper()} features: "
                f"{min(start + batch_size, len(paths))}/{len(paths)}",
                flush=True,
            )
        return np.concatenate(batches, axis=0).astype(np.float32, copy=False)


def load_or_extract_features(args, samples, encoder_name, model_root):
    cache_dir = Path(args.output_dir) / "feature_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{encoder_name}.npz"
    signature = inventory_signature(samples, model_root, encoder_name)
    expected_paths = np.asarray([sample["path"] for sample in samples])
    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as cached:
            if str(cached["signature"].item()) != signature:
                raise RuntimeError(
                    f"Feature cache inputs changed: {cache_path}. Use a new run ID."
                )
            cached_paths = cached["paths"].astype(str)
            if not np.array_equal(cached_paths, expected_paths):
                raise RuntimeError(f"Feature cache path order changed: {cache_path}")
            print(f"Reusing {encoder_name} feature cache: {cache_path}")
            return cached["features"].astype(np.float32, copy=False)

    if args.resume:
        print(f"No reusable {encoder_name} cache found; extracting features")
    extractor = IndependentEncoder(encoder_name, model_root, args.device)
    features = extractor.extract(expected_paths.tolist(), args.batch_size)
    del extractor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    temporary = cache_path.with_name(cache_path.name + ".tmp.npz")
    np.savez_compressed(
        temporary,
        features=features,
        paths=expected_paths,
        signature=np.asarray(signature),
    )
    os.replace(temporary, cache_path)
    return features


def metric_values(y_true, y_pred, labels):
    return {
        "top1": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "balanced_accuracy": float(
            recall_score(
                y_true, y_pred, labels=labels, average="macro", zero_division=0
            )
        ),
    }


def nearest_centroid_predict(train_features, train_labels, test_features, labels):
    centroids = []
    for label in labels:
        selected = train_features[train_labels == label]
        if len(selected) == 0:
            raise ValueError(f"Training fold is missing cluster {label}")
        centroid = selected.mean(axis=0)
        centroid /= max(float(np.linalg.norm(centroid)), 1e-12)
        centroids.append(centroid)
    centroids = np.stack(centroids)
    predictions = np.argmax(test_features @ centroids.T, axis=1)
    return np.asarray(labels, dtype=np.int64)[predictions]


def evaluate_labels(
    features, labels, folds, all_labels, ridge_alpha, classifiers=CLASSIFIERS
):
    fold_metrics = {classifier: [] for classifier in classifiers}
    for train_indices, test_indices in folds:
        x_train = features[train_indices]
        x_test = features[test_indices]
        y_train = labels[train_indices]
        y_test = labels[test_indices]

        if "linear_probe" in classifiers:
            probe = make_pipeline(
                StandardScaler(),
                RidgeClassifier(alpha=ridge_alpha, class_weight="balanced"),
            )
            probe.fit(x_train, y_train)
            fold_metrics["linear_probe"].append(
                metric_values(y_test, probe.predict(x_test), all_labels)
            )
        if "nearest_centroid" in classifiers:
            centroid_predictions = nearest_centroid_predict(
                x_train, y_train, x_test, all_labels
            )
            fold_metrics["nearest_centroid"].append(
                metric_values(y_test, centroid_predictions, all_labels)
            )

    return {
        classifier: {
            metric: float(np.mean([row[metric] for row in rows]))
            for metric in METRICS
        }
        for classifier, rows in fold_metrics.items()
    }


def class_seed(base_seed, class_key):
    digest = hashlib.sha256(class_key.encode("utf-8")).digest()
    return (base_seed + int.from_bytes(digest[:4], "little")) % (2**32)


def matched_random_labels(labels, rng):
    random_labels = rng.permutation(labels)
    if not np.array_equal(
        np.bincount(random_labels, minlength=int(labels.max()) + 1),
        np.bincount(labels, minlength=int(labels.max()) + 1),
    ):
        raise AssertionError("Matched-random partition changed cluster occupancy")
    return random_labels


def evaluate_class(
    class_key,
    spec,
    class_id,
    class_name,
    features,
    labels,
    folds_requested,
    null_partitions,
    linear_null_partitions,
    base_seed,
    ridge_alpha,
):
    original_counts = np.bincount(labels)
    excluded_labels = np.flatnonzero(original_counts < MINIMUM_CV_CLUSTER_SIZE)
    if len(excluded_labels):
        print(
            f"CV support exclusion for {class_key}: clusters "
            f"{excluded_labels.tolist()} contain "
            f"{int(original_counts[excluded_labels].sum())} image(s)",
            flush=True,
        )
        retained = ~np.isin(labels, excluded_labels)
        features = features[retained]
        labels = labels[retained]
    retained_labels = np.unique(labels)
    if len(retained_labels) < 2:
        raise ValueError(
            f"Fewer than two evaluable clusters in {class_key}: "
            f"{original_counts.tolist()}"
        )
    remap = {int(label): index for index, label in enumerate(retained_labels)}
    labels = np.asarray([remap[int(label)] for label in labels], dtype=np.int64)
    counts = np.bincount(labels, minlength=len(retained_labels))
    n_splits = min(folds_requested, int(counts.min()))
    if n_splits < 2:
        raise ValueError(f"Not enough images for CV in {class_key}: {counts.tolist()}")
    splitter = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=class_seed(base_seed, class_key)
    )
    folds = list(splitter.split(features, labels))
    all_labels = np.arange(len(counts), dtype=np.int64)
    true_metrics = evaluate_labels(features, labels, folds, all_labels, ridge_alpha)

    rng = np.random.default_rng(class_seed(base_seed + 1, class_key))
    null_metrics = []
    for null_index in range(null_partitions):
        random_labels = matched_random_labels(labels, rng)
        null_splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=class_seed(base_seed + 2 + null_index, class_key),
        )
        null_folds = list(null_splitter.split(features, random_labels))
        classifiers = ["nearest_centroid"]
        if null_index < linear_null_partitions:
            classifiers.append("linear_probe")
        metrics = evaluate_labels(
            features,
            random_labels,
            null_folds,
            all_labels,
            ridge_alpha,
            classifiers=tuple(classifiers),
        )
        null_metrics.append(metrics)
    return {
        "class_key": class_key,
        "spec": spec,
        "class_id": class_id,
        "class_name": class_name,
        "images": int(original_counts.sum()),
        "evaluated_images": len(labels),
        "excluded_sparse_images": int(
            original_counts[excluded_labels].sum()
        ),
        "excluded_sparse_clusters": excluded_labels.tolist(),
        "original_cluster_counts": original_counts.tolist(),
        "evaluated_cluster_ids": retained_labels.tolist(),
        "cluster_counts": counts.tolist(),
        "folds": n_splits,
        "true": true_metrics,
        "null": null_metrics,
    }


def one_sided_p(true_value, null_values):
    null_values = np.asarray(null_values, dtype=np.float64)
    return float((1 + np.sum(null_values >= true_value)) / (len(null_values) + 1))


def comparison_row(scope, encoder, classifier, metric, true_value, null_values, **extra):
    null_values = np.asarray(null_values, dtype=np.float64)
    row = {
        "scope": scope,
        "encoder": encoder,
        "classifier": classifier,
        "metric": metric,
        "true_value": float(true_value),
        "null_mean": float(null_values.mean()),
        "null_std": float(null_values.std(ddof=1)) if len(null_values) > 1 else 0.0,
        "delta_over_null": float(true_value - null_values.mean()),
        "null_percentile": float(np.mean(null_values < true_value)),
        "permutation_p_one_sided": one_sided_p(true_value, null_values),
        "null_partitions": int(len(null_values)),
    }
    row.update(extra)
    return row


def summarize_encoder(encoder_name, results, null_partitions):
    per_class_rows = []
    null_rows = []
    aggregate_rows = []
    for result in results:
        for classifier in CLASSIFIERS:
            for metric in METRICS:
                null_values = [
                    item[classifier][metric]
                    for item in result["null"]
                    if classifier in item
                ]
                per_class_rows.append(
                    comparison_row(
                        "class",
                        encoder_name,
                        classifier,
                        metric,
                        result["true"][classifier][metric],
                        null_values,
                        spec=result["spec"],
                        class_id=result["class_id"],
                        class_name=result["class_name"],
                        class_key=result["class_key"],
                        images=result["images"],
                        evaluated_images=result["evaluated_images"],
                        excluded_sparse_images=result["excluded_sparse_images"],
                        excluded_sparse_clusters=";".join(
                            str(value)
                            for value in result["excluded_sparse_clusters"]
                        ),
                        minimum_cluster_size=min(result["cluster_counts"]),
                        maximum_cluster_size=max(result["cluster_counts"]),
                    )
                )
        for null_index, metrics in enumerate(result["null"]):
            for classifier in CLASSIFIERS:
                if classifier not in metrics:
                    continue
                null_rows.append(
                    {
                        "encoder": encoder_name,
                        "spec": result["spec"],
                        "class_id": result["class_id"],
                        "class_name": result["class_name"],
                        "class_key": result["class_key"],
                        "null_index": null_index,
                        "classifier": classifier,
                        **metrics[classifier],
                    }
                )

    scopes = [("combined", results)]
    for spec in sorted({result["spec"] for result in results}):
        scopes.append((spec, [result for result in results if result["spec"] == spec]))
    for scope, selected in scopes:
        for classifier in CLASSIFIERS:
            for metric in METRICS:
                true_value = float(
                    np.mean([item["true"][classifier][metric] for item in selected])
                )
                available_indices = [
                    index
                    for index, metrics in enumerate(selected[0]["null"])
                    if classifier in metrics
                ]
                null_values = [
                    float(
                        np.mean(
                            [item["null"][index][classifier][metric] for item in selected]
                        )
                    )
                    for index in available_indices
                ]
                aggregate_rows.append(
                    comparison_row(
                        scope,
                        encoder_name,
                        classifier,
                        metric,
                        true_value,
                        null_values,
                        classes=len(selected),
                    )
                )
    return per_class_rows, aggregate_rows, null_rows


def plot_results(output_dir, per_class_rows, aggregate_rows):
    output_dir = Path(output_dir)
    primary = [
        row
        for row in aggregate_rows
        if row["scope"] == "combined"
        and row["classifier"] == "nearest_centroid"
        and row["metric"] == "macro_f1"
    ]
    heatmap_rows = [
        row
        for row in per_class_rows
        if row["classifier"] == "linear_probe" and row["metric"] == "macro_f1"
    ]
    encoders = ["dino", "clip"]
    class_keys = sorted({row["class_key"] for row in heatmap_rows})
    values = np.full((len(encoders), len(class_keys)), np.nan, dtype=np.float32)
    lookup = {
        (row["encoder"], row["class_key"]): row["delta_over_null"]
        for row in heatmap_rows
    }
    for encoder_index, encoder in enumerate(encoders):
        for class_index, class_key in enumerate(class_keys):
            values[encoder_index, class_index] = lookup[(encoder, class_key)]

    figure, axes = plt.subplots(2, 1, figsize=(max(14, len(class_keys) * 0.55), 9))
    positions = np.arange(len(primary))
    true_values = [row["true_value"] for row in primary]
    null_values = [row["null_mean"] for row in primary]
    null_std = [row["null_std"] for row in primary]
    width = 0.34
    axes[0].bar(positions - width / 2, true_values, width, label="VAE partition")
    axes[0].bar(
        positions + width / 2,
        null_values,
        width,
        yerr=null_std,
        label="matched random",
        capsize=5,
    )
    axes[0].set_xticks(positions, [row["encoder"] for row in primary])
    axes[0].set_ylabel("5-fold Macro-F1")
    axes[0].set_title("Cross-representation recoverability")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)

    bound = max(float(np.nanmax(np.abs(values))), 1e-6)
    image = axes[1].imshow(values, aspect="auto", cmap="coolwarm", vmin=-bound, vmax=bound)
    axes[1].set_yticks(np.arange(len(encoders)), encoders)
    axes[1].set_xticks(np.arange(len(class_keys)), class_keys, rotation=75, ha="right")
    axes[1].set_title("Per-class linear-probe Macro-F1 gain over matched random")
    figure.colorbar(image, ax=axes[1], label="Delta Macro-F1")
    figure.tight_layout()
    figure.savefig(output_dir / "cluster_recoverability.png", dpi=180)
    plt.close(figure)


def main():
    args = parse_args()
    if args.null_partitions < 1:
        raise ValueError("--null-partitions must be positive")
    if not 1 <= args.linear_null_partitions <= args.null_partitions:
        raise ValueError(
            "--linear-null-partitions must be in [1, --null-partitions]"
        )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples, class_metadata = load_or_reconstruct_partition(args, output_dir)
    assignment_rows = [
        {
            key: sample[key]
            for key in (
                "sample_index",
                "class_key",
                "spec",
                "class_id",
                "class_name",
                "local_label",
                "cluster_id",
                "included_in_original_partition",
                "voronoi_cluster_id",
                "distance_sq_to_representative",
                "path",
            )
        }
        for sample in samples
    ]
    write_csv(output_dir / "assignments.csv", assignment_rows)
    atomic_json(
        output_dir / "partition_manifest.json",
        {
            "format_version": 2,
            "assignment_definition": (
                "CoDA final points_mask replayed and frozen in the current environment "
                "with StandardScaler, UMAP, HDBSCAN, and hdbscan_post; label IDs are "
                "permutation-mapped to historical representatives only for auditing"
            ),
            "audit_assignment": (
                "voronoi_cluster_id is the nearest persisted representative in raw "
                "flattened SDXL-VAE space and is not used as the P1 target"
            ),
            "specs": args.specs,
            "ipc": args.ipc,
            "clustering": {
                "n_neighbors": args.n_neighbors,
                "min_cluster_size": args.min_cluster_size,
                "min_samples": args.min_samples,
                "num_seed_candidates": args.num_seed_candidates,
                "umap_random_state": 42,
                "representative_match_rmse_tolerance": args.representative_match_rmse,
                "strict_representative_match": args.strict_representative_match,
            },
            "images": len(samples),
            "evaluated_images": int(
                sum(sample["included_in_original_partition"] for sample in samples)
            ),
            "cv_support_policy": (
                "clusters with fewer than two assigned images cannot support held-out "
                "evaluation and are excluded symmetrically from the true partition "
                "and its exact-occupancy matched-random null"
            ),
            "classes": class_metadata,
        },
    )

    labels_by_class = defaultdict(list)
    indices_by_class = defaultdict(list)
    metadata_by_class = {}
    for sample in samples:
        if not sample["included_in_original_partition"]:
            continue
        key = sample["class_key"]
        indices_by_class[key].append(sample["sample_index"])
        labels_by_class[key].append(sample["cluster_id"])
        metadata_by_class[key] = sample

    all_per_class = []
    all_aggregate = []
    all_null = []
    model_roots = {"dino": args.dino_model, "clip": args.clip_model}
    for encoder_name, model_root in model_roots.items():
        features = load_or_extract_features(
            args, samples, encoder_name, model_root
        )
        tasks = []
        for class_key in sorted(indices_by_class):
            indices = np.asarray(indices_by_class[class_key], dtype=np.int64)
            metadata = metadata_by_class[class_key]
            tasks.append(
                (
                    class_key,
                    metadata["spec"],
                    metadata["class_id"],
                    metadata["class_name"],
                    features[indices],
                    np.asarray(labels_by_class[class_key], dtype=np.int64),
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
        per_class, aggregate, null_rows = summarize_encoder(
            encoder_name, results, args.null_partitions
        )
        all_per_class.extend(per_class)
        all_aggregate.extend(aggregate)
        all_null.extend(null_rows)
        del features, results
        gc.collect()

    write_csv(output_dir / "per_class_metrics.csv", all_per_class)
    write_csv(output_dir / "aggregate_metrics.csv", all_aggregate)
    write_csv(output_dir / "matched_random_metrics.csv", all_null)
    plot_results(output_dir, all_per_class, all_aggregate)

    primary = {
        row["encoder"]: row
        for row in all_aggregate
        if row["scope"] == "combined"
        and row["classifier"] == "nearest_centroid"
        and row["metric"] == "macro_f1"
    }
    passed = {
        encoder: row["delta_over_null"] > 0
        and row["permutation_p_one_sided"] <= 0.05
        for encoder, row in primary.items()
    }
    if all(passed.values()):
        decision = "pass"
    elif any(passed.values()):
        decision = "weak_or_representation_specific"
    else:
        decision = "fail"
    summary = {
        "format_version": 2,
        "p1_decision": decision,
        "decision_rule": (
            "pass iff both DINO and CLIP combined nearest-centroid Macro-F1 exceed "
            "their exact-occupancy matched-random null with one-sided p <= 0.05; "
            "one encoder passing is weak_or_representation_specific"
        ),
        "interpretation_boundary": (
            "A pass establishes cross-representation visual recoverability, not "
            "language describability or semantic purity. When historical_match_all "
            "is false, the claim applies to the frozen current-environment replay, "
            "not the unavailable historical points_mask."
        ),
        "partition_replay": {
            "historical_match_all": all(
                row["representative_match_passed"] for row in class_metadata
            ),
            "maximum_historical_representative_rmse": max(
                row["maximum_representative_match_rmse"]
                for row in class_metadata
            ),
            "evaluated_images": sum(
                row["evaluated_images"] for row in class_metadata
            ),
            "total_images": sum(row["images"] for row in class_metadata),
            "cv_minimum_cluster_size": MINIMUM_CV_CLUSTER_SIZE,
            "cv_sparse_clusters": sum(
                sum(
                    count < MINIMUM_CV_CLUSTER_SIZE
                    for count in row["cluster_counts"]
                )
                for row in class_metadata
            ),
            "cv_sparse_images": sum(
                sum(
                    count
                    for count in row["cluster_counts"]
                    if count < MINIMUM_CV_CLUSTER_SIZE
                )
                for row in class_metadata
            ),
        },
        "primary": primary,
        "configuration": {
            "specs": args.specs,
            "dino_model": str(Path(args.dino_model).resolve()),
            "clip_model": str(Path(args.clip_model).resolve()),
            "folds": args.folds,
            "null_partitions": args.null_partitions,
            "linear_null_partitions": args.linear_null_partitions,
            "random_seed": args.random_seed,
            "ridge_alpha": args.ridge_alpha,
        },
    }
    atomic_json(output_dir / "summary.json", summary)
    atomic_json(output_dir / "complete.json", {"status": "complete"})
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
