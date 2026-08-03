"""Freeze P4 prompt pairs and real-image-only cluster probes."""

import argparse
import csv
import hashlib
import json
import os
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare frozen P4 execution inputs")
    parser.add_argument("--p1-run-dir", required=True)
    parser.add_argument("--p2p3-run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--specs", nargs="+", default=["imageA", "imageB", "imageC"])
    parser.add_argument("--shuffle-shift", type=int, default=1)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--dino-model", required=True)
    parser.add_argument("--clip-model", required=True)
    return parser.parse_args()


def atomic_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path, rows):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def read_csv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_features(features):
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.maximum(norms, 1e-12)


def load_assignments(path, specs):
    rows = read_csv(path)
    selected = []
    for row in rows:
        if row["spec"] not in specs:
            continue
        row = dict(row)
        row["sample_index"] = int(row["sample_index"])
        row["cluster_id"] = int(row["cluster_id"])
        row["included"] = row["included_in_original_partition"].lower() == "true"
        selected.append(row)
    if not selected:
        raise ValueError("No P1 assignments matched the requested specs")
    return selected


def build_prompt_inputs(assignments, dcs_payload, specs, shuffle_shift, output_dir):
    counts = Counter(
        (row["class_key"], row["cluster_id"])
        for row in assignments if row["included"]
    )
    class_info = {}
    for row in assignments:
        class_info[row["class_key"]] = row

    summaries = dcs_payload.get("classes")
    if not isinstance(summaries, dict):
        raise ValueError("P2/P3 DCS payload is missing classes")

    pair_rows = []
    eligible_by_class = {}
    captions_by_spec = defaultdict(dict)
    indices_by_spec = defaultdict(dict)
    maps_by_spec = defaultdict(dict)
    excluded = []
    for class_key in sorted(class_info):
        info = class_info[class_key]
        spec = info["spec"]
        if spec not in specs:
            continue
        class_id = info["class_id"]
        class_payload = summaries.get(class_key, {})
        clusters = class_payload.get("clusters", {})
        eligible = sorted(
            int(cluster_id)
            for cluster_id, entry in clusters.items()
            if isinstance(entry, dict)
            and isinstance(entry.get("caption"), str)
            and entry["caption"].strip()
            and counts[(class_key, int(cluster_id))] >= 2
        )
        all_evaluable = sorted(
            cluster_id
            for (key, cluster_id), count in counts.items()
            if key == class_key and count >= 2
        )
        excluded.extend(
            {
                "class_key": class_key,
                "spec": spec,
                "class_id": class_id,
                "cluster_id": cluster_id,
                "real_members": counts[(class_key, cluster_id)],
                "reason": "missing_dcs_summary",
            }
            for cluster_id in all_evaluable if cluster_id not in eligible
        )
        if len(eligible) < 2:
            raise ValueError(f"Fewer than two P4-eligible clusters in {class_key}: {eligible}")
        eligible_by_class[class_key] = eligible
        shift = shuffle_shift % len(eligible)
        if shift == 0:
            raise ValueError(
                f"shuffle shift maps captions to themselves in {class_key}; "
                "choose a non-zero shift modulo the eligible cluster count"
            )
        captions_by_spec[spec][class_id] = {
            str(cluster_id): clusters[str(cluster_id)]["caption"].strip()
            for cluster_id in eligible
        }
        indices_by_spec[spec][class_id] = eligible
        maps_by_spec[spec][class_id] = {}
        for position, cluster_id in enumerate(eligible):
            caption_source = eligible[(position + shift) % len(eligible)]
            maps_by_spec[spec][class_id][str(cluster_id)] = caption_source
            pair_rows.append(
                {
                    "spec": spec,
                    "class_key": class_key,
                    "class_id": class_id,
                    "class_name": info["class_name"],
                    "visual_cluster_id": cluster_id,
                    "correct_caption_cluster_id": cluster_id,
                    "shuffled_caption_cluster_id": caption_source,
                    "real_members": counts[(class_key, cluster_id)],
                }
            )

    for spec in specs:
        if spec not in captions_by_spec:
            raise ValueError(f"No P4 prompt inputs were built for {spec}")
        atomic_json(
            output_dir / f"{spec}_dcs_captions.json",
            {
                "metadata": {
                    "format_version": 1,
                    "source": "P2/P3 replayed DCS summaries",
                },
                "captions": captions_by_spec[spec],
            },
        )
        atomic_json(output_dir / f"{spec}_cluster_indices.json", indices_by_spec[spec])
        atomic_json(output_dir / f"{spec}_shuffled_source_map.json", maps_by_spec[spec])
    write_csv(output_dir / "pair_manifest.csv", pair_rows)
    if excluded:
        write_csv(output_dir / "excluded_clusters.csv", excluded)
    return pair_rows, eligible_by_class, excluded


def freeze_probes(
    assignments, eligible_by_class, p1_run_dir, ridge_alpha, model_roots
):
    by_class = defaultdict(list)
    for row in assignments:
        if row["included"]:
            by_class[row["class_key"]].append(row)
    probes = {
        "format_version": 1,
        "training_data": "real_images_only",
        "ridge_alpha": ridge_alpha,
        "encoders": {},
    }
    for encoder in ("dino", "clip"):
        cache_path = p1_run_dir / "feature_cache" / f"{encoder}.npz"
        if not cache_path.is_file():
            raise FileNotFoundError(f"Missing P1 feature cache: {cache_path}")
        with np.load(cache_path, allow_pickle=False) as cached:
            features = cached["features"].astype(np.float32, copy=False)
            paths = cached["paths"].astype(str)
            cache_signature = str(cached["signature"].item())
        encoder_payload = {
            "feature_cache_signature": cache_signature,
            "model_root": str(Path(model_roots[encoder]).resolve()),
            "classes": {},
        }
        for class_key in sorted(by_class):
            rows = by_class[class_key]
            indices = np.asarray([row["sample_index"] for row in rows], dtype=np.int64)
            expected_paths = np.asarray([str(Path(row["path"]).resolve()) for row in rows])
            actual_paths = np.asarray([str(Path(path).resolve()) for path in paths[indices]])
            if not np.array_equal(expected_paths, actual_paths):
                raise RuntimeError(f"P1 feature order mismatch for {encoder}/{class_key}")
            x = normalize_features(features[indices])
            y = np.asarray([row["cluster_id"] for row in rows], dtype=np.int64)
            evaluable_ids = np.asarray(
                sorted(cluster_id for cluster_id, count in Counter(y).items() if count >= 2),
                dtype=np.int64,
            )
            keep = np.isin(y, evaluable_ids)
            x = x[keep]
            y = y[keep]
            scaler = StandardScaler().fit(x)
            classifier = RidgeClassifier(alpha=ridge_alpha, class_weight="balanced")
            classifier.fit(scaler.transform(x), y)
            centroids = np.stack(
                [normalize_features(x[y == cluster_id].mean(axis=0, keepdims=True))[0]
                 for cluster_id in evaluable_ids]
            ).astype(np.float32)
            encoder_payload["classes"][class_key] = {
                "class_ids": classifier.classes_.astype(np.int64),
                "centroid_cluster_ids": evaluable_ids,
                "centroids": centroids,
                "scaler_mean": scaler.mean_.astype(np.float32),
                "scaler_scale": scaler.scale_.astype(np.float32),
                "ridge_coef": classifier.coef_.astype(np.float32),
                "ridge_intercept": np.asarray(classifier.intercept_, dtype=np.float32),
                "real_images": int(len(y)),
                "p4_eligible_cluster_ids": np.asarray(
                    eligible_by_class[class_key], dtype=np.int64
                ),
            }
        probes["encoders"][encoder] = encoder_payload
    return probes


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    p1_run_dir = Path(args.p1_run_dir)
    p2p3_run_dir = Path(args.p2p3_run_dir)
    assignments_path = p1_run_dir / "assignments.csv"
    dcs_path = p2p3_run_dir / "replayed_dcs_summaries.json"
    assignments = load_assignments(assignments_path, set(args.specs))
    dcs_payload = json.loads(dcs_path.read_text(encoding="utf-8"))
    pair_rows, eligible_by_class, excluded = build_prompt_inputs(
        assignments,
        dcs_payload,
        set(args.specs),
        args.shuffle_shift,
        output_dir,
    )
    probes = freeze_probes(
        assignments,
        eligible_by_class,
        p1_run_dir,
        args.ridge_alpha,
        {"dino": args.dino_model, "clip": args.clip_model},
    )
    probe_path = output_dir / "frozen_real_image_probes.pkl"
    temporary = probe_path.with_suffix(probe_path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(probes, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, probe_path)
    atomic_json(
        output_dir / "preparation_summary.json",
        {
            "format_version": 1,
            "specs": sorted(args.specs),
            "pairs": len(pair_rows),
            "excluded_clusters": len(excluded),
            "shuffle_shift_within_eligible_order": args.shuffle_shift,
            "ridge_alpha": args.ridge_alpha,
            "probe_training_data": "P1 real-image feature caches only",
            "assignments_sha256": file_sha256(assignments_path),
            "dcs_summaries_sha256": file_sha256(dcs_path),
            "dino_feature_cache_sha256": file_sha256(
                p1_run_dir / "feature_cache" / "dino.npz"
            ),
            "clip_feature_cache_sha256": file_sha256(
                p1_run_dir / "feature_cache" / "clip.npz"
            ),
            "probe_file": probe_path.name,
        },
    )
    print(f"Prepared {len(pair_rows)} P4 cluster pairs in {output_dir}")


if __name__ == "__main__":
    main()
