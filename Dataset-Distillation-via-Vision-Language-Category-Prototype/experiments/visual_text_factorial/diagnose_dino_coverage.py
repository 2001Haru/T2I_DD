import argparse
import csv
import gc
import json
import os
import statistics
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from diagnostic_common import (
    file_inventory_signature,
    image_paths,
    load_json,
    parse_shift_runs,
    seed_directories,
)


BASE_CONDITIONS = (
    "no_visual_label",
    "no_visual_dcs",
    "no_visual_dcs_shuffled",
    "prototype_label",
    "prototype_dcs",
    "prototype_dcs_shuffled",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure real-manifold coverage of visual x text synthetic datasets"
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--base-run-root", required=True)
    parser.add_argument(
        "--shuffle-run",
        action="append",
        default=[],
        metavar="SHIFT=RUN_ROOT",
    )
    parser.add_argument("--dino-model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--nn-block-size", type=int, default=1024)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def condition_metadata(condition, shift):
    if condition.startswith("no_visual_"):
        visual_mode = "no_visual"
        prompt_mode = condition.removeprefix("no_visual_")
    elif condition.startswith("prototype_"):
        visual_mode = "prototype"
        prompt_mode = condition.removeprefix("prototype_")
    else:
        raise ValueError(f"Unknown condition: {condition}")
    if prompt_mode == "dcs_shuffled":
        return visual_mode, "shuffled_dcs", shift
    if prompt_mode == "dcs":
        return visual_mode, "correct_dcs", 0
    if prompt_mode == "label":
        return visual_mode, "label", None
    raise ValueError(f"Unknown condition: {condition}")


def discover_datasets(base_run_root, shuffle_runs):
    datasets = []
    for seed_dir in seed_directories(base_run_root):
        generation_seed = int(seed_dir.name.split("_")[-1])
        for condition in BASE_CONDITIONS:
            condition_dir = seed_dir / condition
            manifest_path = condition_dir / "manifest.json"
            if not manifest_path.is_file() or not (condition_dir / "complete.json").is_file():
                raise FileNotFoundError(f"Incomplete base condition: {condition_dir}")
            manifest = load_json(manifest_path)
            visual_mode, prompt_condition, shift = condition_metadata(condition, 1)
            datasets.append(
                {
                    "dataset_key": f"base_seed{generation_seed}_{condition}",
                    "run_group": "base",
                    "generation_seed": generation_seed,
                    "visual_mode": visual_mode,
                    "prompt_condition": prompt_condition,
                    "shuffle_shift": shift,
                    "root": condition_dir,
                    "manifest": manifest,
                }
            )

    for shift, run_root in sorted(shuffle_runs.items()):
        for seed_dir in seed_directories(run_root):
            generation_seed = int(seed_dir.name.split("_")[-1])
            for visual_mode in ("no_visual", "prototype"):
                condition = f"{visual_mode}_dcs_shuffled"
                condition_dir = seed_dir / condition
                manifest_path = condition_dir / "manifest.json"
                if not manifest_path.is_file() or not (
                    condition_dir / "complete.json"
                ).is_file():
                    raise FileNotFoundError(f"Incomplete shuffled condition: {condition_dir}")
                manifest = load_json(manifest_path)
                manifest_shift = int(manifest["shuffle_strategy"]["shift"])
                if manifest_shift != shift:
                    raise RuntimeError(
                        f"Expected shift {shift}, found {manifest_shift} in {manifest_path}"
                    )
                datasets.append(
                    {
                        "dataset_key": (
                            f"shift{shift}_seed{generation_seed}_{condition}"
                        ),
                        "run_group": f"shift_{shift}",
                        "generation_seed": generation_seed,
                        "visual_mode": visual_mode,
                        "prompt_condition": "shuffled_dcs",
                        "shuffle_shift": shift,
                        "root": condition_dir,
                        "manifest": manifest,
                    }
                )
    return datasets


class DinoExtractor:
    def __init__(self, model_root, device):
        from transformers import AutoImageProcessor, AutoModel

        self.device = torch.device(device)
        self.processor = AutoImageProcessor.from_pretrained(
            model_root, local_files_only=True
        )
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.model = AutoModel.from_pretrained(
            model_root, local_files_only=True, torch_dtype=dtype
        ).to(self.device)
        self.model.eval()
        self.dtype = dtype

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
            features = output.last_hidden_state[:, 0].float()
            features = torch.nn.functional.normalize(features, dim=1)
            batches.append(features.cpu().numpy())
            print(f"DINO features: {min(start + batch_size, len(paths))}/{len(paths)}")
        return np.concatenate(batches, axis=0).astype(np.float32, copy=False)


def feature_cache_path(cache_dir, dataset_key):
    return Path(cache_dir) / f"{dataset_key}.npz"


def load_or_extract_features(
    extractor,
    paths,
    labels,
    cache_path,
    model_root,
    batch_size,
    resume,
):
    signature = file_inventory_signature(
        paths,
        extra={"model_root": str(Path(model_root).resolve()), "schema_version": 1},
    )
    cache_path = Path(cache_path)
    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as cached:
            cached_signature = str(cached["signature"].item())
            if cached_signature != signature:
                raise RuntimeError(
                    f"Feature cache input changed: {cache_path}. "
                    "Remove the cache or use a new diagnostics directory."
                )
            return (
                cached["features"].astype(np.float32, copy=False),
                cached["labels"].astype(str),
                cached["paths"].astype(str),
            )
    if resume:
        print(f"No reusable cache found; extracting {cache_path.stem}")
    features = extractor.extract(paths, batch_size)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(".tmp.npz")
    np.savez(
        temporary,
        features=features,
        labels=np.asarray(labels),
        paths=np.asarray([str(path.resolve()) for path in paths]),
        signature=np.asarray(signature),
    )
    os.replace(temporary, cache_path)
    return features, np.asarray(labels), np.asarray([str(path) for path in paths])


def nearest_self_distances(features, block_size):
    nearest = np.empty(len(features), dtype=np.float32)
    for start in range(0, len(features), block_size):
        end = min(start + block_size, len(features))
        similarities = features[start:end] @ features.T
        row_indices = np.arange(end - start)
        similarities[row_indices, np.arange(start, end)] = -np.inf
        nearest[start:end] = np.clip(
            1.0 - similarities.max(axis=1), a_min=0.0, a_max=2.0
        )
    return nearest


def pairwise_diversity(features):
    if len(features) < 2:
        return 0.0
    similarities = features @ features.T
    upper = np.triu_indices(len(features), k=1)
    return float(
        np.mean(np.clip(1.0 - similarities[upper], a_min=0.0, a_max=2.0))
    )


def class_metrics(real_features, synthetic_features, real_radius):
    similarities = real_features @ synthetic_features.T
    real_to_synthetic = np.clip(
        1.0 - similarities.max(axis=1), a_min=0.0, a_max=2.0
    )
    synthetic_to_real = np.clip(
        1.0 - similarities.max(axis=0), a_min=0.0, a_max=2.0
    )
    real_centroid = real_features.mean(axis=0)
    synthetic_centroid = synthetic_features.mean(axis=0)
    real_centroid /= max(np.linalg.norm(real_centroid), 1e-12)
    synthetic_centroid /= max(np.linalg.norm(synthetic_centroid), 1e-12)
    return {
        "real_to_synthetic_nn_mean": float(real_to_synthetic.mean()),
        "real_to_synthetic_nn_median": float(np.median(real_to_synthetic)),
        "synthetic_to_real_nn_mean": float(synthetic_to_real.mean()),
        "synthetic_to_real_nn_median": float(np.median(synthetic_to_real)),
        "coverage_at_real_nn95": float(np.mean(real_to_synthetic <= real_radius)),
        "precision_at_real_nn95": float(np.mean(synthetic_to_real <= real_radius)),
        "synthetic_pairwise_distance_mean": pairwise_diversity(synthetic_features),
        "centroid_cosine_distance": float(
            np.clip(1.0 - real_centroid @ synthetic_centroid, 0.0, 2.0)
        ),
        "real_nn95_radius": float(real_radius),
        "real_images": int(len(real_features)),
        "synthetic_images": int(len(synthetic_features)),
    }


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def macro_summary(rows):
    group_fields = (
        "dataset_key",
        "run_group",
        "generation_seed",
        "visual_mode",
        "prompt_condition",
        "shuffle_shift",
    )
    metric_fields = (
        "real_to_synthetic_nn_mean",
        "real_to_synthetic_nn_median",
        "synthetic_to_real_nn_mean",
        "synthetic_to_real_nn_median",
        "coverage_at_real_nn95",
        "precision_at_real_nn95",
        "synthetic_pairwise_distance_mean",
        "centroid_cosine_distance",
    )
    grouped = {}
    for row in rows:
        key = tuple(row[field] for field in group_fields)
        grouped.setdefault(key, []).append(row)
    summaries = []
    for key, selected in grouped.items():
        summary = dict(zip(group_fields, key))
        summary["classes"] = len(selected)
        for metric in metric_fields:
            summary[metric] = statistics.fmean(float(row[metric]) for row in selected)
        summaries.append(summary)
    return summaries


def paired_deltas(summaries):
    correct = {
        (row["generation_seed"], row["visual_mode"]): row
        for row in summaries
        if row["prompt_condition"] == "correct_dcs"
    }
    rows = []
    for shuffled in summaries:
        if shuffled["prompt_condition"] != "shuffled_dcs":
            continue
        key = (shuffled["generation_seed"], shuffled["visual_mode"])
        baseline = correct[key]
        rows.append(
            {
                "generation_seed": shuffled["generation_seed"],
                "visual_mode": shuffled["visual_mode"],
                "shuffle_shift": shuffled["shuffle_shift"],
                "coverage_distance_improvement": (
                    baseline["real_to_synthetic_nn_mean"]
                    - shuffled["real_to_synthetic_nn_mean"]
                ),
                "fidelity_distance_improvement": (
                    baseline["synthetic_to_real_nn_mean"]
                    - shuffled["synthetic_to_real_nn_mean"]
                ),
                "coverage_fraction_change": (
                    shuffled["coverage_at_real_nn95"]
                    - baseline["coverage_at_real_nn95"]
                ),
                "precision_fraction_change": (
                    shuffled["precision_at_real_nn95"]
                    - baseline["precision_at_real_nn95"]
                ),
                "diversity_change": (
                    shuffled["synthetic_pairwise_distance_mean"]
                    - baseline["synthetic_pairwise_distance_mean"]
                ),
                "centroid_distance_improvement": (
                    baseline["centroid_cosine_distance"]
                    - shuffled["centroid_cosine_distance"]
                ),
            }
        )
    return rows


def paired_class_deltas(class_rows):
    correct = {
        (row["generation_seed"], row["visual_mode"], row["synset"]): row
        for row in class_rows
        if row["prompt_condition"] == "correct_dcs"
    }
    rows = []
    for shuffled in class_rows:
        if shuffled["prompt_condition"] != "shuffled_dcs":
            continue
        key = (
            shuffled["generation_seed"],
            shuffled["visual_mode"],
            shuffled["synset"],
        )
        baseline = correct[key]
        rows.append(
            {
                "generation_seed": shuffled["generation_seed"],
                "visual_mode": shuffled["visual_mode"],
                "shuffle_shift": shuffled["shuffle_shift"],
                "synset": shuffled["synset"],
                "coverage_distance_improvement": (
                    baseline["real_to_synthetic_nn_mean"]
                    - shuffled["real_to_synthetic_nn_mean"]
                ),
                "fidelity_distance_improvement": (
                    baseline["synthetic_to_real_nn_mean"]
                    - shuffled["synthetic_to_real_nn_mean"]
                ),
                "coverage_fraction_change": (
                    shuffled["coverage_at_real_nn95"]
                    - baseline["coverage_at_real_nn95"]
                ),
                "precision_fraction_change": (
                    shuffled["precision_at_real_nn95"]
                    - baseline["precision_at_real_nn95"]
                ),
                "diversity_change": (
                    shuffled["synthetic_pairwise_distance_mean"]
                    - baseline["synthetic_pairwise_distance_mean"]
                ),
                "centroid_distance_improvement": (
                    baseline["centroid_cosine_distance"]
                    - shuffled["centroid_cosine_distance"]
                ),
            }
        )
    return rows


def main():
    args = parse_args()
    data_root = Path(args.data_root).resolve()
    train_root = data_root / "train"
    if not train_root.is_dir():
        raise FileNotFoundError(f"ImageNette train directory not found: {train_root}")
    base_run_root = Path(args.base_run_root).resolve()
    shuffle_runs = parse_shift_runs(args.shuffle_run)
    output_dir = Path(args.output_dir)
    cache_dir = output_dir / "feature_cache"
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = discover_datasets(base_run_root, shuffle_runs)
    synsets = sorted(
        {record["synset"] for record in datasets[0]["manifest"]["prompt_records"]}
    )
    real_paths = []
    real_labels = []
    for synset in synsets:
        paths = image_paths(train_root / synset)
        real_paths.extend(paths)
        real_labels.extend([synset] * len(paths))

    extractor = DinoExtractor(args.dino_model, args.device)
    real_features, real_labels_array, _ = load_or_extract_features(
        extractor,
        real_paths,
        real_labels,
        feature_cache_path(cache_dir, "real_train"),
        args.dino_model,
        args.batch_size,
        args.resume,
    )
    real_by_class = {}
    real_radii = {}
    for synset in synsets:
        selected = real_features[real_labels_array == synset]
        real_by_class[synset] = selected
        real_radii[synset] = float(
            np.quantile(
                nearest_self_distances(selected, args.nn_block_size),
                0.95,
            )
        )

    class_rows = []
    for dataset in datasets:
        paths = image_paths(dataset["root"])
        labels = [path.parent.name for path in paths]
        expected_synsets = set(synsets)
        if set(labels) != expected_synsets:
            raise RuntimeError(
                f"Synthetic classes differ in {dataset['root']}: "
                f"{sorted(set(labels) ^ expected_synsets)}"
            )
        features, labels_array, _ = load_or_extract_features(
            extractor,
            paths,
            labels,
            feature_cache_path(cache_dir, dataset["dataset_key"]),
            args.dino_model,
            args.batch_size,
            args.resume,
        )
        for synset in synsets:
            selected = features[labels_array == synset]
            row = {
                "dataset_key": dataset["dataset_key"],
                "run_group": dataset["run_group"],
                "generation_seed": dataset["generation_seed"],
                "visual_mode": dataset["visual_mode"],
                "prompt_condition": dataset["prompt_condition"],
                "shuffle_shift": dataset["shuffle_shift"],
                "synset": synset,
            }
            row.update(
                class_metrics(
                    real_by_class[synset],
                    selected,
                    real_radii[synset],
                )
            )
            class_rows.append(row)

    del extractor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    summaries = macro_summary(class_rows)
    deltas = paired_deltas(summaries)
    class_deltas = paired_class_deltas(class_rows)
    write_csv(output_dir / "dino_metrics_per_class.csv", class_rows)
    write_csv(output_dir / "dino_metrics_summary.csv", summaries)
    write_csv(output_dir / "dino_shuffled_minus_correct.csv", deltas)
    write_csv(
        output_dir / "dino_shuffled_minus_correct_per_class.csv",
        class_deltas,
    )
    payload = {
        "data_root": str(data_root),
        "base_run_root": str(base_run_root),
        "shuffle_runs": {
            str(shift): str(root) for shift, root in sorted(shuffle_runs.items())
        },
        "dino_model": str(Path(args.dino_model).resolve()),
        "feature_definition": "L2-normalized DINOv2 CLS token",
        "distance": "cosine distance",
        "real_reference": "ImageNette train split, class conditional",
        "threshold": (
            "Per-class 95th percentile of each real image's nearest-other-real "
            "cosine distance"
        ),
        "positive_delta_convention": {
            "coverage_distance_improvement": "shuffled is closer to real data",
            "fidelity_distance_improvement": "shuffled is closer to real data",
            "coverage_fraction_change": "shuffled covers more real samples",
            "precision_fraction_change": "more shuffled samples lie near real data",
            "diversity_change": "shuffled has greater within-set spread",
            "centroid_distance_improvement": "shuffled centroid is closer to real centroid",
        },
        "macro_summary": summaries,
        "paired_deltas": deltas,
        "paired_class_deltas": class_deltas,
    }
    (output_dir / "dino_coverage_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(deltas, indent=2))


if __name__ == "__main__":
    main()
