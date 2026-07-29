import argparse
import csv
import gc
import json
import math
import os
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from common import sha256_file
from diagnose_dino_coverage import (
    DinoExtractor,
    discover_datasets,
    feature_cache_path,
    load_or_extract_features,
)
from diagnostic_common import atomic_write_json, image_paths, load_json, parse_shift_runs


EPSILON = 1e-12
PAIR_METRICS = (
    "pair_cosine_distance",
    "pair_euclidean_distance",
    "generated_source_cosine_distance",
    "generated_target_cosine_distance",
    "correct_source_cosine_distance",
    "correct_target_cosine_distance",
    "tau_from_source",
    "tau_from_correct",
    "projection_from_source",
    "projection_from_correct",
    "direction_cosine_from_source",
    "direction_cosine_from_correct",
    "residual_from_source",
    "relative_residual_from_source",
    "residual_from_correct",
    "relative_residual_from_correct",
    "shuffled_correct_feature_distance",
    "target_cosine_improvement_vs_correct",
    "source_cosine_change_vs_correct",
    "target_distance_improvement_vs_correct",
    "source_distance_change_vs_correct",
    "between_source_and_target",
    "positive_caption_pull",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Test whether same-class shuffled captions create mixup-like movement "
            "between visual cluster prototypes"
        )
    )
    parser.add_argument("--base-run-root", required=True)
    parser.add_argument(
        "--shuffle-run",
        action="append",
        default=[],
        metavar="SHIFT=RUN_ROOT",
    )
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--dino-model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--feature-cache-dir")
    parser.add_argument("--downstream-csv")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--vae-batch-size", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def normalized_dot(left, right):
    return float(np.clip(np.dot(left, right), -1.0, 1.0))


def safe_direction_cosine(left, right):
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= EPSILON:
        return 0.0
    return float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))


def projection_metrics(source, target, generated, correct):
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    generated = np.asarray(generated, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)

    pair_delta = target - source
    pair_squared = float(np.dot(pair_delta, pair_delta))
    if pair_squared <= EPSILON:
        raise ValueError("Source and target cluster features are indistinguishable")
    pair_distance = math.sqrt(pair_squared)

    generated_delta = generated - source
    correct_delta = correct - source
    caption_delta = generated - correct

    projection_source = float(np.dot(generated_delta, pair_delta))
    projection_correct = float(np.dot(caption_delta, pair_delta))
    tau_source = projection_source / pair_squared
    tau_correct = projection_correct / pair_squared
    residual_source = float(
        np.linalg.norm(generated_delta - tau_source * pair_delta)
    )
    residual_correct = float(
        np.linalg.norm(caption_delta - tau_correct * pair_delta)
    )

    generated_target_distance = float(np.linalg.norm(generated - target))
    correct_target_distance = float(np.linalg.norm(correct - target))
    generated_source_distance = float(np.linalg.norm(generated - source))
    correct_source_distance = float(np.linalg.norm(correct - source))

    return {
        "pair_cosine_distance": 1.0 - normalized_dot(source, target),
        "pair_euclidean_distance": pair_distance,
        "generated_source_cosine_distance": 1.0
        - normalized_dot(generated, source),
        "generated_target_cosine_distance": 1.0
        - normalized_dot(generated, target),
        "correct_source_cosine_distance": 1.0 - normalized_dot(correct, source),
        "correct_target_cosine_distance": 1.0 - normalized_dot(correct, target),
        "tau_from_source": tau_source,
        "tau_from_correct": tau_correct,
        "projection_from_source": projection_source,
        "projection_from_correct": projection_correct,
        "direction_cosine_from_source": safe_direction_cosine(
            generated_delta, pair_delta
        ),
        "direction_cosine_from_correct": safe_direction_cosine(
            caption_delta, pair_delta
        ),
        "residual_from_source": residual_source,
        "relative_residual_from_source": residual_source / pair_distance,
        "residual_from_correct": residual_correct,
        "relative_residual_from_correct": residual_correct / pair_distance,
        "shuffled_correct_feature_distance": float(np.linalg.norm(caption_delta)),
        "target_cosine_improvement_vs_correct": (
            normalized_dot(generated, target) - normalized_dot(correct, target)
        ),
        "source_cosine_change_vs_correct": (
            normalized_dot(generated, source) - normalized_dot(correct, source)
        ),
        "target_distance_improvement_vs_correct": (
            correct_target_distance - generated_target_distance
        ),
        "source_distance_change_vs_correct": (
            generated_source_distance - correct_source_distance
        ),
        "between_source_and_target": float(0.0 <= tau_source <= 1.0),
        "positive_caption_pull": float(projection_correct > 0.0),
    }


def prototype_decode_manifest(prototype_path, base_model):
    return {
        "schema_version": 1,
        "prototype_path": str(Path(prototype_path).resolve()),
        "prototype_sha256": sha256_file(prototype_path),
        "base_model": str(Path(base_model).resolve()),
        "decode": "vae.decode(latent / vae.config.scaling_factor)",
    }


def decode_prototypes(
    prototype_path,
    base_model,
    output_root,
    device,
    batch_size,
    resume,
):
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "decode_manifest.json"
    expected_manifest = prototype_decode_manifest(prototype_path, base_model)
    if manifest_path.is_file():
        current = load_json(manifest_path)
        if current != expected_manifest:
            raise RuntimeError(
                f"Decoded prototype configuration changed: {manifest_path}"
            )
        if not resume:
            raise RuntimeError(
                f"Decoded prototypes already exist; pass --resume: {output_root}"
            )
    elif any(output_root.iterdir()):
        raise RuntimeError(
            f"Non-empty prototype directory has no manifest: {output_root}"
        )
    else:
        atomic_write_json(manifest_path, expected_manifest)

    prototypes = load_json(prototype_path)
    expected_paths = []
    pending = []
    for synset, values in prototypes.items():
        for index, value in enumerate(values):
            path = output_root / synset / f"prototype_{index:05d}.png"
            expected_paths.append(path)
            if not path.is_file():
                pending.append((path, value))

    if pending:
        from diffusers import AutoencoderKL

        torch_device = torch.device(device)
        dtype = torch.float16 if torch_device.type == "cuda" else torch.float32
        vae = AutoencoderKL.from_pretrained(
            base_model,
            subfolder="vae",
            torch_dtype=dtype,
            local_files_only=True,
        ).to(torch_device)
        vae.eval()
        scaling_factor = float(vae.config.scaling_factor)
        for start in range(0, len(pending), batch_size):
            selected = pending[start : start + batch_size]
            latents = torch.tensor(
                [item[1] for item in selected],
                dtype=dtype,
                device=torch_device,
            )
            with torch.inference_mode():
                decoded = vae.decode(latents / scaling_factor).sample
            decoded = (
                decoded.float().div(2.0).add(0.5).clamp(0.0, 1.0).cpu()
            )
            for (path, _), image_tensor in zip(selected, decoded):
                path.parent.mkdir(parents=True, exist_ok=True)
                array = (
                    image_tensor.permute(1, 2, 0).mul(255).round().byte().numpy()
                )
                Image.fromarray(array).save(path)
            print(
                f"Decoded prototypes: {min(start + batch_size, len(pending))}/"
                f"{len(pending)}"
            )
        del vae
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    missing = [path for path in expected_paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"Missing decoded prototypes: {missing[:3]}")
    return prototypes, expected_paths


def feature_index(paths, features):
    return {
        str(Path(path).resolve()): feature
        for path, feature in zip(paths, features)
    }


def record_image_path(dataset_root, record):
    return (
        Path(dataset_root)
        / record["synset"]
        / f"image_{int(record['image_index']):05d}.png"
    )


def dataset_feature_index(
    extractor,
    dataset,
    cache_dir,
    dino_model,
    batch_size,
    resume,
):
    paths = image_paths(dataset["root"])
    labels = [path.parent.name for path in paths]
    features, _, cached_paths = load_or_extract_features(
        extractor,
        paths,
        labels,
        feature_cache_path(cache_dir, dataset["dataset_key"]),
        dino_model,
        batch_size,
        resume,
    )
    return feature_index(cached_paths, features)


def aggregate_rows(rows, group_fields):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in group_fields)].append(row)
    output = []
    for key, selected in grouped.items():
        result = dict(zip(group_fields, key))
        result["observations"] = len(selected)
        for field in PAIR_METRICS:
            result[field] = statistics.fmean(
                float(row[field]) for row in selected
            )
        output.append(result)
    return sorted(output, key=lambda row: tuple(str(row[field]) for field in group_fields))


def rankdata(values):
    values = list(values)
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = 0.5 * (start + 1 + end)
        for position in range(start, end):
            ranks[order[position]] = rank
        start = end
    return ranks


def pearson(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if len(left) < 2 or np.std(left) <= EPSILON or np.std(right) <= EPSILON:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def spearman(left, right):
    return pearson(rankdata(left), rankdata(right))


def quadratic_fit(x_values, y_values):
    x_values = np.asarray(x_values, dtype=np.float64)
    y_values = np.asarray(y_values, dtype=np.float64)
    if len(x_values) < 4 or np.std(x_values) <= EPSILON:
        return None
    normalized = (x_values - x_values.mean()) / x_values.std()
    design = np.column_stack(
        [np.ones(len(normalized)), normalized, normalized**2]
    )
    coefficients, _, _, _ = np.linalg.lstsq(design, y_values, rcond=None)
    predictions = design @ coefficients
    residual_sum = float(np.sum((y_values - predictions) ** 2))
    total_sum = float(np.sum((y_values - y_values.mean()) ** 2))
    return {
        "intercept": float(coefficients[0]),
        "linear": float(coefficients[1]),
        "quadratic": float(coefficients[2]),
        "r_squared": 1.0 - residual_sum / max(total_sum, EPSILON),
        "x_standardization_mean": float(x_values.mean()),
        "x_standardization_std": float(x_values.std()),
    }


def join_downstream(class_rows, downstream_csv):
    downstream_rows = read_csv(downstream_csv)
    index = {
        (
            int(float(row["generation_seed"])),
            row["visual_mode"],
            int(float(row["shuffle_shift"])),
            row["synset"],
        ): float(row["downstream_gain"])
        for row in downstream_rows
    }
    joined = []
    for row in class_rows:
        key = (
            int(row["generation_seed"]),
            "prototype",
            int(row["shuffle_shift"]),
            row["synset"],
        )
        if key not in index:
            raise KeyError(f"Missing downstream result for {key}")
        joined.append({**row, "downstream_gain": index[key]})
    return joined


def relationship_summary(pair_rows, downstream_rows):
    summary = {
        "h1_caption_pull": {
            "mean_projection_from_correct": statistics.fmean(
                row["projection_from_correct"] for row in pair_rows
            ),
            "mean_direction_cosine_from_correct": statistics.fmean(
                row["direction_cosine_from_correct"] for row in pair_rows
            ),
            "positive_caption_pull_fraction": statistics.fmean(
                row["positive_caption_pull"] for row in pair_rows
            ),
            "mean_target_cosine_improvement_vs_correct": statistics.fmean(
                row["target_cosine_improvement_vs_correct"] for row in pair_rows
            ),
        },
        "h2_distance_response": {},
    }
    distance = [row["pair_cosine_distance"] for row in pair_rows]
    for field in (
        "projection_from_correct",
        "direction_cosine_from_correct",
        "shuffled_correct_feature_distance",
        "target_cosine_improvement_vs_correct",
    ):
        values = [row[field] for row in pair_rows]
        summary["h2_distance_response"][field] = {
            "pearson": pearson(distance, values),
            "spearman": spearman(distance, values),
        }

    if downstream_rows:
        gains = [row["downstream_gain"] for row in downstream_rows]
        h3 = {"n": len(downstream_rows), "metrics": {}}
        for field in (
            "pair_cosine_distance",
            "tau_from_correct",
            "projection_from_correct",
            "direction_cosine_from_correct",
            "relative_residual_from_correct",
            "target_cosine_improvement_vs_correct",
        ):
            values = [row[field] for row in downstream_rows]
            h3["metrics"][field] = {
                "pearson": pearson(values, gains),
                "spearman": spearman(values, gains),
            }
        h3["distance_quadratic_fit"] = quadratic_fit(
            [row["pair_cosine_distance"] for row in downstream_rows],
            gains,
        )
        h3["tau_quadratic_fit"] = quadratic_fit(
            [row["tau_from_correct"] for row in downstream_rows],
            gains,
        )
        summary["h3_downstream_utility"] = h3
    return summary


def plot_shift_summary(shift_rows, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as pyplot

    shifts = [int(row["shuffle_shift"]) for row in shift_rows]
    panels = (
        ("pair_cosine_distance", "Actual prototype-pair cosine distance"),
        ("projection_from_correct", "Caption-pull projection"),
        ("direction_cosine_from_correct", "Caption-pull direction cosine"),
        ("positive_caption_pull", "Positive caption-pull fraction"),
    )
    figure, axes = pyplot.subplots(2, 2, figsize=(12, 8))
    for axis, (field, title) in zip(axes.flat, panels):
        axis.plot(
            shifts,
            [float(row[field]) for row in shift_rows],
            marker="o",
        )
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_xlabel("Cyclic permutation ID (not a distance)")
        axis.set_ylabel(title)
        axis.grid(alpha=0.25)
    figure.suptitle("Cross-modal recombination geometry by permutation")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    pyplot.close(figure)


def plot_pair_geometry(pair_rows, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as pyplot

    figure, axes = pyplot.subplots(1, 3, figsize=(16, 5))
    panels = (
        ("projection_from_correct", "Caption-pull projection"),
        ("direction_cosine_from_correct", "Caption-pull direction cosine"),
        (
            "target_cosine_improvement_vs_correct",
            "Target similarity improvement",
        ),
    )
    shifts = sorted({int(row["shuffle_shift"]) for row in pair_rows})
    for axis, (field, title) in zip(axes, panels):
        for shift in shifts:
            selected = [
                row for row in pair_rows if int(row["shuffle_shift"]) == shift
            ]
            axis.scatter(
                [row["pair_cosine_distance"] for row in selected],
                [row[field] for row in selected],
                alpha=0.55,
                label=f"shift {shift}",
            )
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_xlabel("Actual source-target prototype cosine distance")
        axis.set_ylabel(title)
        axis.grid(alpha=0.25)
    axes[0].legend()
    figure.suptitle("Does shuffled text pull outputs toward its source cluster?")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    pyplot.close(figure)


def plot_downstream(rows, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as pyplot

    panels = (
        ("pair_cosine_distance", "Mean prototype-pair distance"),
        ("projection_from_correct", "Mean caption-pull projection"),
        ("direction_cosine_from_correct", "Mean caption-pull direction cosine"),
        ("relative_residual_from_correct", "Mean off-axis residual"),
    )
    figure, axes = pyplot.subplots(2, 2, figsize=(12, 9))
    shifts = sorted({int(row["shuffle_shift"]) for row in rows})
    for axis, (field, title) in zip(axes.flat, panels):
        for shift in shifts:
            selected = [row for row in rows if int(row["shuffle_shift"]) == shift]
            axis.scatter(
                [row[field] for row in selected],
                [row["downstream_gain"] for row in selected],
                alpha=0.7,
                label=f"shift {shift}",
            )
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_xlabel(title)
        axis.set_ylabel("Shuffled - correct per-class accuracy")
        axis.grid(alpha=0.25)
    axes.flat[0].legend()
    figure.suptitle("Recombination geometry versus downstream class gain")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    pyplot.close(figure)


def main():
    args = parse_args()
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

    prototype_paths = {
        str(Path(dataset["manifest"]["prototype_path"]).resolve())
        for dataset in [*correct_datasets.values(), *shuffled_datasets]
    }
    if len(prototype_paths) != 1:
        raise RuntimeError(f"Runs use different prototype files: {prototype_paths}")
    prototype_path = Path(prototype_paths.pop())
    prototypes, decoded_paths = decode_prototypes(
        prototype_path,
        args.base_model,
        output_dir / "decoded_prototypes",
        args.device,
        args.vae_batch_size,
        args.resume,
    )

    extractor = DinoExtractor(args.dino_model, args.device)
    prototype_labels = [path.parent.name for path in decoded_paths]
    prototype_features, _, prototype_cached_paths = load_or_extract_features(
        extractor,
        decoded_paths,
        prototype_labels,
        feature_cache_path(cache_dir, "decoded_cluster_prototypes"),
        args.dino_model,
        args.batch_size,
        args.resume,
    )
    prototype_path_features = feature_index(
        prototype_cached_paths, prototype_features
    )
    prototype_index = {}
    for synset, values in prototypes.items():
        for index in range(len(values)):
            path = (
                output_dir
                / "decoded_prototypes"
                / synset
                / f"prototype_{index:05d}.png"
            )
            prototype_index[(synset, index)] = prototype_path_features[
                str(path.resolve())
            ]

    dataset_features = {}
    for dataset in [*correct_datasets.values(), *shuffled_datasets]:
        dataset_features[dataset["dataset_key"]] = dataset_feature_index(
            extractor,
            dataset,
            cache_dir,
            args.dino_model,
            args.batch_size,
            args.resume,
        )

    pair_rows = []
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
            source_index = int(record["prototype_index"])
            target_index = int(record["prompt_source_index"])
            count = len(prototypes[synset])
            if target_index != (source_index + shift) % count:
                raise RuntimeError(
                    f"Unexpected permutation for {synset}/{image_index}: "
                    f"{source_index}->{target_index}, shift={shift}"
                )
            correct_record = correct_records[(synset, image_index)]
            if int(correct_record["prototype_index"]) != source_index:
                raise RuntimeError("Correct/shuffled prototype index mismatch")
            if int(correct_record["prompt_source_index"]) != source_index:
                raise RuntimeError("Correct DCS is not matched to its prototype")

            shuffled_path = record_image_path(shuffled["root"], record).resolve()
            correct_path = record_image_path(correct["root"], correct_record).resolve()
            metrics = projection_metrics(
                prototype_index[(synset, source_index)],
                prototype_index[(synset, target_index)],
                shuffled_features[str(shuffled_path)],
                correct_features[str(correct_path)],
            )
            pair_rows.append(
                {
                    "generation_seed": generation_seed,
                    "shuffle_shift": shift,
                    "synset": synset,
                    "image_index": image_index,
                    "source_prototype_index": source_index,
                    "caption_source_prototype_index": target_index,
                    "caption": record["prompt"],
                    "shuffled_image": str(shuffled_path),
                    "correct_image": str(correct_path),
                    **metrics,
                }
            )

    del extractor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    class_rows = aggregate_rows(
        pair_rows,
        ("generation_seed", "shuffle_shift", "synset"),
    )
    generation_shift_rows = aggregate_rows(
        pair_rows,
        ("generation_seed", "shuffle_shift"),
    )
    shift_rows = aggregate_rows(pair_rows, ("shuffle_shift",))
    downstream_rows = []
    if args.downstream_csv:
        downstream_rows = join_downstream(class_rows, args.downstream_csv)

    summary = {
        "schema_version": 1,
        "base_run_root": str(Path(args.base_run_root).resolve()),
        "shuffle_runs": {
            str(key): str(value) for key, value in sorted(shuffle_runs.items())
        },
        "prototype_path": str(prototype_path),
        "base_model": str(Path(args.base_model).resolve()),
        "dino_model": str(Path(args.dino_model).resolve()),
        "feature_definition": "L2-normalized DINOv2 CLS token",
        "geometry": {
            "tau_from_source": (
                "Projection coefficient of generated-source onto target-source"
            ),
            "tau_from_correct": (
                "Projection coefficient of shuffled-correct output displacement "
                "onto target-source; isolates the caption reassignment effect"
            ),
            "between_source_and_target": "1 when 0 <= tau_from_source <= 1",
            "positive_caption_pull": "1 when projection_from_correct > 0",
            "relative_residual": (
                "Orthogonal residual divided by source-target prototype distance"
            ),
        },
        "caveats": [
            "Cyclic shift IDs are permutation identifiers, not mismatch distances.",
            (
                "A DCS caption is attributed to its source cluster by construction, "
                "but the caption may not faithfully describe every visual property "
                "of that cluster."
            ),
            "DINO geometry is a diagnostic representation, not a proven downstream proxy.",
            "The same correct-DCS classifier result is reused across shuffle conditions.",
            "Quadratic fits are exploratory and do not establish an inverted-U mechanism.",
        ],
        "relationships": relationship_summary(pair_rows, downstream_rows),
        "shift_summary": shift_rows,
    }

    write_csv(output_dir / "recombination_per_image.csv", pair_rows)
    write_csv(output_dir / "recombination_per_class.csv", class_rows)
    write_csv(
        output_dir / "recombination_per_generation_shift.csv",
        generation_shift_rows,
    )
    write_csv(output_dir / "recombination_per_shift.csv", shift_rows)
    if downstream_rows:
        write_csv(
            output_dir / "recombination_vs_downstream_per_class.csv",
            downstream_rows,
        )
    atomic_write_json(output_dir / "recombination_summary.json", summary)
    plot_shift_summary(
        shift_rows,
        output_dir / "recombination_by_permutation.png",
    )
    plot_pair_geometry(
        pair_rows,
        output_dir / "recombination_pair_geometry.png",
    )
    if downstream_rows:
        plot_downstream(
            downstream_rows,
            output_dir / "recombination_vs_downstream.png",
        )
    print(json.dumps(summary["relationships"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
