"""Select real or Diffusion candidates using local gradient matching."""

import argparse
import csv
import itertools
import json
import math
import os
import pickle
import random
import shutil
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy import stats
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


PROGRAM_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.join(PROGRAM_DIR, "test")
if TEST_DIR not in sys.path:
    sys.path.insert(0, TEST_DIR)

import resnet_ap as RNAP
from data import transform_imagenet
from reconstruct_representatives import _selected_classes


def _read_pickle(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Required clustering artifact was not found: {path}")
    with open(path, "rb") as file:
        return pickle.load(file)


def _write_json(path, payload):
    temporary = f"{path}.tmp"
    with open(temporary, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")
    os.replace(temporary, path)


def _write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _to_flat_numpy(items):
    if isinstance(items, np.ndarray) and items.ndim == 2:
        return items.astype(np.float32, copy=False)
    return np.stack([
        np.asarray(item, dtype=np.float32).reshape(-1) for item in items
    ])


def _load_cluster_data(args):
    feature_path = os.path.join(
        args.program_path, "results", "clusterfile", args.spec,
        "original_features_cache.pkl_0",
    )
    center_path = os.path.join(
        args.program_path, "results", "clusterfile", args.spec,
        f"{args.ipc}_n_{args.n_neighbors}_s_{args.min_cluster_size}_saved_clusters_0.pkl",
    )
    feature_cache = _read_pickle(feature_path)
    centers = _read_pickle(center_path)
    return feature_cache, centers, feature_path, center_path


def _candidate_path(root, class_id, index):
    path = os.path.join(root, class_id, f"{index}.png")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Candidate image was not found: {path}")
    return os.path.abspath(path)


def _chunked_squared_distances(features, center, scale, chunk_size=64):
    distances = np.empty(len(features), dtype=np.float64)
    for start in range(0, len(features), chunk_size):
        batch = features[start:start + chunk_size]
        difference = (batch - center) / scale
        distances[start:start + len(batch)] = np.einsum(
            "ij,ij->i", difference, difference, dtype=np.float64
        )
    return distances


def _neighbor_weights(distances, weighting):
    distances = np.asarray(distances, dtype=np.float64)
    if weighting == "uniform":
        weights = np.full(len(distances), 1.0 / len(distances), dtype=np.float64)
    elif weighting == "distance":
        temperature = max(float(distances[-1]), np.finfo(np.float64).eps)
        logits = -distances / temperature
        logits -= logits.max()
        weights = np.exp(logits)
        weights /= weights.sum()
    else:
        raise ValueError(f"Unsupported neighbor weighting: {weighting}")
    return weights.astype(np.float32)


def build_neighbor_manifest(args, class_ids, feature_cache, centers):
    manifest = {
        "metric": args.neighbor_metric,
        "neighbor_count": args.neighbor_count,
        "weighting": args.neighbor_weighting,
        "classes": {},
    }
    all_neighbor_paths = set()
    overlap_rows = []

    for local_label, class_id in enumerate(class_ids):
        raw_features = feature_cache["features"].get(local_label)
        paths = feature_cache["paths"].get(local_label)
        class_centers = centers.get(local_label)
        if raw_features is None or paths is None or class_centers is None:
            raise KeyError(f"Missing clustering cache for {class_id} (label {local_label}).")
        if len(class_centers) != args.ipc:
            raise ValueError(
                f"Expected {args.ipc} centers for {class_id}, found {len(class_centers)}."
            )
        if len(paths) <= args.neighbor_count:
            raise ValueError(
                f"Class {class_id} has {len(paths)} samples, insufficient for "
                f"{args.neighbor_count} leave-one-out neighbors."
            )

        features = _to_flat_numpy(raw_features)
        if len(features) != len(paths):
            raise ValueError(
                f"Feature/path mismatch for {class_id}: {len(features)} vs {len(paths)}."
            )
        if args.neighbor_metric == "standardized_l2":
            scale = features.std(axis=0, dtype=np.float32)
            scale[scale < args.standardization_epsilon] = 1.0
        elif args.neighbor_metric == "raw_l2":
            scale = np.ones(features.shape[1], dtype=np.float32)
        else:
            raise ValueError(f"Unsupported neighbor metric: {args.neighbor_metric}")

        class_entries = []
        neighbor_sets = []
        for index, center in enumerate(class_centers):
            center = np.asarray(center, dtype=np.float32).reshape(-1)
            if center.shape[0] != features.shape[1]:
                raise ValueError(
                    f"Center dimension mismatch for {class_id}/{index}: "
                    f"{center.shape[0]} vs {features.shape[1]}."
                )
            distances = _chunked_squared_distances(features, center, scale)
            nearest_count = args.neighbor_count + 1
            candidates = np.argpartition(distances, nearest_count - 1)[:nearest_count]
            ordered = candidates[np.lexsort((candidates, distances[candidates]))]

            # Every saved CoDA center is an original sample. The closest source is
            # excluded so the local target cannot trivially contain the real candidate.
            representative_source_index = int(ordered[0])
            if distances[representative_source_index] > args.representative_distance_tolerance:
                raise ValueError(
                    f"Saved center {class_id}/{index} does not match an original source: "
                    f"nearest standardized squared distance is "
                    f"{distances[representative_source_index]:.6g}."
                )
            neighbor_indices = ordered[1:nearest_count]
            neighbor_distances = distances[neighbor_indices]
            weights = _neighbor_weights(neighbor_distances, args.neighbor_weighting)
            source_paths = [os.path.abspath(paths[int(i)]) for i in neighbor_indices]
            missing = [path for path in source_paths if not os.path.isfile(path)]
            if missing:
                raise FileNotFoundError(f"Local-neighbor source image was not found: {missing[0]}")

            all_neighbor_paths.update(source_paths)
            neighbor_sets.append(set(source_paths))
            class_entries.append({
                "pair_index": index,
                "representative_source_path": os.path.abspath(paths[representative_source_index]),
                "representative_source_distance": float(distances[representative_source_index]),
                "neighbor_paths": source_paths,
                "neighbor_distances": [float(value) for value in neighbor_distances],
                "neighbor_weights": [float(value) for value in weights],
                "effective_sample_size": float(1.0 / np.square(weights).sum()),
            })

        pair_overlaps = []
        for first, second in itertools.combinations(range(args.ipc), 2):
            intersection = len(neighbor_sets[first] & neighbor_sets[second])
            union = len(neighbor_sets[first] | neighbor_sets[second])
            jaccard = intersection / union if union else 0.0
            pair_overlaps.append(jaccard)
            overlap_rows.append({
                "spec": args.spec,
                "class_id": class_id,
                "first_pair_index": first,
                "second_pair_index": second,
                "intersection": intersection,
                "jaccard": jaccard,
            })
        manifest["classes"][class_id] = {
            "local_label": local_label,
            "sample_count": len(paths),
            "mean_pairwise_neighbor_jaccard": float(np.mean(pair_overlaps)),
            "max_pairwise_neighbor_jaccard": float(np.max(pair_overlaps)),
            "pairs": class_entries,
        }

    manifest["unique_neighbor_image_count"] = len(all_neighbor_paths)
    return manifest, sorted(all_neighbor_paths), overlap_rows


class PathDataset(Dataset):
    def __init__(self, paths, transform):
        self.paths = list(paths)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)
        return tensor, path


def _seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def _build_model(seed, args, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = RNAP.ResNetAP(
        "imagenet", args.depth, args.nclass, width=args.width,
        norm_type="instance", size=args.image_size, nch=3,
    )
    return model.to(device).eval()


def _fc_gradients(model, features, labels):
    features = features.float()
    logits = model.fc(features)
    probabilities = logits.float().softmax(dim=1)
    residual = probabilities
    residual[torch.arange(len(labels), device=labels.device), labels] -= 1.0
    weight_gradient = residual.unsqueeze(2) * features.float().unsqueeze(1)
    return torch.cat((weight_gradient.flatten(1), residual), dim=1)


def compute_gradient_map(paths, labels_by_path, model, transform, args, device, run_seed):
    generator = torch.Generator()
    generator.manual_seed(run_seed)
    dataset = PathDataset(paths, transform)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0, worker_init_fn=_seed_worker,
        generator=generator,
    )
    gradients = {}
    autocast_enabled = device.type == "cuda" and args.amp
    with torch.inference_mode():
        for images, batch_paths in tqdm(loader, desc="FC gradients", leave=False):
            images = images.to(device, non_blocking=True)
            labels = torch.tensor(
                [labels_by_path[path] for path in batch_paths],
                dtype=torch.long, device=device,
            )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=autocast_enabled,
            ):
                features = model.get_feature(images, idx_from=5, idx_to=5)[0]
            batch_gradients = _fc_gradients(model, features, labels).cpu().numpy()
            gradients.update(zip(batch_paths, batch_gradients))
    return gradients


def _cosine(first, second, epsilon=1e-12):
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    return float(np.dot(first, second) / max(denominator, epsilon))


def _combination_matrix(ipc):
    values = np.arange(2 ** ipc, dtype=np.uint16)[:, None]
    bits = np.arange(ipc, dtype=np.uint16)[None, :]
    return ((values >> bits) & 1).astype(np.float64)


def _relative_combination_losses(real, diffusion, target, combinations):
    # All vectors are sums rather than means; the common IPC factor cancels.
    base = real.sum(axis=0) - target.sum(axis=0)
    deltas = diffusion - real
    target_norm = max(float(np.square(target.sum(axis=0)).sum()), 1e-12)
    linear = 2.0 * deltas.dot(base)
    gram = deltas.dot(deltas.T)
    losses = (
        float(np.dot(base, base))
        + combinations.dot(linear)
        + np.einsum("bi,ij,bj->b", combinations, gram, combinations)
    ) / target_norm
    return np.maximum(losses, 0.0)


def _load_accuracy(path):
    if path is None:
        return None
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Downstream accuracy file was not found: {path}")
    with open(path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    return {
        row["class_id"]: float(row["mean"])
        for row in payload["class_summary"]
    }


def _copy_selection(output_dir, class_ids, candidate_paths, choices):
    if os.path.exists(output_dir):
        raise FileExistsError(f"Refusing to overwrite selected dataset: {output_dir}")
    os.makedirs(output_dir)
    records = []
    for local_label, class_id in enumerate(class_ids):
        class_dir = os.path.join(output_dir, class_id)
        os.makedirs(class_dir)
        for pair_index, source in enumerate(choices[class_id]):
            source_path = candidate_paths[class_id][pair_index][source]
            destination = os.path.join(class_dir, f"{pair_index}.png")
            shutil.copy2(source_path, destination)
            records.append({
                "local_label": local_label,
                "class_id": class_id,
                "pair_index": pair_index,
                "selected_source": source,
                "source_path": source_path,
                "destination_path": os.path.abspath(destination),
            })
    return records


def _sign_accuracy(predicted, observed):
    valid = [(p, o) for p, o in zip(predicted, observed) if o != 0.0 and p != 0.0]
    if not valid:
        return None, 0
    correct = sum(np.sign(p) == np.sign(o) for p, o in valid)
    return float(correct / len(valid)), len(valid)


def _finite_correlation(function, x, y):
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return {"statistic": None, "pvalue": None}
    result = function(x, y)
    statistic = float(result.statistic) if np.isfinite(result.statistic) else None
    pvalue = float(result.pvalue) if np.isfinite(result.pvalue) else None
    return {"statistic": statistic, "pvalue": pvalue}


def _plot_outputs(output_dir, class_rows, pair_rows, class_ids):
    pair_summary = defaultdict(list)
    for row in pair_rows:
        pair_summary[(row["class_id"], row["pair_index"])].append(row["delta_g"])
    heatmap = np.zeros((len(class_ids), max(row["pair_index"] for row in pair_rows) + 1))
    for class_index, class_id in enumerate(class_ids):
        for pair_index in range(heatmap.shape[1]):
            heatmap[class_index, pair_index] = np.mean(pair_summary[(class_id, pair_index)])

    figure, axis = plt.subplots(figsize=(10, 5))
    limit = max(float(np.max(np.abs(heatmap))), 1e-8)
    image = axis.imshow(
        heatmap, cmap="coolwarm", aspect="auto", vmin=-limit, vmax=limit
    )
    axis.set_xlabel("Cluster pair index")
    axis.set_ylabel("Class")
    axis.set_yticks(range(len(class_ids)), class_ids)
    axis.set_title("Diffusion minus real local gradient cosine")
    figure.colorbar(image, ax=axis, label="Mean delta G")
    figure.tight_layout()
    figure.savefig(os.path.join(output_dir, "pair_delta_g_heatmap.png"), dpi=180)
    plt.close(figure)

    if all(row.get("accuracy_gain_diffusion_vs_real") is not None for row in class_rows):
        x = [row["set_score_diffusion_vs_real"] for row in class_rows]
        y = [row["accuracy_gain_diffusion_vs_real"] for row in class_rows]
        figure, axis = plt.subplots(figsize=(7, 6))
        axis.axhline(0.0, color="gray", linewidth=1)
        axis.axvline(0.0, color="gray", linewidth=1)
        axis.scatter(x, y)
        for row in class_rows:
            axis.annotate(row["class_name"], (row["set_score_diffusion_vs_real"], row["accuracy_gain_diffusion_vs_real"]), fontsize=8)
        axis.set_xlabel("GM set score: Diffusion over real")
        axis.set_ylabel("Accuracy gain: Diffusion over real")
        axis.set_title(f"{class_rows[0]['spec']}: gradient signal vs observed gain")
        figure.tight_layout()
        figure.savefig(os.path.join(output_dir, "gradient_score_accuracy_relationship.png"), dpi=180)
        plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program_path", default=".")
    parser.add_argument("--spec", required=True)
    parser.add_argument("--real_dir", required=True)
    parser.add_argument("--diffusion_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--real_accuracy")
    parser.add_argument("--diffusion_accuracy")
    parser.add_argument("--nclass", type=int, default=10)
    parser.add_argument("--phase", type=int, default=0)
    parser.add_argument("--ipc", type=int, default=10)
    parser.add_argument("--n_neighbors", type=int, default=85)
    parser.add_argument("--min_cluster_size", type=int, default=55)
    parser.add_argument("--neighbor_count", type=int, default=32)
    parser.add_argument(
        "--neighbor_metric", choices=("standardized_l2", "raw_l2"),
        default="standardized_l2",
    )
    parser.add_argument(
        "--neighbor_weighting", choices=("uniform", "distance"), default="uniform"
    )
    parser.add_argument("--standardization_epsilon", type=float, default=1e-6)
    parser.add_argument("--representative_distance_tolerance", type=float, default=1e-4)
    parser.add_argument("--model_seeds", default="0,1,2,3")
    parser.add_argument("--augmentations", type=int, default=4)
    parser.add_argument("--random_selection_seed", type=int, default=20260717)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument("--width", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if os.path.exists(args.output_dir):
        raise FileExistsError(f"Refusing to overwrite gradient-selection run: {args.output_dir}")
    if args.ipc > 15:
        parser.error("Exact paired enumeration currently supports IPC <= 15.")
    if args.neighbor_count < 1 or args.augmentations < 1 or args.batch_size < 1:
        parser.error("Neighbor count, augmentations, and batch size must be positive.")
    model_seeds = [int(value) for value in args.model_seeds.split(",") if value.strip()]
    if not model_seeds:
        parser.error("--model_seeds must contain at least one integer.")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA gradient selection requested, but CUDA is unavailable.")

    started = time.perf_counter()
    os.makedirs(args.output_dir)
    class_ids = _selected_classes(args.spec, args.nclass, args.phase)
    feature_cache, centers, feature_path, center_path = _load_cluster_data(args)
    neighbor_manifest, neighbor_paths, overlap_rows = build_neighbor_manifest(
        args, class_ids, feature_cache, centers
    )
    _write_json(os.path.join(args.output_dir, "neighbor_manifest.json"), neighbor_manifest)
    _write_csv(os.path.join(args.output_dir, "neighbor_overlap.csv"), overlap_rows)
    del feature_cache, centers

    candidate_paths = {}
    labels_by_path = {}
    for local_label, class_id in enumerate(class_ids):
        candidate_paths[class_id] = []
        for pair_index in range(args.ipc):
            real_path = _candidate_path(args.real_dir, class_id, pair_index)
            diffusion_path = _candidate_path(args.diffusion_dir, class_id, pair_index)
            candidate_paths[class_id].append({"real": real_path, "diffusion": diffusion_path})
            labels_by_path[real_path] = local_label
            labels_by_path[diffusion_path] = local_label
        for pair in neighbor_manifest["classes"][class_id]["pairs"]:
            for path in pair["neighbor_paths"]:
                labels_by_path[path] = local_label

    all_paths = sorted(labels_by_path)
    combinations = _combination_matrix(args.ipc)
    accumulated_losses = {
        class_id: np.zeros(len(combinations), dtype=np.float64) for class_id in class_ids
    }
    all_real_losses = defaultdict(list)
    all_diffusion_losses = defaultdict(list)
    pair_rows = []
    run_best_masks = defaultdict(list)
    total_runs = len(model_seeds) * args.augmentations

    transform, _ = transform_imagenet(
        augment=True, size=args.image_size, from_tensor=False, normalize=True, rrc=True
    )
    run_index = 0
    for model_seed in model_seeds:
        model = _build_model(model_seed, args, device)
        for augmentation_index in range(args.augmentations):
            run_seed = args.random_selection_seed + model_seed * 10000 + augmentation_index
            random.seed(run_seed)
            np.random.seed(run_seed)
            torch.manual_seed(run_seed)
            gradient_map = compute_gradient_map(
                all_paths, labels_by_path, model, transform, args, device, run_seed
            )

            for class_id in class_ids:
                local_gradients = []
                real_gradients = []
                diffusion_gradients = []
                for pair_index, pair in enumerate(
                    neighbor_manifest["classes"][class_id]["pairs"]
                ):
                    neighbor_gradient = np.average(
                        np.stack([gradient_map[path] for path in pair["neighbor_paths"]]),
                        axis=0, weights=np.asarray(pair["neighbor_weights"], dtype=np.float64),
                    ).astype(np.float32)
                    real_gradient = gradient_map[candidate_paths[class_id][pair_index]["real"]]
                    diffusion_gradient = gradient_map[
                        candidate_paths[class_id][pair_index]["diffusion"]
                    ]
                    local_gradients.append(neighbor_gradient)
                    real_gradients.append(real_gradient)
                    diffusion_gradients.append(diffusion_gradient)
                    real_g = _cosine(real_gradient, neighbor_gradient)
                    diffusion_g = _cosine(diffusion_gradient, neighbor_gradient)
                    pair_rows.append({
                        "spec": args.spec,
                        "class_id": class_id,
                        "pair_index": pair_index,
                        "model_seed": model_seed,
                        "augmentation_index": augmentation_index,
                        "real_g": real_g,
                        "diffusion_g": diffusion_g,
                        "delta_g": diffusion_g - real_g,
                    })

                local_gradients = np.stack(local_gradients)
                real_gradients = np.stack(real_gradients)
                diffusion_gradients = np.stack(diffusion_gradients)
                losses = _relative_combination_losses(
                    real_gradients, diffusion_gradients, local_gradients, combinations
                )
                accumulated_losses[class_id] += losses
                all_real_losses[class_id].append(float(losses[0]))
                all_diffusion_losses[class_id].append(float(losses[-1]))
                run_best_masks[class_id].append(int(np.argmin(losses)))

            del gradient_map
            run_index += 1
            print(f"Completed gradient run {run_index}/{total_runs}")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    real_accuracy = _load_accuracy(args.real_accuracy)
    diffusion_accuracy = _load_accuracy(args.diffusion_accuracy)
    class_rows = []
    gm_choices = {}
    random_choices = {}
    rng = np.random.default_rng(args.random_selection_seed)

    pair_grouped = defaultdict(list)
    for row in pair_rows:
        pair_grouped[(row["class_id"], row["pair_index"])].append(row)
    pair_summary_rows = []
    for class_id in class_ids:
        for pair_index in range(args.ipc):
            rows = pair_grouped[(class_id, pair_index)]
            deltas = np.asarray([row["delta_g"] for row in rows])
            pair_summary_rows.append({
                "spec": args.spec,
                "class_id": class_id,
                "pair_index": pair_index,
                "real_g_mean": float(np.mean([row["real_g"] for row in rows])),
                "diffusion_g_mean": float(np.mean([row["diffusion_g"] for row in rows])),
                "delta_g_mean": float(deltas.mean()),
                "delta_g_std": float(deltas.std(ddof=1)) if len(deltas) > 1 else 0.0,
                "diffusion_preference_fraction": float(np.mean(deltas > 0.0)),
            })

    class_name_path = os.path.join(args.program_path, "misc", "class_names.txt")
    index_path = os.path.join(args.program_path, "misc", "class_indices.txt")
    with open(class_name_path, "r", encoding="utf-8") as file:
        all_names = [line.strip() for line in file]
    with open(index_path, "r", encoding="utf-8") as file:
        all_ids = [line.strip() for line in file if line.strip()]
    name_by_id = dict(zip(all_ids, all_names))

    for local_label, class_id in enumerate(class_ids):
        mean_losses = accumulated_losses[class_id] / total_runs
        best_mask_index = int(np.argmin(mean_losses))
        best_bits = combinations[best_mask_index].astype(int)
        gm_choices[class_id] = ["diffusion" if bit else "real" for bit in best_bits]
        diffusion_count = int(best_bits.sum())
        random_bits = np.zeros(args.ipc, dtype=int)
        if diffusion_count:
            selected = rng.choice(args.ipc, size=diffusion_count, replace=False)
            random_bits[selected] = 1
        random_choices[class_id] = ["diffusion" if bit else "real" for bit in random_bits]
        random_mask_index = int(sum(int(bit) << index for index, bit in enumerate(random_bits)))

        pair_deltas = [
            row["delta_g_mean"] for row in pair_summary_rows if row["class_id"] == class_id
        ]
        real_loss = float(np.mean(all_real_losses[class_id]))
        diffusion_loss = float(np.mean(all_diffusion_losses[class_id]))
        accuracy_gain = None
        if real_accuracy is not None and diffusion_accuracy is not None:
            accuracy_gain = diffusion_accuracy[class_id] - real_accuracy[class_id]
        class_rows.append({
            "spec": args.spec,
            "local_label": local_label,
            "class_id": class_id,
            "class_name": name_by_id.get(class_id, class_id).split(",")[0],
            "mean_pair_delta_g": float(np.mean(pair_deltas)),
            "all_real_relative_error": real_loss,
            "all_diffusion_relative_error": diffusion_loss,
            "set_score_diffusion_vs_real": real_loss - diffusion_loss,
            "gm_selected_relative_error": float(mean_losses[best_mask_index]),
            "random_matched_relative_error": float(mean_losses[random_mask_index]),
            "gm_error_improvement_over_random": float(
                mean_losses[random_mask_index] - mean_losses[best_mask_index]
            ),
            "gm_selected_diffusion_count": diffusion_count,
            "run_level_selection_agreement": float(
                np.mean(np.asarray(run_best_masks[class_id]) == best_mask_index)
            ),
            "accuracy_gain_diffusion_vs_real": accuracy_gain,
        })

    gm_records = _copy_selection(
        os.path.join(args.output_dir, "selected_gm"), class_ids, candidate_paths, gm_choices
    )
    random_records = _copy_selection(
        os.path.join(args.output_dir, "selected_random_matched"),
        class_ids, candidate_paths, random_choices,
    )

    _write_csv(os.path.join(args.output_dir, "pair_scores_raw.csv"), pair_rows)
    _write_csv(os.path.join(args.output_dir, "pair_scores_summary.csv"), pair_summary_rows)
    _write_csv(os.path.join(args.output_dir, "class_diagnostics.csv"), class_rows)
    _write_csv(os.path.join(args.output_dir, "gm_selection.csv"), gm_records)
    _write_csv(os.path.join(args.output_dir, "random_matched_selection.csv"), random_records)
    _plot_outputs(args.output_dir, class_rows, pair_rows, class_ids)

    diagnostics = {
        "format_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "spec": args.spec,
        "configuration": {
            "real_dir": os.path.abspath(args.real_dir),
            "diffusion_dir": os.path.abspath(args.diffusion_dir),
            "real_accuracy": os.path.abspath(args.real_accuracy) if args.real_accuracy else None,
            "diffusion_accuracy": os.path.abspath(args.diffusion_accuracy) if args.diffusion_accuracy else None,
            "feature_cache": os.path.abspath(feature_path),
            "cluster_centers": os.path.abspath(center_path),
            "ipc": args.ipc,
            "neighbor_count": args.neighbor_count,
            "neighbor_metric": args.neighbor_metric,
            "neighbor_weighting": args.neighbor_weighting,
            "model_seeds": model_seeds,
            "augmentations": args.augmentations,
            "random_selection_seed": args.random_selection_seed,
            "network": f"ResNet-AP-{args.depth}",
            "gradient": "last_layer_weight_and_bias",
            "selection_constraint": "exactly_one_candidate_per_cluster_pair",
        },
        "counts": {
            "candidate_images": args.nclass * args.ipc * 2,
            "unique_neighbor_images": neighbor_manifest["unique_neighbor_image_count"],
            "unique_images_forwarded_per_run": len(all_paths),
            "gradient_runs": total_runs,
        },
        "timing_seconds": time.perf_counter() - started,
    }
    if all(row["accuracy_gain_diffusion_vs_real"] is not None for row in class_rows):
        set_scores = [row["set_score_diffusion_vs_real"] for row in class_rows]
        pair_scores = [row["mean_pair_delta_g"] for row in class_rows]
        gains = [row["accuracy_gain_diffusion_vs_real"] for row in class_rows]
        set_sign, set_n = _sign_accuracy(set_scores, gains)
        pair_sign, pair_n = _sign_accuracy(pair_scores, gains)
        diagnostics["accuracy_relationship"] = {
            "set_score_spearman": _finite_correlation(stats.spearmanr, set_scores, gains),
            "set_score_pearson": _finite_correlation(stats.pearsonr, set_scores, gains),
            "set_score_sign_accuracy": set_sign,
            "set_score_sign_count": set_n,
            "mean_pair_score_spearman": _finite_correlation(stats.spearmanr, pair_scores, gains),
            "mean_pair_score_sign_accuracy": pair_sign,
            "mean_pair_score_sign_count": pair_n,
        }
    _write_json(os.path.join(args.output_dir, "diagnostics.json"), diagnostics)
    print(f"Gradient selection completed: {args.output_dir}")


if __name__ == "__main__":
    main()
