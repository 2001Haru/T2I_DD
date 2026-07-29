import argparse
import csv
import gc
import json
import math
import os
import random
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps

from common import sha256_file
from diagnostic_common import atomic_write_json


EPSILON = 1e-12


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Assign real ImageNette images to stored VAE KMeans centers and "
            "inspect nearest cluster members"
        )
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--prototype", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--decoded-prototype-root")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--nearest-count", type=int, default=9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--posterior-mode",
        choices=("sample", "mean"),
        default="sample",
    )
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


def set_replay_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def nearest_center_assignments(latents, centers):
    latents = latents.float().flatten(1)
    centers = centers.float().flatten(1)
    dimensions = latents.shape[1]
    squared = (
        latents.square().sum(dim=1, keepdim=True)
        + centers.square().sum(dim=1).unsqueeze(0)
        - 2.0 * latents @ centers.T
    ).clamp_min_(0.0)
    sorted_squared, sorted_indices = squared.sort(dim=1)
    nearest_rmse = torch.sqrt(sorted_squared[:, 0] / dimensions)
    second_rmse = torch.sqrt(sorted_squared[:, 1] / dimensions)
    return (
        sorted_indices[:, 0],
        nearest_rmse,
        second_rmse,
        second_rmse - nearest_rmse,
    )


def nearest_other_center_distances(centers):
    centers = np.asarray(centers, dtype=np.float64)
    flattened = centers.reshape(len(centers), -1)
    differences = flattened[:, None, :] - flattened[None, :, :]
    rmse = np.sqrt(np.mean(differences**2, axis=2))
    np.fill_diagonal(rmse, np.inf)
    return rmse.min(axis=1)


def quantile(values, probability):
    return float(np.quantile(np.asarray(values, dtype=np.float64), probability))


def summarize_cluster(rows, nearest_other_center_rmse):
    distances = [float(row["center_rmse"]) for row in rows]
    margins = [float(row["assignment_margin_rmse"]) for row in rows]
    nearest = min(distances)
    median = statistics.median(distances)
    p10 = quantile(distances, 0.10)
    return {
        "assigned_images": len(rows),
        "nearest_center_rmse": nearest,
        "p10_center_rmse": p10,
        "p25_center_rmse": quantile(distances, 0.25),
        "median_center_rmse": median,
        "p75_center_rmse": quantile(distances, 0.75),
        "p90_center_rmse": quantile(distances, 0.90),
        "maximum_center_rmse": max(distances),
        "nearest_to_median_ratio": nearest / max(median, EPSILON),
        "nearest_to_p10_ratio": nearest / max(p10, EPSILON),
        "nearest_other_center_rmse": nearest_other_center_rmse,
        "nearest_member_to_other_center_ratio": (
            nearest / max(nearest_other_center_rmse, EPSILON)
        ),
        "mean_assignment_margin_rmse": statistics.fmean(margins),
        "median_assignment_margin_rmse": statistics.median(margins),
    }


def audit_manifest(args):
    data_root = Path(args.data_root).resolve()
    prototype = Path(args.prototype).resolve()
    base_model = Path(args.base_model).resolve()
    decoded_root = (
        str(Path(args.decoded_prototype_root).resolve())
        if args.decoded_prototype_root
        else None
    )
    return {
        "schema_version": 1,
        "data_root": str(data_root),
        "prototype": str(prototype),
        "prototype_sha256": sha256_file(prototype),
        "base_model": str(base_model),
        "decoded_prototype_root": decoded_root,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "image_size": args.image_size,
        "nearest_count": args.nearest_count,
        "seed": args.seed,
        "posterior_mode": args.posterior_mode,
        "distance": "per-latent-element RMSE to stored KMeans center",
    }


def ensure_output(output_dir, manifest, resume):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "cluster_member_audit_manifest.json"
    if manifest_path.is_file():
        current = json.loads(manifest_path.read_text(encoding="utf-8"))
        if current != manifest:
            raise RuntimeError(f"Resume configuration differs from {manifest_path}")
        if not resume:
            raise RuntimeError(
                f"Cluster-member audit already exists; pass --resume: {output_dir}"
            )
    elif any(output_dir.iterdir()):
        raise RuntimeError(f"Non-empty audit directory has no manifest: {output_dir}")
    else:
        atomic_write_json(manifest_path, manifest)


def build_dataset(data_root, image_size):
    from torchvision import datasets, transforms

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )

    class IndexedImageFolder(datasets.ImageFolder):
        def __getitem__(self, index):
            path, target = self.samples[index]
            image = self.loader(path)
            if self.transform is not None:
                image = self.transform(image)
            return image, target, index

    dataset = IndexedImageFolder(
        root=str(Path(data_root) / "train"),
        transform=transform,
    )
    return dataset


def extract_assignments(args, assignments_path):
    from diffusers import AutoencoderKL
    from torch.utils.data import DataLoader

    set_replay_seed(args.seed)
    dataset = build_dataset(args.data_root, args.image_size)
    prototypes = json.loads(Path(args.prototype).read_text(encoding="utf-8"))
    if set(dataset.classes) != set(prototypes):
        raise RuntimeError(
            "Dataset/prototype class mismatch: "
            f"{sorted(set(dataset.classes) ^ set(prototypes))}"
        )

    centers = {
        synset: torch.tensor(value, dtype=torch.float32, device=args.device)
        for synset, value in prototypes.items()
    }
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
    )
    dtype = torch.float16 if str(args.device).startswith("cuda") else torch.float32
    vae = AutoencoderKL.from_pretrained(
        args.base_model,
        subfolder="vae",
        torch_dtype=dtype,
        local_files_only=True,
    ).to(args.device)
    vae.eval()

    rows = []
    completed = 0
    for images, labels, indices in loader:
        images = images.to(device=args.device, dtype=dtype) * 2.0 - 1.0
        with torch.inference_mode():
            distribution = vae.encode(images).latent_dist
            if args.posterior_mode == "sample":
                latents = distribution.sample()
            else:
                latents = distribution.mean
            latents = float(vae.config.scaling_factor) * latents

        labels_cpu = labels.tolist()
        indices_cpu = indices.tolist()
        for class_index in sorted(set(labels_cpu)):
            selected_positions = [
                position
                for position, value in enumerate(labels_cpu)
                if value == class_index
            ]
            synset = dataset.classes[class_index]
            selected_latents = latents[selected_positions]
            cluster_ids, distances, second_distances, margins = (
                nearest_center_assignments(selected_latents, centers[synset])
            )
            for local_position, dataset_index in enumerate(
                [indices_cpu[position] for position in selected_positions]
            ):
                path = Path(dataset.samples[dataset_index][0]).resolve()
                rows.append(
                    {
                        "synset": synset,
                        "dataset_index": dataset_index,
                        "image_path": str(path),
                        "assigned_cluster": int(cluster_ids[local_position].item()),
                        "center_rmse": float(distances[local_position].item()),
                        "second_center_rmse": float(
                            second_distances[local_position].item()
                        ),
                        "assignment_margin_rmse": float(
                            margins[local_position].item()
                        ),
                    }
                )
        completed += len(images)
        print(f"VAE assignment: {completed}/{len(dataset)}")

    del vae, centers
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    rows.sort(
        key=lambda row: (
            row["synset"],
            row["assigned_cluster"],
            row["center_rmse"],
            row["image_path"],
        )
    )
    temporary = Path(assignments_path).with_suffix(".csv.tmp")
    write_csv(temporary, rows)
    os.replace(temporary, assignments_path)
    return rows, prototypes


def load_or_extract_assignments(args, output_dir):
    assignments_path = Path(output_dir) / "latent_assignments.csv"
    if assignments_path.is_file():
        if not args.resume:
            raise RuntimeError(
                f"Assignments already exist; pass --resume: {assignments_path}"
            )
        rows = read_csv(assignments_path)
        prototypes = json.loads(Path(args.prototype).read_text(encoding="utf-8"))
        return rows, prototypes
    return extract_assignments(args, assignments_path)


def fit_tile(image, size):
    return ImageOps.fit(image.convert("RGB"), (size, size), method=Image.Resampling.LANCZOS)


def make_montage(
    synset,
    cluster_index,
    selected_rows,
    summary,
    output_path,
    decoded_path=None,
):
    tile_size = 224
    label_height = 34
    columns = 3
    entries = []
    if decoded_path is not None and decoded_path.is_file():
        entries.append(("decoded center", decoded_path))
    for rank, row in enumerate(selected_rows, start=1):
        entries.append(
            (
                f"#{rank} rmse={float(row['center_rmse']):.4f}",
                Path(row["image_path"]),
            )
        )
    rows = math.ceil(len(entries) / columns)
    header_height = 62
    canvas = Image.new(
        "RGB",
        (columns * tile_size, header_height + rows * (tile_size + label_height)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 7), f"{synset} cluster {cluster_index}", fill="black")
    draw.text(
        (8, 29),
        (
            f"nearest={summary['nearest_center_rmse']:.4f}  "
            f"median={summary['median_center_rmse']:.4f}  "
            f"ratio={summary['nearest_to_median_ratio']:.3f}"
        ),
        fill="black",
    )
    for position, (label, path) in enumerate(entries):
        column = position % columns
        row = position // columns
        x = column * tile_size
        y = header_height + row * (tile_size + label_height)
        with Image.open(path) as image:
            canvas.paste(fit_tile(image, tile_size), (x, y))
        draw.text((x + 5, y + tile_size + 8), label, fill="black")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def summarize_assignments(args, rows, prototypes, output_dir):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["synset"], int(row["assigned_cluster"]))].append(row)

    cluster_rows = []
    nearest_payload = {}
    missing_clusters = []
    for synset, center_values in prototypes.items():
        other_center_distances = nearest_other_center_distances(center_values)
        nearest_payload[synset] = {}
        for cluster_index in range(len(center_values)):
            selected = grouped.get((synset, cluster_index), [])
            if not selected:
                missing_clusters.append((synset, cluster_index))
                continue
            selected.sort(key=lambda row: float(row["center_rmse"]))
            summary = summarize_cluster(
                selected, float(other_center_distances[cluster_index])
            )
            nearest = selected[: args.nearest_count]
            decoded_path = None
            if args.decoded_prototype_root:
                decoded_path = (
                    Path(args.decoded_prototype_root)
                    / synset
                    / f"prototype_{cluster_index:05d}.png"
                )
            montage_path = (
                Path(output_dir)
                / "montages"
                / synset
                / f"cluster_{cluster_index:02d}.png"
            )
            make_montage(
                synset,
                cluster_index,
                nearest,
                summary,
                montage_path,
                decoded_path,
            )
            nearest_payload[synset][str(cluster_index)] = [
                {
                    "rank": rank,
                    "image_path": row["image_path"],
                    "center_rmse": float(row["center_rmse"]),
                    "second_center_rmse": float(row["second_center_rmse"]),
                    "assignment_margin_rmse": float(
                        row["assignment_margin_rmse"]
                    ),
                }
                for rank, row in enumerate(nearest, start=1)
            ]
            cluster_rows.append(
                {
                    "synset": synset,
                    "cluster_index": cluster_index,
                    **summary,
                    "nearest_image_path": nearest[0]["image_path"],
                    "montage_path": str(montage_path.resolve()),
                }
            )
    if missing_clusters:
        raise RuntimeError(f"Clusters with no assigned images: {missing_clusters}")

    class_rows = []
    class_grouped = defaultdict(list)
    for row in cluster_rows:
        class_grouped[row["synset"]].append(row)
    for synset, selected in sorted(class_grouped.items()):
        class_rows.append(
            {
                "synset": synset,
                "clusters": len(selected),
                "minimum_cluster_size": min(
                    int(row["assigned_images"]) for row in selected
                ),
                "maximum_cluster_size": max(
                    int(row["assigned_images"]) for row in selected
                ),
                "mean_nearest_center_rmse": statistics.fmean(
                    row["nearest_center_rmse"] for row in selected
                ),
                "mean_median_center_rmse": statistics.fmean(
                    row["median_center_rmse"] for row in selected
                ),
                "mean_nearest_to_median_ratio": statistics.fmean(
                    row["nearest_to_median_ratio"] for row in selected
                ),
                "mean_nearest_member_to_other_center_ratio": statistics.fmean(
                    row["nearest_member_to_other_center_ratio"]
                    for row in selected
                ),
            }
        )

    write_csv(Path(output_dir) / "cluster_distance_summary.csv", cluster_rows)
    write_csv(Path(output_dir) / "class_distance_summary.csv", class_rows)
    atomic_write_json(
        Path(output_dir) / "nearest_cluster_members.json", nearest_payload
    )
    return cluster_rows, class_rows


def plot_center_gaps(cluster_rows, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as pyplot

    nearest = [row["nearest_center_rmse"] for row in cluster_rows]
    median = [row["median_center_rmse"] for row in cluster_rows]
    ratios = [row["nearest_to_median_ratio"] for row in cluster_rows]
    separation = [
        row["nearest_member_to_other_center_ratio"] for row in cluster_rows
    ]
    sizes = [row["assigned_images"] for row in cluster_rows]

    figure, axes = pyplot.subplots(1, 3, figsize=(16, 5))
    axes[0].scatter(median, nearest, alpha=0.7)
    maximum = max(max(median), max(nearest))
    axes[0].plot([0, maximum], [0, maximum], linestyle="--", color="black")
    axes[0].set_xlabel("Median assigned-member RMSE")
    axes[0].set_ylabel("Nearest-member RMSE")
    axes[0].set_title("Is the nearest image actually near the center?")

    axes[1].hist(ratios, bins=15)
    axes[1].set_xlabel("Nearest / median member RMSE")
    axes[1].set_ylabel("Clusters")
    axes[1].set_title("Center occupancy gap")

    axes[2].scatter(separation, sizes, alpha=0.7)
    axes[2].axvline(1.0, linestyle="--", color="black")
    axes[2].set_xlabel("Nearest member / nearest other-center RMSE")
    axes[2].set_ylabel("Assigned images")
    axes[2].set_title("Member proximity versus center separation")

    for axis in axes:
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    pyplot.close(figure)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    manifest = audit_manifest(args)
    ensure_output(output_dir, manifest, args.resume)
    rows, prototypes = load_or_extract_assignments(args, output_dir)
    cluster_rows, class_rows = summarize_assignments(
        args, rows, prototypes, output_dir
    )
    plot_center_gaps(
        cluster_rows,
        output_dir / "cluster_center_gap.png",
    )
    payload = {
        **manifest,
        "images": len(rows),
        "classes": len(class_rows),
        "clusters": len(cluster_rows),
        "important_caveats": [
            (
                "Nearest means nearest among available images; absolute RMSE and "
                "nearest-to-median ratios determine whether it is genuinely close."
            ),
            (
                "The original prototype fit removed 10% LOF outliers. This audit "
                "assigns every training image to its nearest stored center."
            ),
            (
                "Sample replay uses the original seed, image size, batch size, and "
                "shuffle protocol, but exact posterior samples can still depend on "
                "software and hardware determinism."
            ),
            (
                "VAE latent distance defines the original clusters but does not "
                "guarantee object-level semantic coherence."
            ),
        ],
        "global_summary": {
            "mean_nearest_to_median_ratio": statistics.fmean(
                row["nearest_to_median_ratio"] for row in cluster_rows
            ),
            "mean_nearest_member_to_other_center_ratio": statistics.fmean(
                row["nearest_member_to_other_center_ratio"]
                for row in cluster_rows
            ),
            "minimum_assigned_images": min(
                row["assigned_images"] for row in cluster_rows
            ),
            "maximum_assigned_images": max(
                row["assigned_images"] for row in cluster_rows
            ),
        },
    }
    atomic_write_json(output_dir / "cluster_member_audit_summary.json", payload)
    print(json.dumps(payload["global_summary"], indent=2))


if __name__ == "__main__":
    main()
