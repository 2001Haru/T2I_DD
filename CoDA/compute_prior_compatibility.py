"""Measure SDXL class-prior compatibility on saved CoDA representatives."""

import argparse
import csv
import json
import math
import os
import pickle
from datetime import datetime, timezone

import numpy as np
import torch
from diffusers import DDPMScheduler
from tqdm import tqdm

from CoDA_SDXLBasePipeline import CoDA_SDXL
from reconstruct_representatives import _selected_classes


def _class_names():
    misc_dir = os.path.join(os.path.dirname(__file__), "misc")
    with open(os.path.join(misc_dir, "class_indices.txt"), "r", encoding="utf-8") as file:
        class_ids = [line.strip() for line in file if line.strip()]
    with open(os.path.join(misc_dir, "class_names.txt"), "r", encoding="utf-8") as file:
        names = [line.strip() for line in file]
    return dict(zip(class_ids, names))


def _load_centers(program_path, spec, ipc, n_neighbors, min_cluster_size, nclass):
    cluster_dir = os.path.join(program_path, "results", "clusterfile", spec)
    centers = {}
    source_paths = []
    for chunk_id in range((nclass + 9) // 10):
        path = os.path.join(
            cluster_dir,
            f"{ipc}_n_{n_neighbors}_s_{min_cluster_size}_saved_clusters_{chunk_id}.pkl",
        )
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Saved cluster centers were not found: {path}")
        with open(path, "rb") as file:
            centers.update(pickle.load(file))
        source_paths.append(path)
    return centers, source_paths


def _reshape_center(center, latent_channels=4):
    flat = np.asarray(center, dtype=np.float32).reshape(-1)
    spatial_values = flat.size / latent_channels
    side = math.isqrt(int(spatial_values))
    if spatial_values != side * side:
        raise ValueError(f"Invalid latent center size: {flat.size}")
    return flat.reshape(latent_channels, side, side)


def _timesteps(num_train_timesteps, count, minimum_fraction, maximum_fraction):
    if count < 1:
        raise ValueError("Timestep count must be positive.")
    low = round((num_train_timesteps - 1) * minimum_fraction)
    high = round((num_train_timesteps - 1) * maximum_fraction)
    return np.linspace(low, high, count, dtype=np.int64).tolist()


def _encode_conditions(pipeline, class_ids, name_by_id, device):
    pipeline.text_encoder.to(device)
    pipeline.text_encoder_2.to(device)
    encoded = {}
    projection_dim = int(pipeline.text_encoder_2.config.projection_dim)

    with torch.inference_mode():
        for class_id in tqdm(class_ids, desc="Encoding class prompts"):
            class_name = name_by_id[class_id].split(",")[0].strip()
            positive, negative, pooled_positive, pooled_negative = pipeline.encode_prompt(
                prompt=class_name,
                device=device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
                negative_prompt=None,
            )
            time_ids = pipeline._get_add_time_ids(
                original_size=(1024, 1024),
                crops_coords_top_left=(0, 0),
                target_size=(1024, 1024),
                dtype=positive.dtype,
                text_encoder_projection_dim=projection_dim,
            )
            encoded[class_id] = {
                "class_name": class_name,
                "positive": positive.cpu(),
                "negative": negative.cpu(),
                "pooled_positive": pooled_positive.cpu(),
                "pooled_negative": pooled_negative.cpu(),
                "time_ids": time_ids.cpu(),
            }

    pipeline.text_encoder.to("cpu")
    pipeline.text_encoder_2.to("cpu")
    torch.cuda.empty_cache()
    return encoded


def _prediction_target(scheduler, clean_latents, noise, timesteps):
    prediction_type = scheduler.config.prediction_type
    if prediction_type == "epsilon":
        return noise
    if prediction_type == "v_prediction":
        return scheduler.get_velocity(clean_latents, noise, timesteps)
    if prediction_type == "sample":
        return clean_latents
    raise ValueError(f"Unsupported scheduler prediction type: {prediction_type}")


def _write_csv(path, rows):
    if not rows:
        raise ValueError(f"Cannot write an empty CSV: {path}")
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _aggregate(records, keys):
    groups = {}
    for record in records:
        key = tuple(record[name] for name in keys)
        groups.setdefault(key, []).append(record)

    rows = []
    for key, items in groups.items():
        pcs = np.asarray([item["pcs"] for item in items], dtype=np.float64)
        row = {name: value for name, value in zip(keys, key)}
        first = items[0]
        for name in ("class_name", "local_label"):
            if name in first and name not in row:
                row[name] = first[name]
        row.update({
            "pcs_mean": float(np.mean(pcs)),
            "pcs_median": float(np.median(pcs)),
            "pcs_std": float(np.std(pcs, ddof=1)) if len(pcs) > 1 else 0.0,
            "pcs_positive_fraction": float(np.mean(pcs > 0.0)),
            "unconditional_mse_mean": float(np.mean([item["unconditional_mse"] for item in items])),
            "conditional_mse_mean": float(np.mean([item["conditional_mse"] for item in items])),
            "record_count": len(items),
        })
        rows.append(row)
    return sorted(rows, key=lambda row: tuple(row[name] for name in keys))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program_path", default=".")
    parser.add_argument("--local_model_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--nclass", type=int, default=10)
    parser.add_argument("--phase", type=int, default=0)
    parser.add_argument("--ipc", type=int, default=10)
    parser.add_argument("--n_neighbors", type=int, default=85)
    parser.add_argument("--min_cluster_size", type=int, default=55)
    parser.add_argument("--num_timesteps", type=int, default=8)
    parser.add_argument("--min_timestep_fraction", type=float, default=0.05)
    parser.add_argument("--max_timestep_fraction", type=float, default=0.95)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    if os.path.exists(args.output_dir):
        raise FileExistsError(f"Refusing to overwrite PCS output: {args.output_dir}")
    if args.batch_size < 1:
        parser.error("--batch_size must be positive.")
    if not 0 <= args.min_timestep_fraction < args.max_timestep_fraction <= 1:
        parser.error("Timestep fractions must satisfy 0 <= min < max <= 1.")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA PCS computation requested, but CUDA is unavailable.")

    class_ids = _selected_classes(args.spec, args.nclass, args.phase)
    name_by_id = _class_names()
    centers, center_paths = _load_centers(
        args.program_path, args.spec, args.ipc, args.n_neighbors,
        args.min_cluster_size, args.nclass,
    )

    base_path = os.path.join(args.local_model_path, "sdxl-base")
    if not os.path.isfile(os.path.join(base_path, "model_index.json")):
        raise FileNotFoundError(f"Complete SDXL base pipeline was not found: {base_path}")
    dtype = torch.float16 if args.device.startswith("cuda") else torch.float32
    pipeline = CoDA_SDXL.from_pretrained(
        base_path,
        torch_dtype=dtype,
        use_safetensors=True,
        variant="fp16",
    )
    scheduler = DDPMScheduler.from_config(pipeline.scheduler.config)
    if scheduler.config.prediction_type not in ("epsilon", "v_prediction", "sample"):
        raise ValueError(f"Unsupported prediction type: {scheduler.config.prediction_type}")
    timestep_values = _timesteps(
        scheduler.config.num_train_timesteps,
        args.num_timesteps,
        args.min_timestep_fraction,
        args.max_timestep_fraction,
    )
    conditions = _encode_conditions(pipeline, class_ids, name_by_id, args.device)
    pipeline.unet.to(args.device).eval()

    tasks = []
    for local_label, class_id in enumerate(class_ids):
        class_centers = centers.get(local_label)
        if class_centers is None or len(class_centers) != args.ipc:
            found = 0 if class_centers is None else len(class_centers)
            raise ValueError(f"Expected {args.ipc} centers for {class_id}, found {found}.")
        for representative_index, center in enumerate(class_centers):
            latent = _reshape_center(center)
            for timestep_index, timestep in enumerate(timestep_values):
                tasks.append({
                    "local_label": local_label,
                    "class_id": class_id,
                    "representative_index": representative_index,
                    "timestep_index": timestep_index,
                    "timestep": timestep,
                    "latent": latent,
                })

    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    records = []
    with torch.inference_mode():
        for start in tqdm(range(0, len(tasks), args.batch_size), desc=f"PCS {args.spec}"):
            batch = tasks[start:start + args.batch_size]
            clean = torch.from_numpy(np.stack([item["latent"] for item in batch])).to(
                device=args.device, dtype=dtype
            )
            timesteps = torch.tensor(
                [item["timestep"] for item in batch], device=args.device, dtype=torch.long
            )
            noise = torch.randn(clean.shape, generator=generator, device=args.device, dtype=dtype)
            noisy = scheduler.add_noise(clean, noise, timesteps)
            target = _prediction_target(scheduler, clean, noise, timesteps)

            negative = torch.cat([
                conditions[item["class_id"]]["negative"] for item in batch
            ]).to(args.device)
            positive = torch.cat([
                conditions[item["class_id"]]["positive"] for item in batch
            ]).to(args.device)
            pooled_negative = torch.cat([
                conditions[item["class_id"]]["pooled_negative"] for item in batch
            ]).to(args.device)
            pooled_positive = torch.cat([
                conditions[item["class_id"]]["pooled_positive"] for item in batch
            ]).to(args.device)
            time_ids = torch.cat([
                conditions[item["class_id"]]["time_ids"] for item in batch
            ]).to(args.device)

            predictions = pipeline.unet(
                torch.cat([noisy, noisy]),
                torch.cat([timesteps, timesteps]),
                encoder_hidden_states=torch.cat([negative, positive]),
                added_cond_kwargs={
                    "text_embeds": torch.cat([pooled_negative, pooled_positive]),
                    "time_ids": torch.cat([time_ids, time_ids]),
                },
                return_dict=True,
            ).sample
            unconditional, conditional = predictions.chunk(2)
            reduce_dims = tuple(range(1, target.ndim))
            unconditional_mse = (unconditional.float() - target.float()).square().mean(dim=reduce_dims)
            conditional_mse = (conditional.float() - target.float()).square().mean(dim=reduce_dims)
            pcs = (unconditional_mse - conditional_mse) / unconditional_mse.clamp_min(1e-12)

            for index, item in enumerate(batch):
                records.append({
                    "spec": args.spec,
                    "local_label": item["local_label"],
                    "class_id": item["class_id"],
                    "class_name": conditions[item["class_id"]]["class_name"],
                    "representative_index": item["representative_index"],
                    "timestep_index": item["timestep_index"],
                    "timestep": item["timestep"],
                    "unconditional_mse": float(unconditional_mse[index].item()),
                    "conditional_mse": float(conditional_mse[index].item()),
                    "pcs": float(pcs[index].item()),
                })

    os.makedirs(args.output_dir)
    _write_csv(os.path.join(args.output_dir, "pcs_raw.csv"), records)
    class_rows = _aggregate(records, ("spec", "class_id"))
    class_timestep_rows = _aggregate(
        records, ("spec", "class_id", "timestep_index", "timestep")
    )
    timestep_rows = _aggregate(records, ("spec", "timestep_index", "timestep"))
    _write_csv(os.path.join(args.output_dir, "pcs_per_class.csv"), class_rows)
    _write_csv(
        os.path.join(args.output_dir, "pcs_per_class_timestep.csv"), class_timestep_rows
    )
    _write_csv(os.path.join(args.output_dir, "pcs_per_timestep.csv"), timestep_rows)
    metadata = {
        "format_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "spec": args.spec,
        "class_ids": class_ids,
        "ipc": args.ipc,
        "timesteps": timestep_values,
        "seed": args.seed,
        "pcs_definition": "(unconditional_mse - conditional_mse) / unconditional_mse",
        "condition_prompt": "first ImageNet class name",
        "unconditional_prompt": None,
        "prediction_type": scheduler.config.prediction_type,
        "source_cluster_files": center_paths,
        "model_path": base_path,
        "record_count": len(records),
    }
    with open(os.path.join(args.output_dir, "pcs_config.json"), "w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(f"Saved {len(records)} PCS measurements to: {args.output_dir}")


if __name__ == "__main__":
    main()
