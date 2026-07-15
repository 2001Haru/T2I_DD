"""Decode saved CoDA representative latents without running diffusion."""

import argparse
import json
import math
import os
import pickle
from datetime import datetime, timezone

import numpy as np
import torch
from diffusers import AutoencoderKL
from PIL import Image
from tqdm import tqdm


CLASS_FILES = {
    "woof": "class_woof.txt",
    "nette": "class_nette.txt",
    "imagenet100": "class100.txt",
    "imagenet1k": "class_indices.txt",
    "IDC": "class_IDC.txt",
    "imageA": "imagenet-a.txt",
    "imageB": "imagenet-b.txt",
    "imageC": "imagenet-c.txt",
    "imageD": "imagenet-d.txt",
    "imageE": "imagenet-e.txt",
}


def _read_lines(path):
    with open(path, "r", encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]


def _selected_classes(spec, nclass, phase):
    try:
        filename = CLASS_FILES[spec]
    except KeyError as error:
        raise ValueError(f"Unsupported dataset subset: {spec}") from error
    class_ids = _read_lines(os.path.join(os.path.dirname(__file__), "misc", filename))
    start = max(phase, 0) * nclass
    selected = class_ids[start:start + nclass]
    if len(selected) != nclass:
        raise ValueError(f"Expected {nclass} classes for {spec}, found {len(selected)}.")
    return selected


def _load_centers(program_path, spec, ipc, n_neighbors, min_cluster_size, nclass):
    cluster_dir = os.path.join(program_path, "results", "clusterfile", spec)
    centers = {}
    for chunk_id in range((nclass + 9) // 10):
        path = os.path.join(
            cluster_dir,
            f"{ipc}_n_{n_neighbors}_s_{min_cluster_size}_saved_clusters_{chunk_id}.pkl",
        )
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Saved cluster centers were not found: {path}")
        with open(path, "rb") as file:
            centers.update(pickle.load(file))
    return centers


def _reshape_latent(center, latent_channels):
    flat = np.asarray(center, dtype=np.float32).reshape(-1)
    pixels = flat.size / latent_channels
    side = math.isqrt(int(pixels))
    if pixels != side * side:
        raise ValueError(
            f"Cannot reshape representative with {flat.size} values into {latent_channels} latent channels."
        )
    return flat.reshape(latent_channels, side, side)


def _to_pil(decoded, output_size):
    image = (decoded / 2 + 0.5).clamp(0, 1)
    image = image.detach().float().cpu().permute(1, 2, 0).numpy()
    image = Image.fromarray((image * 255).round().astype(np.uint8))
    if image.size != (output_size, output_size):
        image = image.resize((output_size, output_size), Image.Resampling.LANCZOS)
    return image


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
    parser.add_argument("--output_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    if os.path.exists(args.output_dir):
        raise FileExistsError(f"Refusing to overwrite reconstruction directory: {args.output_dir}")
    if args.batch_size < 1 or args.output_size < 1:
        parser.error("--batch_size and --output_size must be positive.")

    class_ids = _selected_classes(args.spec, args.nclass, args.phase)
    centers = _load_centers(
        args.program_path, args.spec, args.ipc, args.n_neighbors,
        args.min_cluster_size, args.nclass,
    )
    tasks = []
    for local_label, class_id in enumerate(class_ids):
        class_centers = centers.get(local_label)
        if class_centers is None or len(class_centers) != args.ipc:
            found = 0 if class_centers is None else len(class_centers)
            raise ValueError(f"Expected {args.ipc} centers for {class_id}, found {found}.")
        tasks.extend((class_id, shift, center) for shift, center in enumerate(class_centers))

    vae_path = os.path.join(args.local_model_path, "sdxl-base", "vaefixfp16")
    if not os.path.isfile(os.path.join(vae_path, "config.json")):
        raise FileNotFoundError(f"CoDA feature-extraction VAE was not found: {vae_path}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA reconstruction requested, but CUDA is unavailable.")

    dtype = torch.float16 if args.device.startswith("cuda") else torch.float32
    vae = AutoencoderKL.from_pretrained(vae_path, torch_dtype=dtype).to(args.device).eval()
    latent_channels = int(vae.config.latent_channels)
    scaling_factor = float(vae.config.scaling_factor)
    os.makedirs(args.output_dir)

    with torch.inference_mode():
        for start in tqdm(range(0, len(tasks), args.batch_size), desc="VAE reconstruction"):
            batch = tasks[start:start + args.batch_size]
            latents = np.stack([
                _reshape_latent(center, latent_channels) for _, _, center in batch
            ])
            latents = torch.from_numpy(latents).to(device=args.device, dtype=dtype)
            decoded = vae.decode(latents / scaling_factor, return_dict=True).sample
            for (class_id, shift, _), image_tensor in zip(batch, decoded):
                class_dir = os.path.join(args.output_dir, class_id)
                os.makedirs(class_dir, exist_ok=True)
                _to_pil(image_tensor, args.output_size).save(
                    os.path.join(class_dir, f"{shift}.png")
                )

    metadata = {
        "format_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "spec": args.spec,
        "ipc": args.ipc,
        "n_neighbors": args.n_neighbors,
        "min_cluster_size": args.min_cluster_size,
        "source": "saved_cluster_center_latents",
        "vae_path": vae_path,
        "vae_scaling_factor": scaling_factor,
        "output_size": args.output_size,
        "count": len(tasks),
    }
    with open(os.path.join(args.output_dir, "reconstruction_config.json"), "w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(f"Saved {len(tasks)} VAE reconstructions to: {args.output_dir}")


if __name__ == "__main__":
    main()
