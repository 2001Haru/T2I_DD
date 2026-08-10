import argparse
import gc
import json
import sys
from pathlib import Path

import torch
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionPipeline


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DISTILLATION_DIR = REPO_ROOT / "03_distiilation"
sys.path.insert(0, str(DISTILLATION_DIR))
from classes import IMAGENET2012_CLASSES  # noqa: E402
from common import (  # noqa: E402
    PROMPT_MODES,
    SUPERVISION_MODES,
    atomic_write_json,
    condition_name,
    ensure_manifest,
    sha256_file,
    shuffled_prompt_index,
    stable_image_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a text-supervision x inference-prompt factorial")
    parser.add_argument("--prototype", required=True)
    parser.add_argument("--dcs", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--model", action="append", default=[], help="MODE=DIFFUSERS_PATH")
    parser.add_argument("--supervisions", nargs="+", choices=SUPERVISION_MODES, default=SUPERVISION_MODES)
    parser.add_argument("--prompts", nargs="+", choices=PROMPT_MODES, default=PROMPT_MODES)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--generation-seeds", type=int, nargs="+", default=(0, 1))
    parser.add_argument("--ipc", type=int, default=10)
    parser.add_argument("--strength", type=float, default=0.7)
    parser.add_argument(
        "--visual-mode",
        choices=("prototype", "schedule_matched_noise", "pure_noise"),
        default="prototype",
    )
    parser.add_argument("--guidance-scale", type=float, default=10.0)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--negative-prompt", default="cartoon, anime, painting")
    parser.add_argument("--shuffle-shift", type=int, default=1)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def parse_models(base_model, entries):
    models = {"frozen": base_model}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"Expected MODE=PATH for --model, got {entry}")
        mode, path = entry.split("=", 1)
        if mode not in SUPERVISION_MODES or mode == "frozen":
            raise ValueError(f"Invalid trained supervision mode: {mode}")
        models[mode] = path
    return models


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def checkpoint_identity(reference):
    path = Path(reference)
    payload = {"reference": reference}
    if path.is_dir():
        payload["resolved_path"] = str(path.resolve())
        if (path / "model_index.json").is_file():
            payload["model_index_sha256"] = sha256_file(path / "model_index.json")
    return payload


def get_pipeline_embeds(pipe, prompt, negative_prompt, device):
    chunk = pipe.tokenizer.model_max_length
    prompt_ids = pipe.tokenizer(prompt, return_tensors="pt", truncation=False).input_ids
    negative_ids = pipe.tokenizer(negative_prompt, return_tensors="pt", truncation=False).input_ids
    length = max(prompt_ids.shape[-1], negative_ids.shape[-1])
    length = max(chunk, ((length + chunk - 1) // chunk) * chunk)
    prompt_ids = pipe.tokenizer(prompt, padding="max_length", max_length=length, truncation=False, return_tensors="pt").input_ids.to(device)
    negative_ids = pipe.tokenizer(negative_prompt, padding="max_length", max_length=length, truncation=False, return_tensors="pt").input_ids.to(device)
    prompt_embeds = [pipe.text_encoder(prompt_ids[:, start:start + chunk])[0] for start in range(0, length, chunk)]
    negative_embeds = [pipe.text_encoder(negative_ids[:, start:start + chunk])[0] for start in range(0, length, chunk)]
    return torch.cat(prompt_embeds, dim=1), torch.cat(negative_embeds, dim=1)


def validate(prototypes, dcs, ipc, shift):
    if set(prototypes) != set(dcs):
        raise ValueError("Prototype and DCS classes differ")
    for synset, values in prototypes.items():
        if synset not in IMAGENET2012_CLASSES or not values or ipc % len(values):
            raise ValueError(f"Invalid prototypes for {synset}")
        if len(values) != len(dcs[synset]):
            raise ValueError(f"DCS count differs for {synset}")
        shuffled_prompt_index(0, len(values), shift)


def prompt_for(synset, index, mode, dcs, shift):
    if mode == "label":
        return IMAGENET2012_CLASSES[synset], None
    source = index if mode == "correct" else shuffled_prompt_index(index, len(dcs[synset]), shift)
    return str(dcs[synset][source]), source


def schedule_matched_noise(prototype_data, image_seed, device, dtype=torch.float16):
    """Return an independent N(0, I) latent without consuming diffusion RNG state."""
    shape = tuple(torch.as_tensor(prototype_data).shape)
    generator = torch.Generator(device="cpu").manual_seed(
        (int(image_seed) + 2_000_000_011) % (2**63 - 1)
    )
    return torch.randn(shape, generator=generator, dtype=torch.float32).to(
        device=device, dtype=dtype
    ).unsqueeze(0)


def generate(pipe, prototypes, dcs, supervision, prompt_mode, generation_seed, output_dir, args):
    records = []
    completed = 0
    for class_index, (synset, values) in enumerate(prototypes.items()):
        class_dir = output_dir / synset
        class_dir.mkdir(parents=True, exist_ok=True)
        repeats = args.ipc // len(values)
        for repetition in range(repeats):
            for prototype_index, prototype_data in enumerate(values):
                image_index = repetition * len(values) + prototype_index
                prompt, source = prompt_for(synset, prototype_index, prompt_mode, dcs, args.shuffle_shift)
                image_seed = stable_image_seed(generation_seed, class_index, image_index)
                records.append({
                    "synset": synset,
                    "image_index": image_index,
                    "prototype_index": prototype_index,
                    "prompt_source_index": source,
                    "prompt": prompt,
                    "image_seed": image_seed,
                })
                destination = class_dir / f"image_{image_index:05d}.png"
                if destination.is_file():
                    completed += 1
                    continue
                prompt_embeds, negative_embeds = get_pipeline_embeds(pipe, prompt, args.negative_prompt, args.device)
                generator = torch.Generator(device=args.device).manual_seed(image_seed)
                call_args = {
                    "prompt_embeds": prompt_embeds,
                    "negative_prompt_embeds": negative_embeds,
                    "guidance_scale": args.guidance_scale,
                    "num_inference_steps": args.num_inference_steps,
                    "generator": generator,
                }
                if args.visual_mode in {"prototype", "schedule_matched_noise"}:
                    if args.visual_mode == "prototype":
                        visual_latent = torch.tensor(
                            prototype_data, dtype=torch.float16, device=args.device
                        ).unsqueeze(0)
                    else:
                        visual_latent = schedule_matched_noise(
                            prototype_data, image_seed, args.device
                        )
                    call_args["image"] = visual_latent
                    call_args["strength"] = args.strength
                else:
                    call_args["height"] = args.size
                    call_args["width"] = args.size
                image = pipe(**call_args).images[0]
                image.resize((args.size, args.size)).save(destination)
                completed += 1
                print(f"[{supervision}/{prompt_mode} seed={generation_seed}] {completed}/{len(prototypes) * args.ipc}")
    return records, completed


def main():
    args = parse_args()
    models = parse_models(args.base_model, args.model)
    missing = sorted(set(args.supervisions) - set(models))
    if missing:
        raise ValueError(f"No checkpoint supplied for: {missing}")
    prototype_path = Path(args.prototype).resolve()
    dcs_path = Path(args.dcs).resolve()
    prototypes, dcs = load_json(prototype_path), load_json(dcs_path)
    validate(prototypes, dcs, args.ipc, args.shuffle_shift)
    output_root = Path(args.output_root).resolve()

    for supervision in args.supervisions:
        checkpoint = models[supervision]
        pipeline_class = (
            StableDiffusionImg2ImgPipeline
            if args.visual_mode in {"prototype", "schedule_matched_noise"}
            else StableDiffusionPipeline
        )
        pipe = pipeline_class.from_pretrained(
            checkpoint, torch_dtype=torch.float16, safety_checker=None, requires_safety_checker=False
        ).to(args.device)
        pipe.set_progress_bar_config(disable=True)
        for generation_seed in args.generation_seeds:
            for prompt_mode in args.prompts:
                condition = condition_name(supervision, prompt_mode)
                output_dir = output_root / f"seed_{generation_seed}" / condition
                manifest = {
                    "format_version": 1,
                    "condition": condition,
                    "supervision_mode": supervision,
                    "prompt_mode": prompt_mode,
                    "checkpoint": checkpoint_identity(checkpoint),
                    "prototype_path": str(prototype_path),
                    "prototype_sha256": sha256_file(prototype_path),
                    "dcs_path": str(dcs_path),
                    "dcs_sha256": sha256_file(dcs_path),
                    "generation_seed": generation_seed,
                    "ipc": args.ipc,
                    "strength": args.strength if args.visual_mode != "pure_noise" else None,
                    "visual_mode": args.visual_mode,
                    "schedule_matched_noise_definition": (
                        "independent standard-normal x0 latent plus the ordinary img2img "
                        "forward noise at the requested strength; diffusion RNG is paired "
                        "with prototype mode"
                        if args.visual_mode == "schedule_matched_noise"
                        else None
                    ),
                    "guidance_scale": args.guidance_scale,
                    "num_inference_steps": args.num_inference_steps,
                    "negative_prompt": args.negative_prompt,
                    "shuffle_shift": args.shuffle_shift,
                    "paired_noise_across_all_cells": True,
                }
                ensure_manifest(output_dir, manifest, resume=args.resume)
                records, count = generate(pipe, prototypes, dcs, supervision, prompt_mode, generation_seed, output_dir, args)
                expected = len(prototypes) * args.ipc
                if count != expected:
                    raise RuntimeError(f"Generated {count}, expected {expected}: {output_dir}")
                atomic_write_json(output_dir / "prompt_records.json", records)
                atomic_write_json(output_dir / "complete.json", {"images": count})
        del pipe
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
