import argparse
import gc
import json
import os
import sys
from pathlib import Path

import torch
from diffusers import StableDiffusionImg2ImgPipeline


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
DISTILLATION_DIR = REPO_ROOT / "03_distiilation"
sys.path.insert(0, str(DISTILLATION_DIR))

from classes import IMAGENET2012_CLASSES  # noqa: E402
from common import (  # noqa: E402
    condition_matrix,
    ensure_manifest,
    sha256_file,
    shuffled_prompt_index,
    stable_image_seed,
)


VISUAL_INITIALIZATION = {
    "no_visual": (
        "Matched-schedule counterfactual q(z_t|z_0=0): a zero latent is noised by the "
        "same Img2Img pipeline, timestep schedule, and paired epsilon as the prototype cell."
    ),
    "prototype": "VLCP initialization q(z_t|z_0=cluster_prototype).",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate the frozen-SD ImageNette visual x text factorial"
    )
    parser.add_argument("--prototype", required=True)
    parser.add_argument("--dcs", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--generation-seeds", type=int, nargs="+", default=(0, 1))
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=[item["condition"] for item in condition_matrix()],
    )
    parser.add_argument("--ipc", type=int, default=10)
    parser.add_argument("--strength", type=float, default=0.7)
    parser.add_argument("--guidance-scale", type=float, default=10.0)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--negative-prompt", default="cartoon, anime, painting")
    parser.add_argument("--shuffle-shift", type=int, default=1)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def checkpoint_identity(checkpoint):
    path = Path(checkpoint)
    identity = {"reference": checkpoint}
    if path.is_dir():
        identity["resolved_path"] = str(path.resolve())
        model_index = path / "model_index.json"
        if model_index.is_file():
            identity["model_index_sha256"] = sha256_file(model_index)
    return identity


def get_pipeline_embeds(pipeline, prompt, negative_prompt, device):
    max_length = pipeline.tokenizer.model_max_length
    prompt_ids = pipeline.tokenizer(prompt, return_tensors="pt", truncation=False).input_ids
    negative_ids = pipeline.tokenizer(
        negative_prompt, return_tensors="pt", truncation=False
    ).input_ids
    sequence_length = max(prompt_ids.shape[-1], negative_ids.shape[-1])
    prompt_ids = pipeline.tokenizer(
        prompt,
        truncation=False,
        padding="max_length",
        max_length=sequence_length,
        return_tensors="pt",
    ).input_ids.to(device)
    negative_ids = pipeline.tokenizer(
        negative_prompt,
        truncation=False,
        padding="max_length",
        max_length=sequence_length,
        return_tensors="pt",
    ).input_ids.to(device)

    prompt_chunks = []
    negative_chunks = []
    for start in range(0, sequence_length, max_length):
        prompt_chunks.append(pipeline.text_encoder(prompt_ids[:, start : start + max_length])[0])
        negative_chunks.append(
            pipeline.text_encoder(negative_ids[:, start : start + max_length])[0]
        )
    return torch.cat(prompt_chunks, dim=1), torch.cat(negative_chunks, dim=1)


def validate_inputs(prototypes, dcs, ipc, shuffle_shift):
    if set(prototypes) != set(dcs):
        missing_dcs = sorted(set(prototypes) - set(dcs))
        missing_prototypes = sorted(set(dcs) - set(prototypes))
        raise ValueError(
            "Prototype/DCS class mismatch; "
            f"missing DCS={missing_dcs}, missing prototypes={missing_prototypes}"
        )
    for synset, values in prototypes.items():
        if synset not in IMAGENET2012_CLASSES:
            raise ValueError(f"Unknown ImageNet synset: {synset}")
        if not values or ipc % len(values) != 0:
            raise ValueError(f"IPC {ipc} must be divisible by {len(values)} prototypes for {synset}")
        if len(dcs[synset]) != len(values):
            raise ValueError(f"DCS/prototype count mismatch for {synset}")
        shuffled_prompt_index(0, len(values), shuffle_shift)


def select_prompt(synset, prototype_index, prompt_mode, dcs, shuffle_shift):
    if prompt_mode == "label":
        return IMAGENET2012_CLASSES[synset], None
    if prompt_mode == "dcs":
        source_index = prototype_index
    elif prompt_mode == "dcs_shuffled":
        source_index = shuffled_prompt_index(
            prototype_index, len(dcs[synset]), shift=shuffle_shift
        )
    else:
        raise ValueError(f"Unknown prompt mode: {prompt_mode}")
    return str(dcs[synset][source_index]), source_index


def build_prompt_records(prototypes, dcs, condition, generation_seed, args):
    records = []
    for class_index, synset in enumerate(prototypes):
        class_prototypes = prototypes[synset]
        repeats = args.ipc // len(class_prototypes)
        for repetition in range(repeats):
            for prototype_index in range(len(class_prototypes)):
                image_index = repetition * len(class_prototypes) + prototype_index
                prompt, source_index = select_prompt(
                    synset,
                    prototype_index,
                    condition["prompt_mode"],
                    dcs,
                    args.shuffle_shift,
                )
                records.append(
                    {
                        "synset": synset,
                        "class_index": class_index,
                        "image_index": image_index,
                        "prototype_index": prototype_index,
                        "prompt_source_index": source_index,
                        "prompt": prompt,
                        "image_seed": stable_image_seed(
                            generation_seed, class_index, image_index
                        ),
                    }
                )
    return records


def generate_condition(pipe, prototypes, records, condition, generation_seed, output_dir, args):
    records_by_key = {
        (record["synset"], record["image_index"]): record for record in records
    }
    expected = len(records)
    completed = 0
    for synset, class_prototypes in prototypes.items():
        class_dir = output_dir / synset
        class_dir.mkdir(parents=True, exist_ok=True)
        repeats = args.ipc // len(class_prototypes)
        for repetition in range(repeats):
            for prototype_index, prototype_data in enumerate(class_prototypes):
                image_index = repetition * len(class_prototypes) + prototype_index
                destination = class_dir / f"image_{image_index:05d}.png"
                if destination.is_file():
                    completed += 1
                    continue

                record = records_by_key[(synset, image_index)]
                prompt_embeds, negative_embeds = get_pipeline_embeds(
                    pipe, record["prompt"], args.negative_prompt, args.device
                )
                prototype = torch.tensor(
                    prototype_data, dtype=torch.float16, device=args.device
                ).unsqueeze(0)
                if condition["visual_mode"] == "no_visual":
                    init_latent = torch.zeros_like(prototype)
                else:
                    init_latent = prototype
                generator = torch.Generator(device=args.device).manual_seed(
                    record["image_seed"]
                )
                result = pipe(
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_embeds,
                    image=init_latent,
                    strength=args.strength,
                    guidance_scale=args.guidance_scale,
                    num_inference_steps=args.num_inference_steps,
                    generator=generator,
                )
                result.images[0].resize((args.size, args.size)).save(destination)
                completed += 1
                print(
                    f"[{condition['condition']} seed={generation_seed}] "
                    f"{completed}/{expected}: {destination}"
                )
    return completed


def atomic_write_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def main():
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if not 0 < args.strength <= 1:
        raise ValueError("--strength must be in (0, 1]")

    matrix = {item["condition"]: item for item in condition_matrix()}
    unknown_conditions = sorted(set(args.conditions) - set(matrix))
    if unknown_conditions:
        raise ValueError(f"Unknown conditions: {unknown_conditions}")

    prototype_path = Path(args.prototype).resolve()
    dcs_path = Path(args.dcs).resolve()
    output_root = Path(args.output_root).resolve()
    prototypes = load_json(prototype_path)
    dcs = load_json(dcs_path)
    validate_inputs(prototypes, dcs, args.ipc, args.shuffle_shift)

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(args.device)
    pipe.set_progress_bar_config(disable=True)

    for generation_seed in args.generation_seeds:
        for condition_name in args.conditions:
            condition = matrix[condition_name]
            output_dir = output_root / f"seed_{generation_seed}" / condition_name
            prompt_records = build_prompt_records(
                prototypes, dcs, condition, generation_seed, args
            )
            manifest = {
                "schema_version": 1,
                "condition": condition_name,
                "visual_mode": condition["visual_mode"],
                "visual_initialization": VISUAL_INITIALIZATION[condition["visual_mode"]],
                "prompt_mode": condition["prompt_mode"],
                "checkpoint": checkpoint_identity(args.base_model),
                "prototype_path": str(prototype_path),
                "prototype_sha256": sha256_file(prototype_path),
                "dcs_path": str(dcs_path),
                "dcs_sha256": sha256_file(dcs_path),
                "generation_seed": generation_seed,
                "ipc": args.ipc,
                "strength": args.strength,
                "guidance_scale": args.guidance_scale,
                "num_inference_steps": args.num_inference_steps,
                "negative_prompt": args.negative_prompt,
                "shuffle_strategy": {
                    "scope": "within_class",
                    "type": "cyclic_derangement",
                    "shift": args.shuffle_shift,
                    "fixed_across_generation_seeds": True,
                },
                "size": args.size,
                "seed_formula": "generation_seed*1000000 + class_index*10000 + image_index",
                "paired_noise_across_conditions": True,
                "prompt_records": prompt_records,
            }
            ensure_manifest(output_dir, manifest, resume=args.resume)
            count = generate_condition(
                pipe,
                prototypes,
                prompt_records,
                condition,
                generation_seed,
                output_dir,
                args,
            )
            if count != len(prompt_records):
                raise RuntimeError(
                    f"Generated {count} images for {condition_name}, expected {len(prompt_records)}"
                )
            atomic_write_json(output_dir / "complete.json", {"images": count})

    del pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
