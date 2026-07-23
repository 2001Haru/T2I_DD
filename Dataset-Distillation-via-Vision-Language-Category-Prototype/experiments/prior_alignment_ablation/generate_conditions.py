import argparse
import gc
import json
import sys
from pathlib import Path

import torch
from diffusers import StableDiffusionImg2ImgPipeline


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
DISTILLATION_DIR = REPO_ROOT / "03_distiilation"
sys.path.insert(0, str(DISTILLATION_DIR))

from classes import IMAGENET2012_CLASSES  # noqa: E402
from common import condition_matrix, ensure_manifest, sha256_file, stable_image_seed  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Generate the paired frozen/fine-tuned x label/DCS ablation")
    parser.add_argument("--prototype", required=True)
    parser.add_argument("--dcs", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--finetuned-model", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--generation-seeds", type=int, nargs="+", default=(0, 1))
    parser.add_argument("--ipc", type=int, default=10)
    parser.add_argument("--strength", type=float, default=0.7)
    parser.add_argument("--guidance-scale", type=float, default=10.0)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--negative-prompt", default="cartoon, anime, painting")
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
    negative_ids = pipeline.tokenizer(negative_prompt, return_tensors="pt", truncation=False).input_ids
    sequence_length = max(prompt_ids.shape[-1], negative_ids.shape[-1])
    prompt_ids = pipeline.tokenizer(
        prompt, truncation=False, padding="max_length", max_length=sequence_length, return_tensors="pt"
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
        negative_chunks.append(pipeline.text_encoder(negative_ids[:, start : start + max_length])[0])
    return torch.cat(prompt_chunks, dim=1), torch.cat(negative_chunks, dim=1)


def validate_inputs(prototypes, dcs, ipc):
    if set(prototypes) != set(dcs):
        missing_dcs = sorted(set(prototypes) - set(dcs))
        missing_prototypes = sorted(set(dcs) - set(prototypes))
        raise ValueError(
            f"Prototype/DCS class mismatch; missing DCS={missing_dcs}, missing prototypes={missing_prototypes}"
        )
    for synset, values in prototypes.items():
        if synset not in IMAGENET2012_CLASSES:
            raise ValueError(f"Unknown ImageNet synset: {synset}")
        if not values or ipc % len(values) != 0:
            raise ValueError(f"IPC {ipc} must be divisible by {len(values)} prototypes for {synset}")
        if len(dcs[synset]) != len(values):
            raise ValueError(f"DCS/prototype count mismatch for {synset}")


def generate_condition(pipe, prototypes, dcs, condition, generation_seed, output_dir, args):
    expected = len(prototypes) * args.ipc
    completed = 0
    for class_index, synset in enumerate(prototypes):
        class_dir = output_dir / synset
        class_dir.mkdir(parents=True, exist_ok=True)
        class_prototypes = prototypes[synset]
        repeats = args.ipc // len(class_prototypes)
        for repetition in range(repeats):
            for prototype_index, prototype_data in enumerate(class_prototypes):
                image_index = repetition * len(class_prototypes) + prototype_index
                destination = class_dir / f"image_{image_index:05d}.png"
                if destination.is_file():
                    completed += 1
                    continue

                if condition["prompt_mode"] == "label":
                    prompt = IMAGENET2012_CLASSES[synset]
                else:
                    prompt = str(dcs[synset][prototype_index])
                prompt_embeds, negative_embeds = get_pipeline_embeds(
                    pipe, prompt, args.negative_prompt, args.device
                )
                prototype = torch.tensor(prototype_data, dtype=torch.float16, device=args.device).unsqueeze(0)
                generator = torch.Generator(device=args.device).manual_seed(
                    stable_image_seed(generation_seed, class_index, image_index)
                )
                result = pipe(
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_embeds,
                    image=prototype,
                    strength=args.strength,
                    guidance_scale=args.guidance_scale,
                    num_inference_steps=args.num_inference_steps,
                    generator=generator,
                )
                result.images[0].resize((args.size, args.size)).save(destination)
                completed += 1
                print(f"[{condition['condition']} seed={generation_seed}] {completed}/{expected}: {destination}")
    return completed


def main():
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    prototype_path = Path(args.prototype).resolve()
    dcs_path = Path(args.dcs).resolve()
    output_root = Path(args.output_root).resolve()
    prototypes = load_json(prototype_path)
    dcs = load_json(dcs_path)
    validate_inputs(prototypes, dcs, args.ipc)

    matrix = condition_matrix(args.base_model, args.finetuned_model)
    for model_mode in ("frozen", "finetuned"):
        model_conditions = [item for item in matrix if item["model_mode"] == model_mode]
        checkpoint = model_conditions[0]["checkpoint"]
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            checkpoint,
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False,
        ).to(args.device)
        pipe.set_progress_bar_config(disable=True)

        for generation_seed in args.generation_seeds:
            for condition in model_conditions:
                output_dir = output_root / f"seed_{generation_seed}" / condition["condition"]
                manifest = {
                    "schema_version": 1,
                    "condition": condition["condition"],
                    "model_mode": condition["model_mode"],
                    "prompt_mode": condition["prompt_mode"],
                    "checkpoint": checkpoint_identity(checkpoint),
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
                    "size": args.size,
                    "seed_formula": "generation_seed*1000000 + class_index*10000 + image_index",
                }
                ensure_manifest(output_dir, manifest, resume=args.resume)
                count = generate_condition(
                    pipe, prototypes, dcs, condition, generation_seed, output_dir, args
                )
                completion = output_dir / "complete.json"
                completion.write_text(json.dumps({"images": count}, indent=2) + "\n", encoding="utf-8")

        del pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
