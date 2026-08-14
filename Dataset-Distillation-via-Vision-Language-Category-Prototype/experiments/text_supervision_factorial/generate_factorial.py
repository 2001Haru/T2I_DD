import argparse
import gc
import json
import re
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

GENERATION_SUPERVISION_MODES = SUPERVISION_MODES + ("sparse_ft",)
GENERATION_PROMPT_MODES = PROMPT_MODES + (
    "bank",
    "label_var",
    "first_sentence",
    "correct_t77",
    "label_pad_dcs",
    "correct_t77_pad_dcs",
    "correct_head_pad_dcs",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a text-supervision x inference-prompt factorial")
    parser.add_argument("--prototype", required=True)
    parser.add_argument("--dcs", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--model", action="append", default=[], help="MODE=DIFFUSERS_PATH")
    parser.add_argument("--supervisions", nargs="+", choices=GENERATION_SUPERVISION_MODES, default=SUPERVISION_MODES)
    parser.add_argument("--prompts", nargs="+", choices=GENERATION_PROMPT_MODES, default=PROMPT_MODES)
    parser.add_argument("--prompt-bank", default=None)
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
    parser.add_argument(
        "--guidance-diagnostic-timesteps", type=int, nargs="+", default=(),
        help="Requested training timesteps for paired CFG branch-norm diagnostics",
    )
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
        if mode not in GENERATION_SUPERVISION_MODES or mode == "frozen":
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


def text_chunk_count(tokenizer, text):
    chunk = tokenizer.model_max_length
    token_count = tokenizer(text, return_tensors="pt", truncation=False).input_ids.shape[-1]
    return max(1, (token_count + chunk - 1) // chunk)


def official_exact_length(tokenizer, prompt, negative_prompt):
    """Return the shared token length selected by VLCP's official helper."""
    longer = prompt if len(prompt.split(" ")) >= len(negative_prompt.split(" ")) else negative_prompt
    return tokenizer(longer, return_tensors="pt", truncation=False).input_ids.shape[-1]


def first_sentence(text):
    text = str(text).strip()
    match = re.match(r"^.*?[.!?](?=\s|$)", text)
    return match.group(0).strip() if match else text


def get_pipeline_embeds(
    pipe,
    prompt,
    negative_prompt,
    device,
    policy="chunked",
    target_chunks=None,
):
    chunk = pipe.tokenizer.model_max_length
    if policy == "official_exact":
        # Match VLCP's get_pipeline_embeds implementation, including its word-count
        # branch and a final raw slice that may be shorter than model_max_length.
        count_prompt = len(prompt.split(" "))
        count_negative = len(negative_prompt.split(" "))
        if count_prompt >= count_negative:
            prompt_ids = pipe.tokenizer(
                prompt, return_tensors="pt", truncation=False,
            ).input_ids
            length = prompt_ids.shape[-1]
            negative_ids = pipe.tokenizer(
                negative_prompt, padding="max_length", max_length=length,
                truncation=False, return_tensors="pt",
            ).input_ids
        else:
            negative_ids = pipe.tokenizer(
                negative_prompt, return_tensors="pt", truncation=False,
            ).input_ids
            length = negative_ids.shape[-1]
            prompt_ids = pipe.tokenizer(
                prompt, padding="max_length", max_length=length,
                truncation=False, return_tensors="pt",
            ).input_ids
        prompt_ids = prompt_ids.to(device)
        negative_ids = negative_ids.to(device)
        prompt_embeds = [
            pipe.text_encoder(prompt_ids[:, start:start + chunk])[0]
            for start in range(0, length, chunk)
        ]
        negative_embeds = [
            pipe.text_encoder(negative_ids[:, start:start + chunk])[0]
            for start in range(0, length, chunk)
        ]
        return torch.cat(prompt_embeds, dim=1), torch.cat(negative_embeds, dim=1)

    if policy == "single":
        prompt_ids = pipe.tokenizer(
            prompt, padding="max_length", max_length=chunk, truncation=True,
            return_tensors="pt",
        ).input_ids.to(device)
        negative_ids = pipe.tokenizer(
            negative_prompt, padding="max_length", max_length=chunk, truncation=True,
            return_tensors="pt",
        ).input_ids.to(device)
        return pipe.text_encoder(prompt_ids)[0], pipe.text_encoder(negative_ids)[0]

    if policy in {"pad_extended", "single_pad_extended"}:
        if target_chunks is None or target_chunks < 1:
            raise ValueError(f"{policy} requires target_chunks >= 1")
        prompt_embeds, negative_embeds = get_pipeline_embeds(
            pipe, prompt, negative_prompt, device, policy="single"
        )
        if target_chunks == 1:
            return prompt_embeds, negative_embeds
        length = target_chunks * chunk
        negative_ids = pipe.tokenizer(
            negative_prompt, padding="max_length", max_length=length, truncation=False,
            return_tensors="pt",
        ).input_ids.to(device)
        negative_tail = [
            pipe.text_encoder(negative_ids[:, start:start + chunk])[0]
            for start in range(chunk, length, chunk)
        ]
        return (
            torch.cat([prompt_embeds, *negative_tail], dim=1),
            torch.cat([negative_embeds, *negative_tail], dim=1),
        )

    if policy == "chunk_head_pad_extended":
        if target_chunks is None or target_chunks < 1:
            raise ValueError("chunk_head_pad_extended requires target_chunks >= 1")
        length = target_chunks * chunk
        prompt_ids = pipe.tokenizer(
            prompt, padding="max_length", max_length=length, truncation=False,
            return_tensors="pt",
        ).input_ids.to(device)
        negative_ids = pipe.tokenizer(
            negative_prompt, padding="max_length", max_length=length, truncation=False,
            return_tensors="pt",
        ).input_ids.to(device)
        prompt_head = pipe.text_encoder(prompt_ids[:, :chunk])[0]
        negative_chunks = [
            pipe.text_encoder(negative_ids[:, start:start + chunk])[0]
            for start in range(0, length, chunk)
        ]
        return (
            torch.cat([prompt_head, *negative_chunks[1:]], dim=1),
            torch.cat(negative_chunks, dim=1),
        )

    if policy != "chunked":
        raise ValueError(f"Unknown conditioning policy: {policy}")
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


def prompt_for(synset, index, image_index, mode, dcs, shift, prompt_bank):
    correct = str(dcs[synset][index])
    if mode == "label":
        return IMAGENET2012_CLASSES[synset], None, None, "chunked", correct
    if mode == "label_var":
        return IMAGENET2012_CLASSES[synset], None, None, "official_exact", correct
    if mode == "first_sentence":
        return first_sentence(correct), index, None, "single", correct
    if mode == "correct_t77":
        return correct, index, None, "single", correct
    if mode == "label_pad_dcs":
        return IMAGENET2012_CLASSES[synset], None, None, "pad_extended", correct
    if mode == "correct_t77_pad_dcs":
        return correct, index, None, "single_pad_extended", correct
    if mode == "correct_head_pad_dcs":
        return correct, index, None, "chunk_head_pad_extended", correct
    if mode == "bank":
        entries = prompt_bank["classes"][synset]
        source = image_index % len(entries)
        return (
            str(entries[source]["caption"]), source, entries[source]["relative"],
            "chunked", correct,
        )
    source = index if mode == "correct" else shuffled_prompt_index(index, len(dcs[synset]), shift)
    return str(dcs[synset][source]), source, None, "chunked", correct


def schedule_matched_noise(prototype_data, image_seed, device, dtype=torch.float16):
    """Return an independent N(0, I) latent without consuming diffusion RNG state."""
    shape = tuple(torch.as_tensor(prototype_data).shape)
    generator = torch.Generator(device="cpu").manual_seed(
        (int(image_seed) + 2_000_000_011) % (2**63 - 1)
    )
    return torch.randn(shape, generator=generator, dtype=torch.float32).to(
        device=device, dtype=dtype
    ).unsqueeze(0)


def planned_guidance_timesteps(pipe, args):
    """Map requested training timesteps to the nearest timestep this call executes."""
    if not args.guidance_diagnostic_timesteps:
        return {}
    pipe.scheduler.set_timesteps(args.num_inference_steps, device=args.device)
    timesteps = pipe.scheduler.timesteps
    if args.visual_mode in {"prototype", "schedule_matched_noise"}:
        init_steps = min(int(args.num_inference_steps * args.strength), args.num_inference_steps)
        start = max(args.num_inference_steps - init_steps, 0) * getattr(pipe.scheduler, "order", 1)
        timesteps = timesteps[start:]
    actual = [int(value.item()) for value in timesteps]
    if not actual:
        raise RuntimeError("No diffusion timestep is executed by the requested visual schedule")
    return {
        requested: min(actual, key=lambda value: (abs(value - requested), value))
        for requested in args.guidance_diagnostic_timesteps
    }


def epsilon_branch_metrics(sample):
    """Return independent conditional, unconditional, and CFG-residual magnitudes."""
    if sample.ndim < 2 or sample.shape[0] % 2:
        raise RuntimeError(f"Expected an even CFG batch, got {tuple(sample.shape)}")
    unconditional, conditional = sample.float().chunk(2)

    def magnitudes(value):
        flattened = value.flatten(1)
        return {
            "l2": torch.linalg.vector_norm(flattened, dim=1).mean().item(),
            "rms": flattened.square().mean(dim=1).sqrt().mean().item(),
        }

    residual = conditional - unconditional
    return {
        "epsilon_cond": magnitudes(conditional),
        "epsilon_uncond": magnitudes(unconditional),
        "epsilon_residual": magnitudes(residual),
    }


def run_with_guidance_diagnostics(pipe, call_args, requested_to_actual):
    diagnostics = []
    actual_to_requested = {}
    for requested, actual in requested_to_actual.items():
        actual_to_requested.setdefault(actual, []).append(requested)
    captured = set()

    def hook(_module, inputs, output):
        if len(inputs) < 2:
            return
        timestep = inputs[1]
        actual = int(timestep.flatten()[0].item()) if torch.is_tensor(timestep) else int(timestep)
        if actual not in actual_to_requested or actual in captured:
            return
        sample = output.sample if hasattr(output, "sample") else output[0]
        metrics = epsilon_branch_metrics(sample)
        for requested in actual_to_requested[actual]:
            diagnostics.append({
                "requested_timestep": requested,
                "actual_timestep": actual,
                "absolute_timestep_error": abs(actual - requested),
                "exact_requested_timestep": actual == requested,
                **{
                    f"{branch}_{magnitude}": value
                    for branch, values in metrics.items()
                    for magnitude, value in values.items()
                },
            })
        captured.add(actual)

    handle = pipe.unet.register_forward_hook(hook)
    try:
        result = pipe(**call_args)
    finally:
        handle.remove()
    missing = sorted(set(requested_to_actual.values()) - captured)
    if missing:
        raise RuntimeError(f"UNet hook missed planned diagnostic timesteps: {missing}")
    return result, sorted(diagnostics, key=lambda row: row["requested_timestep"])


def generate(pipe, prototypes, dcs, prompt_bank, supervision, prompt_mode, generation_seed, output_dir, args):
    records = []
    completed = 0
    requested_to_actual = planned_guidance_timesteps(pipe, args)
    for class_index, (synset, values) in enumerate(prototypes.items()):
        class_dir = output_dir / synset
        class_dir.mkdir(parents=True, exist_ok=True)
        repeats = args.ipc // len(values)
        for repetition in range(repeats):
            for prototype_index, prototype_data in enumerate(values):
                image_index = repetition * len(values) + prototype_index
                prompt, source, source_relative, embedding_policy, reference_dcs = prompt_for(
                    synset, prototype_index, image_index, prompt_mode, dcs,
                    args.shuffle_shift, prompt_bank
                )
                reference_dcs_chunks = text_chunk_count(pipe.tokenizer, reference_dcs)
                conditioning_chunks = (
                    reference_dcs_chunks
                    if embedding_policy in {
                        "pad_extended", "single_pad_extended", "chunk_head_pad_extended"
                    }
                    else (1 if embedding_policy == "single" else text_chunk_count(pipe.tokenizer, prompt))
                )
                conditioning_sequence_length = (
                    official_exact_length(pipe.tokenizer, prompt, args.negative_prompt)
                    if embedding_policy == "official_exact"
                    else conditioning_chunks * pipe.tokenizer.model_max_length
                )
                image_seed = stable_image_seed(generation_seed, class_index, image_index)
                record = {
                    "synset": synset,
                    "image_index": image_index,
                    "prototype_index": prototype_index,
                    "prompt_source_index": source,
                    "prompt_source_relative": source_relative,
                    "prompt": prompt,
                    "embedding_policy": embedding_policy,
                    "conditioning_chunks": conditioning_chunks,
                    "conditioning_sequence_length": conditioning_sequence_length,
                    "reference_dcs": reference_dcs,
                    "reference_dcs_chunks": reference_dcs_chunks,
                    "chunk_count_unit": "individual_prompt_not_batch_max",
                    "image_seed": image_seed,
                }
                destination = class_dir / f"image_{image_index:05d}.png"
                diagnostic_path = class_dir / f"image_{image_index:05d}.guidance.json"
                if destination.is_file() and (
                    not requested_to_actual or diagnostic_path.is_file()
                ):
                    if requested_to_actual:
                        record["guidance_diagnostics"] = json.loads(
                            diagnostic_path.read_text(encoding="utf-8")
                        )["guidance_diagnostics"]
                    records.append(record)
                    completed += 1
                    continue
                prompt_embeds, negative_embeds = get_pipeline_embeds(
                    pipe,
                    prompt,
                    args.negative_prompt,
                    args.device,
                    policy=embedding_policy,
                    target_chunks=conditioning_chunks,
                )
                if prompt_embeds.shape[1] != negative_embeds.shape[1]:
                    raise RuntimeError("Positive and negative conditioning lengths differ")
                if prompt_embeds.shape[1] != conditioning_sequence_length:
                    raise RuntimeError("Conditioning sequence length does not match its manifest record")
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
                if requested_to_actual:
                    result, diagnostics = run_with_guidance_diagnostics(
                        pipe, call_args, requested_to_actual
                    )
                    record["guidance_diagnostics"] = diagnostics
                    atomic_write_json(diagnostic_path, {
                        "requested_to_actual_timestep": requested_to_actual,
                        "guidance_diagnostics": diagnostics,
                    })
                else:
                    result = pipe(**call_args)
                image = result.images[0]
                image.resize((args.size, args.size)).save(destination)
                records.append(record)
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
    prompt_bank = load_json(Path(args.prompt_bank).resolve()) if args.prompt_bank else None
    if "bank" in args.prompts:
        if prompt_bank is None:
            raise ValueError("The bank prompt mode requires --prompt-bank")
        if set(prompt_bank.get("classes", {})) != set(prototypes):
            raise ValueError("Prompt-bank and prototype classes differ")
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
                condition = (
                    condition_name(supervision, prompt_mode)
                    if supervision in SUPERVISION_MODES and prompt_mode in PROMPT_MODES
                    else f"{supervision}_{prompt_mode}"
                )
                output_dir = output_root / f"seed_{generation_seed}" / condition
                manifest = {
                    "format_version": 3,
                    "condition": condition,
                    "supervision_mode": supervision,
                    "prompt_mode": prompt_mode,
                    "checkpoint": checkpoint_identity(checkpoint),
                    "prototype_path": str(prototype_path),
                    "prototype_sha256": sha256_file(prototype_path),
                    "dcs_path": str(dcs_path),
                    "dcs_sha256": sha256_file(dcs_path),
                    "prompt_bank_path": str(Path(args.prompt_bank).resolve()) if args.prompt_bank else None,
                    "prompt_bank_sha256": sha256_file(args.prompt_bank) if args.prompt_bank else None,
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
                    "guidance_diagnostic_timesteps": list(args.guidance_diagnostic_timesteps),
                    "guidance_diagnostic_definition": (
                        "At the nearest actually executed scheduler timestep to each requested "
                        "training timestep, record separate conditional, unconditional, and "
                        "conditional-minus-unconditional UNet output L2 and RMS magnitudes."
                        if args.guidance_diagnostic_timesteps else None
                    ),
                    "negative_prompt": args.negative_prompt,
                    "shuffle_shift": args.shuffle_shift,
                    "paired_noise_across_all_cells": True,
                    "conditioning_length_control": (
                        "Length-matched controls append padding-derived blocks along sequence "
                        "dimension to match paired correct-DCS chunk count. correct_head_pad_dcs "
                        "uses the exact raw-sliced first block from full DCS and replaces only "
                        "later positive blocks with the corresponding negative-tail blocks."
                        if prompt_mode in {
                            "label_pad_dcs", "correct_t77_pad_dcs", "correct_head_pad_dcs"
                        }
                        else None
                    ),
                    "chunk_count_unit": (
                        "computed independently for each sampled caption before its one-image "
                        "pipeline call; never derived from a batch maximum"
                    ),
                }
                ensure_manifest(output_dir, manifest, resume=args.resume)
                records, count = generate(
                    pipe, prototypes, dcs, prompt_bank, supervision, prompt_mode,
                    generation_seed, output_dir, args
                )
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
