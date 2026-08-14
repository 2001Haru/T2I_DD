#!/usr/bin/env python3
"""Run B-0: semantic content versus cross-attention sequence length."""

import argparse
import json
import signal
import subprocess
import sys
import time
from pathlib import Path

from run_generality import EVAL_DIR, REPO_ROOT, Task, complete_eval, eval_command
from run_sparse_interface_transfer import acquire_run_lock, append_scheduler_event, run_scheduler


HERE = Path(__file__).resolve().parent
FAMILIES = ("label_ft", "matched_ft", "sparse_m4_ft")
MATCHED_PROMPTS = (
    "label",
    "first_sentence",
    "correct_t77",
    "correct",
    "label_pad_dcs",
    "correct_t77_pad_dcs",
    "correct_head_pad_dcs",
)
CONTROL_PROMPTS = ("label", "correct_t77", "correct", "correct_head_pad_dcs")
FAMILY_PROMPTS = {
    "label_ft": CONTROL_PROMPTS,
    "matched_ft": MATCHED_PROMPTS,
    "sparse_m4_ft": CONTROL_PROMPTS,
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--prototype", required=True)
    parser.add_argument("--dcs", required=True)
    parser.add_argument("--label-model", required=True)
    parser.add_argument("--matched-model", required=True)
    parser.add_argument("--sparse-model", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--matched-generation-seeds", type=int, nargs="+", default=(0, 1, 2, 3, 4))
    parser.add_argument("--control-generation-seeds", type=int, nargs="+", default=(0, 1, 2))
    parser.add_argument("--ipc", type=int, default=10)
    parser.add_argument("--strength", type=float, default=0.7)
    parser.add_argument("--classifier-repeats", type=int, default=2)
    parser.add_argument("--classifier-seed", type=int, default=0)
    parser.add_argument("--guidance-scale", type=float, default=10.0)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-diagnostic-timesteps", type=int, nargs="+", default=(200, 500, 800))
    parser.add_argument("--retry-delay-seconds", type=int, default=120)
    parser.add_argument("--max-retries", type=int, default=0)
    parser.add_argument("--diffusers-src", default="")
    return parser.parse_args()


def family_config(args, family):
    return {
        "label_ft": ("label_ft", Path(args.label_model).resolve()),
        "matched_ft": ("matched_ft", Path(args.matched_model).resolve()),
        "sparse_m4_ft": ("sparse_ft", Path(args.sparse_model).resolve()),
    }[family]


def condition(supervision, prompt):
    return f"{supervision}_{prompt}"


def complete_generation(seed_root, supervision, prompts, expected):
    def check():
        for prompt in prompts:
            path = seed_root / condition(supervision, prompt) / "complete.json"
            if not path.is_file():
                return False
            if int(json.loads(path.read_text(encoding="utf-8"))["images"]) != expected:
                return False
        return True
    return check


def generation_command(args, supervision, model, output, generation_seed, prompts):
    return [
        sys.executable, str(HERE / "generate_factorial.py"),
        "--prototype", args.prototype, "--dcs", args.dcs,
        "--base-model", args.base_model, "--model", f"{supervision}={model}",
        "--supervisions", supervision, "--prompts", *prompts,
        "--output-root", str(output), "--generation-seeds", str(generation_seed),
        "--ipc", str(args.ipc), "--strength", str(args.strength),
        "--guidance-scale", str(args.guidance_scale),
        "--num-inference-steps", str(args.num_inference_steps),
        "--guidance-diagnostic-timesteps", *map(str, args.guidance_diagnostic_timesteps),
        "--shuffle-shift", "1", "--size", "256", "--resume",
    ]


def build_tasks(args):
    root = Path(args.run_root).resolve()
    tasks, index = {}, []
    for family in FAMILIES:
        supervision, model = family_config(args, family)
        prompts = FAMILY_PROMPTS[family]
        generation_seeds = (
            args.matched_generation_seeds
            if family == "matched_ft"
            else args.control_generation_seeds
        )
        for generation_seed in sorted(set(generation_seeds)):
            token = f"{family}_g{generation_seed}"
            output = root / "synthetic" / family
            seed_root = output / f"seed_{generation_seed}"
            gen_name = f"gen_{token}"
            tasks[gen_name] = Task(
                gen_name, 1, "generate",
                generation_command(args, supervision, model, output, generation_seed, prompts),
                REPO_ROOT, root / "scheduler_logs" / f"{gen_name}.log",
                complete_generation(seed_root, supervision, prompts, 10 * args.ipc),
            )
            for prompt in prompts:
                eval_name = f"eval_{token}_{prompt}"
                log = root / "evaluation" / family / f"seed_{generation_seed}" / f"{prompt}.log"
                tasks[eval_name] = Task(
                    eval_name, 1, "eval",
                    eval_command(
                        args, seed_root / condition(supervision, prompt), args.data_root,
                        args.ipc, "nette", eval_name,
                    ),
                    EVAL_DIR, log, complete_eval(log), dependencies=(gen_name,),
                )
                index.append({
                    "experiment": "b0_conditioning_length", "spec": "nette",
                    "ipc": args.ipc, "strength": args.strength,
                    "checkpoint_family": family, "training_seed": 0,
                    "generation_seed": generation_seed, "prompt": prompt,
                    "evaluation_log": str(log),
                })
    return tasks, index


def main():
    args = parse_args()
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
    for path in (
        args.data_root, args.base_model, args.prototype, args.dcs,
        args.label_model, args.matched_model, args.sparse_model,
    ):
        if not Path(path).exists():
            raise FileNotFoundError(path)
    root = Path(args.run_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    run_lock = acquire_run_lock(root)
    tasks, index = build_tasks(args)
    manifest = {
        "format_version": 3,
        "experiment": "b0_conditioning_length",
        "causal_question": "Does DCS utility come from text content or extra cross-attention positions?",
        "data_root": str(Path(args.data_root).resolve()),
        "base_model": str(Path(args.base_model).resolve()),
        "prototype": str(Path(args.prototype).resolve()),
        "dcs": str(Path(args.dcs).resolve()),
        "models": {
            family: str(family_config(args, family)[1]) for family in FAMILIES
        },
        "generation_seeds": {
            "matched_ft": sorted(set(args.matched_generation_seeds)),
            "controls": sorted(set(args.control_generation_seeds)),
        },
        "prompts": {family: list(FAMILY_PROMPTS[family]) for family in FAMILIES},
        "ipc": args.ipc, "strength": args.strength,
        "guidance_scale": args.guidance_scale,
        "num_inference_steps": args.num_inference_steps,
        "guidance_diagnostic_timesteps": list(args.guidance_diagnostic_timesteps),
        "guidance_diagnostic_timestep_policy": (
            "Record requested t and nearest actually executed scheduler t separately; never "
            "label a nearest img2img step as an exact requested timestep."
        ),
        "classifier_repeats": args.classifier_repeats,
        "chunk_count_unit": (
            "per sampled caption and visual slot; generation uses one image per pipeline call, "
            "so batch composition cannot change sequence length"
        ),
        "power_allocation": {
            "matched_ft": "all seven conditions with five generation seeds",
            "label_ft_and_sparse_m4_ft": "a/c/d/g only with three generation seeds",
        },
        "conditioning_definitions": {
            "label": "one ordinary 77-position class-label block",
            "first_sentence": "first DCS sentence truncated/padded to one 77-position block",
            "correct_t77": "full DCS text truncated/padded to its first 77-position block",
            "correct": "existing full DCS, encoded in 77-position chunks and concatenated on sequence dimension",
            "label_pad_dcs": "label block plus the paired long-DCS negative-tail blocks, with per-caption chunk count",
            "correct_t77_pad_dcs": "training-style T77 DCS block plus paired negative-tail blocks; completes the content x length 2x2",
            "correct_head_pad_dcs": "exact raw-sliced first block of full DCS plus its negative-tail blocks; differs from full DCS only in positive tail content",
        },
        "preregistered_estimands": {
            "correct_t77_minus_label": "full population; both interventions always use one chunk",
            "label_pad_dcs_minus_label": "long-slot partial population; e equals a when n=1",
            "full_minus_head_pad_dcs": "long-slot partial population; d equals g when n=1",
            "head_pad_dcs_minus_t77_pad_dcs": "long-slot partial population",
            "full_minus_t77_pad_dcs": "long-slot partial population",
            "length_x_chunk1_content_interaction": "long-slot partial population",
        },
    }
    manifest_path = root / "run_manifest.json"
    if manifest_path.is_file() and json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
        raise RuntimeError(f"Resume configuration differs from {manifest_path}")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    index_path = root / "evaluation_index.json"
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    append_scheduler_event(root, "scheduler_start_requested", experiment="b0_conditioning_length")
    run_scheduler(args, tasks)
    subprocess.run([
        sys.executable, str(HERE / "summarize_b0_conditioning_length.py"),
        "--evaluation-index", str(index_path), "--output-dir", str(root / "summary"),
    ], cwd=REPO_ROOT, check=True)
    (root / "COMPLETE").write_text(time.strftime("%F %T\n"), encoding="utf-8")
    run_lock.close()


if __name__ == "__main__":
    main()
