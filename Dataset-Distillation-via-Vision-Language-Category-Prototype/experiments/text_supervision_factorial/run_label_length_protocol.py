#!/usr/bin/env python3
"""Run the six-cell Label/DCS conditioning-layout control matrix."""

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
PROMPTS = (
    "label",                 # a_pad77
    "label_pad_dcs",         # e: raw/padding-derived empty tail blocks
    "label_wrapped_dcs",     # e_wrapped: independently BOS-wrapped empty blocks
    "correct_t77",           # c
    "correct",               # d_pad
    "correct_tokenmax_var",  # d_var
)
OPTIONAL_PRIOR_PROMPTS = ("raw_label_tokenmax_var",)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--prototype", required=True)
    parser.add_argument("--dcs", required=True)
    parser.add_argument("--matched-model", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--generation-seeds", type=int, nargs="+", default=(0, 1, 2))
    parser.add_argument("--ipc", type=int, default=50)
    parser.add_argument("--strength", type=float, default=0.8)
    parser.add_argument("--classifier-repeats", type=int, default=2)
    parser.add_argument("--classifier-seed", type=int, default=0)
    parser.add_argument("--guidance-scale", type=float, default=10.0)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--retry-delay-seconds", type=int, default=120)
    parser.add_argument("--max-retries", type=int, default=0)
    parser.add_argument("--diffusers-src", default="/linxi/packages/VLCP/diffusers")
    parser.add_argument("--historical-strength-index", default="")
    return parser.parse_args()


def condition(prompt):
    return f"matched_ft_{prompt}"


def complete_generation(seed_root, expected):
    def check():
        for prompt in PROMPTS:
            path = seed_root / condition(prompt) / "complete.json"
            if not path.is_file() or int(json.loads(path.read_text(encoding="utf-8"))["images"]) != expected:
                return False
        return True
    return check


def generation_command(args, output, generation_seed):
    return [
        sys.executable, str(HERE / "generate_factorial.py"),
        "--prototype", args.prototype, "--dcs", args.dcs,
        "--base-model", args.base_model,
        "--model", f"matched_ft={args.matched_model}",
        "--supervisions", "matched_ft", "--prompts", *PROMPTS,
        "--output-root", str(output), "--generation-seeds", str(generation_seed),
        "--ipc", str(args.ipc), "--strength", str(args.strength),
        "--guidance-scale", str(args.guidance_scale),
        "--num-inference-steps", str(args.num_inference_steps),
        "--shuffle-shift", "1", "--size", "256", "--resume",
    ]


def build_tasks(args):
    root = Path(args.run_root).resolve()
    tasks, index = {}, []
    output = root / "synthetic"
    for generation_seed in sorted(set(args.generation_seeds)):
        seed_root = output / f"seed_{generation_seed}"
        gen_name = f"gen_g{generation_seed}"
        tasks[gen_name] = Task(
            gen_name, 1, "generate",
            generation_command(args, output, generation_seed),
            REPO_ROOT, root / "scheduler_logs" / f"{gen_name}.log",
            complete_generation(seed_root, 10 * args.ipc),
        )
        for prompt in PROMPTS:
            eval_name = f"eval_g{generation_seed}_{prompt}"
            log = root / "evaluation" / f"seed_{generation_seed}" / f"{prompt}.log"
            per_class = (
                root / "evaluation" / f"seed_{generation_seed}"
                / f"{prompt}.per_class.json"
            )
            command = eval_command(
                args, seed_root / condition(prompt), args.data_root,
                args.ipc, "nette", eval_name,
            )
            command.extend(("--per_class_output", str(per_class)))
            tasks[eval_name] = Task(
                eval_name, 1, "eval",
                command,
                EVAL_DIR, log,
                lambda log=log, per_class=per_class: (
                    complete_eval(log)() and per_class.is_file()
                ),
                dependencies=(gen_name,),
            )
            index.append({
                "experiment": "label_length_protocol", "spec": "nette",
                "ipc": args.ipc, "strength": args.strength,
                "checkpoint_family": "matched_ft", "training_seed": 0,
                "generation_seed": generation_seed, "prompt": prompt,
                "synthetic_dir": str(seed_root / condition(prompt)),
                "evaluation_log": str(log), "per_class_output": str(per_class),
            })
    return tasks, index


def main():
    args = parse_args()
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
    for path in (args.data_root, args.base_model, args.prototype, args.dcs, args.matched_model, args.diffusers_src):
        if not Path(path).exists():
            raise FileNotFoundError(path)
    root = Path(args.run_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = acquire_run_lock(root)
    tasks, index = build_tasks(args)
    if any(not task.complete() for task in tasks.values()):
        (root / "COMPLETE").unlink(missing_ok=True)
    manifest = {
        "format_version": 5,
        "experiment": "six_cell_conditioning_layout",
        "preregistered_primary": "wrapped_empty_minus_raw_empty",
        "protocol": {
            "spec": "nette", "ipc": args.ipc, "strength": args.strength,
            "checkpoint": str(Path(args.matched_model).resolve()),
            "generation_seeds": sorted(set(args.generation_seeds)),
            "classifier_repeats": args.classifier_repeats,
            "guidance_scale": args.guidance_scale,
            "num_inference_steps": args.num_inference_steps,
            "negative_prompt": "cartoon, anime, painting",
            "paired_noise_across_prompts": True,
            "label": "raw class string padded/truncated to [B,77,768]",
            "label_pad_dcs": (
                "raw Label head plus padding-derived empty tail blocks, with the number "
                "of blocks matched independently to the paired Correct DCS caption"
            ),
            "label_wrapped_dcs": (
                "raw Label head plus independently encoded empty-string tail blocks; "
                "every appended block restarts CLIP positions as [BOS, EOS, PAD...]"
            ),
            "correct_t77": (
                "the paired cluster-specific Correct DCS caption truncated/padded to one "
                "77-position SD1.5 CLIP block"
            ),
            "correct_dcs_pad77_chunks": (
                "full Correct DCS rounded up to 77-position blocks; both CFG branches "
                "share the padded multiple-of-77 sequence length"
            ),
            "correct_dcs_tokenmax_var": (
                "full Correct DCS at max(actual positive CLIP tokens, actual negative "
                "CLIP tokens), without rounding to a 77-position multiple"
            ),
            "official_whitespace_heuristic": (
                "audited only; never used for generation because it can select the "
                "token-shorter CFG branch"
            ),
            "diffusers_src": str(Path(args.diffusers_src).resolve()),
            "matrix_prompts": list(PROMPTS),
            "optional_prior_control": list(OPTIONAL_PRIOR_PROMPTS),
        },
    }
    manifest_path = root / "run_manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        legacy_v4 = {
            "format_version": 4,
            "experiment": "label_length_protocol",
            "preregistered_primary": "raw_label_tokenmax_var_minus_raw_label_pad77",
            "protocol": {
                key: value for key, value in manifest["protocol"].items()
                if key not in {"label_pad_dcs", "label_wrapped_dcs", "matrix_prompts", "optional_prior_control"}
            },
        }
        legacy_v4["protocol"]["raw_label_tokenmax_var"] = (
            "raw class string with both CFG branches independently tokenized, then "
            "padded to max(positive CLIP tokens, negative CLIP tokens); deterministic "
            "per prompt and independent of batch composition"
        )
        legacy_v3 = json.loads(json.dumps(legacy_v4))
        legacy_v3["format_version"] = 3
        legacy_v3["protocol"].pop("correct_dcs_pad77_chunks")
        legacy_v3["protocol"].pop("correct_dcs_tokenmax_var")
        legacy_v2 = json.loads(json.dumps(legacy_v3))
        legacy_v2["format_version"] = 2
        legacy_v2["protocol"].pop("correct_t77")
        if existing not in (manifest, legacy_v4, legacy_v3, legacy_v2):
            raise RuntimeError(f"Resume configuration differs from {manifest_path}")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    index_path = root / "evaluation_index.json"
    if index_path.is_file():
        prior_index = json.loads(index_path.read_text(encoding="utf-8"))
        index.extend(
            row for row in prior_index
            if row.get("prompt") in OPTIONAL_PRIOR_PROMPTS
        )
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    append_scheduler_event(root, "scheduler_start_requested", experiment="label_length_protocol")
    run_scheduler(args, tasks)
    subprocess.run([
        sys.executable, str(HERE / "summarize_label_length_protocol.py"),
        "--evaluation-index", str(index_path), "--output-dir", str(root / "summary"),
    ], cwd=REPO_ROOT, check=True)
    if args.historical_strength_index:
        historical_index = Path(args.historical_strength_index).resolve()
        if not historical_index.is_file():
            raise FileNotFoundError(historical_index)
        subprocess.run([
            sys.executable, str(HERE / "audit_label_reproducibility.py"),
            "--current-index", str(index_path),
            "--historical-index", str(historical_index),
            "--output-dir", str(root / "summary" / "historical_reproducibility"),
        ], cwd=REPO_ROOT, check=True)
    (root / "COMPLETE").write_text(time.strftime("%F %T\n"), encoding="utf-8")
    lock.close()


if __name__ == "__main__":
    main()
