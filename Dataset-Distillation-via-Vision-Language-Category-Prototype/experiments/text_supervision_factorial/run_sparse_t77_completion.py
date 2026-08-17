#!/usr/bin/env python3
"""Schedule low-budget Sparse-T77 and fixed-checkpoint DCS-T77 controls."""

import argparse
import ast
import csv
import json
import random
import re
import signal
import statistics
import subprocess
import sys
import time
from argparse import Namespace
from pathlib import Path

from build_sparse_caption_banks import build_banks
from run_sparse_prompt_search import (
    EVAL_DIR,
    REPO_ROOT,
    Task,
    build_tasks as build_sparse_tasks,
    complete_eval,
    complete_model,
    eval_command,
    launch,
    write_state,
)


HERE = Path(__file__).resolve().parent
RESULT = re.compile(r"Best, last acc:----(\[[^\]]+\])")
FIXED_FAMILIES = ("matched_ft", "unpaired_ft")
FIXED_PROMPTS = ("correct_t77", "shuffled_t77")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--caption-file", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--prototype", required=True)
    parser.add_argument("--dcs", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--generation-seeds", type=int, nargs="+", default=(0, 1))
    parser.add_argument("--dense-training-seeds", type=int, nargs="+", default=(0, 1))
    parser.add_argument("--ipc", type=int, default=50)
    parser.add_argument("--strength", type=float, default=0.8)
    parser.add_argument("--classifier-repeats", type=int, default=2)
    parser.add_argument("--classifier-seed", type=int, default=0)
    parser.add_argument("--retry-delay-seconds", type=int, default=120)
    parser.add_argument("--max-retries", type=int, default=0)
    parser.add_argument("--diffusers-src", default="")
    return parser.parse_args()


def fixed_generation_complete(seed_root, family, expected):
    def check():
        for prompt in FIXED_PROMPTS:
            complete = seed_root / f"{family}_{prompt}" / "complete.json"
            if not complete.is_file():
                return False
            if int(json.loads(complete.read_text(encoding="utf-8"))["images"]) != expected:
                return False
        return True
    return check


def fixed_generation_command(args, family, model, output, generation_seed):
    return [
        sys.executable, str(HERE / "generate_factorial.py"),
        "--prototype", args.prototype, "--dcs", args.dcs,
        "--base-model", args.base_model, "--model", f"{family}={model}",
        "--supervisions", family, "--prompts", *FIXED_PROMPTS,
        "--output-root", str(output), "--generation-seeds", str(generation_seed),
        "--ipc", str(args.ipc), "--strength", str(args.strength),
        "--guidance-scale", "10", "--num-inference-steps", "50",
        "--shuffle-shift", "1", "--size", "256", "--resume",
    ]


def dense_train_command(args, family, output, training_seed):
    supervision = {"matched_ft": "matched", "unpaired_ft": "unpaired"}[family]
    return [
        "accelerate", "launch", "--num_processes", "1", "--num_machines", "1",
        "--mixed_precision", "fp16", "--dynamo_backend", "no",
        str(HERE / "train_text_to_image_supervision.py"),
        "--pretrained-model", args.base_model,
        "--train-root", str(Path(args.data_root) / "train"),
        "--caption-file", args.caption_file, "--output-dir", str(output),
        "--supervision", supervision, "--resolution", "512",
        "--train-batch-size", "8", "--gradient-accumulation-steps", "4",
        "--num-train-epochs", "8", "--learning-rate", "1e-5",
        "--lr-scheduler", "constant", "--lr-warmup-steps", "0",
        "--max-grad-norm", "1", "--mixed-precision", "fp16",
        "--seed", str(training_seed),
        "--num-workers", "2", "--checkpointing-steps", "500",
        "--checkpoints-total-limit", "2", "--loss-log-steps", "50",
        "--timestep-bins", "10", "--random-flip", "--gradient-checkpointing",
        "--use-ema",
    ]


def sparse_namespace(args, sparse_root):
    return Namespace(
        run_root=str(sparse_root), bank_seeds=(0, 1), budgets=(4, 8),
        generation_seeds=tuple(args.generation_seeds), training_seed=0,
        data_root=args.data_root, caption_file=args.caption_file,
        base_model=args.base_model, prototype=args.prototype, dcs=args.dcs,
        ipc=args.ipc, strength=args.strength,
        classifier_repeats=args.classifier_repeats,
        classifier_seed=args.classifier_seed, train_batch_size=8,
        gradient_accumulation_steps=4, num_workers=2,
        mixed_precision="fp16", prompts=("label", "bank_t77"), spec="nette",
    )


def build_tasks(args):
    root = Path(args.run_root).resolve()
    sparse_root = root / "sparse"
    sparse_args = sparse_namespace(args, sparse_root)
    build_banks(
        Path(args.data_root) / "train", args.caption_file,
        sparse_root / "caption_banks", sparse_args.budgets, sparse_args.bank_seeds,
    )
    tasks, sparse_index = build_sparse_tasks(sparse_args)
    fixed_index = []
    for family in FIXED_FAMILIES:
        for training_seed in getattr(args, "dense_training_seeds", (0, 1)):
            model = root / "fixed" / "models" / f"train_seed_{training_seed}" / family
            train_name = f"train_{family}_s{training_seed}_t77_official_batch"
            tasks[train_name] = Task(
                train_name, 1, "train",
                dense_train_command(args, family, model, training_seed),
                REPO_ROOT, root / "scheduler_logs" / f"{train_name}.log",
                complete_model(model),
            )
            for generation_seed in args.generation_seeds:
                output = (
                    root / "fixed" / "synthetic" / f"train_seed_{training_seed}" / family
                )
                seed_root = output / f"seed_{generation_seed}"
                generation_name = (
                    f"gen_{family}_s{training_seed}_g{generation_seed}_t77"
                )
                tasks[generation_name] = Task(
                    generation_name, 2, "generate",
                    fixed_generation_command(args, family, model, output, generation_seed),
                    REPO_ROOT, root / "scheduler_logs" / f"{generation_name}.log",
                    fixed_generation_complete(seed_root, family, 10 * args.ipc),
                    dependencies=(train_name,),
                )
                for prompt in FIXED_PROMPTS:
                    condition = f"{family}_{prompt}"
                    evaluation_name = (
                        f"eval_{family}_s{training_seed}_g{generation_seed}_{prompt}"
                    )
                    evaluation_log = (
                        root / "fixed" / "evaluation" / f"train_seed_{training_seed}"
                        / family / f"seed_{generation_seed}" / f"{condition}.log"
                    )
                    tasks[evaluation_name] = Task(
                        evaluation_name, 2, "eval",
                        eval_command(args, seed_root / condition, evaluation_name),
                        EVAL_DIR, evaluation_log, complete_eval(evaluation_log),
                        dependencies=(generation_name,),
                    )
                    fixed_index.append({
                        "method": "refit_dense_checkpoint_t77",
                        "checkpoint_family": family,
                        "training_seed": int(training_seed),
                        "generation_seed": int(generation_seed), "prompt": prompt,
                        "ipc": args.ipc, "strength": args.strength,
                        "synthetic_dir": str(seed_root / condition),
                        "evaluation_log": str(evaluation_log),
                    })
    return tasks, sparse_index, fixed_index


def scores(path):
    matches = RESULT.findall(Path(path).read_text(encoding="utf-8", errors="replace"))
    if not matches:
        raise ValueError(f"No completed classifier result: {path}")
    return [float(value) for value in ast.literal_eval(matches[-1])]


def percentile(values, probability):
    values = sorted(values)
    return values[min(len(values) - 1, int(probability * len(values)))]


def bootstrap_fixed(rows, contrast=False, samples=10000, seed=20260818):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["training_seed"], {}).setdefault(
            row["generation_seed"], {}
        )[row["prompt"]] = scores(row["evaluation_log"])
    training_seeds = sorted(grouped)
    target_prompt = rows[0]["prompt"] if not contrast else None
    rng = random.Random(seed)

    def estimate(draw_rng=None):
        selected_training = training_seeds if draw_rng is None else [
            draw_rng.choice(training_seeds) for _ in training_seeds
        ]
        values = []
        for training_seed in selected_training:
            generations = grouped[training_seed]
            generation_seeds = sorted(generations)
            selected_generation = generation_seeds if draw_rng is None else [
                draw_rng.choice(generation_seeds) for _ in generation_seeds
            ]
            for generation_seed in selected_generation:
                cell = generations[generation_seed]
                repeat_count = len(cell["correct_t77" if contrast else target_prompt])
                repeat_indices = range(repeat_count) if draw_rng is None else [
                    draw_rng.randrange(repeat_count) for _ in range(repeat_count)
                ]
                for repeat_index in repeat_indices:
                    if contrast:
                        values.append(
                            cell["correct_t77"][repeat_index]
                            - cell["shuffled_t77"][repeat_index]
                        )
                    else:
                        values.append(cell[target_prompt][repeat_index])
        return statistics.fmean(values)

    point = estimate()
    draws = [estimate(rng) for _ in range(samples)]
    return point, percentile(draws, 0.025), percentile(draws, 0.975)


def write_fixed_summary(rows, output):
    output.mkdir(parents=True, exist_ok=True)
    performance, contrasts = [], []
    for family in FIXED_FAMILIES:
        family_rows = [row for row in rows if row["checkpoint_family"] == family]
        for prompt in FIXED_PROMPTS:
            selected = [row for row in family_rows if row["prompt"] == prompt]
            mean, lower, upper = bootstrap_fixed(selected)
            performance.append({
                "checkpoint_family": family, "prompt": prompt,
                "mean_accuracy": mean, "bootstrap_ci95_lower": lower,
                "bootstrap_ci95_upper": upper,
                "training_generation_cells": len(selected),
                "classifier_observations": sum(len(scores(row["evaluation_log"])) for row in selected),
            })
        mean, lower, upper = bootstrap_fixed(family_rows, contrast=True)
        contrasts.append({
            "checkpoint_family": family,
            "contrast": "correct_t77_minus_shuffled_t77",
            "mean_difference": mean, "bootstrap_ci95_lower": lower,
            "bootstrap_ci95_upper": upper,
            "training_generation_cells": len({
                (row["training_seed"], row["generation_seed"]) for row in family_rows
            }),
            "paired_classifier_observations": sum(
                len(scores(row["evaluation_log"]))
                for row in family_rows if row["prompt"] == "correct_t77"
            ),
        })
    for name, values in (("performance.csv", performance), ("contrasts.csv", contrasts)):
        with (output / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    (output / "summary.json").write_text(
        json.dumps({"performance": performance, "contrasts": contrasts}, indent=2) + "\n",
        encoding="utf-8",
    )


def semantic_manifest(payload):
    return {key: value for key, value in payload.items() if key != "gpus"}


def main():
    args = parse_args()
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
    gpus = [value.strip() for value in args.gpus.split(",") if value.strip()]
    if not gpus:
        raise ValueError("At least one GPU is required")
    for path in (
        args.data_root, args.caption_file, args.base_model, args.prototype, args.dcs,
    ):
        if not Path(path).exists():
            raise FileNotFoundError(path)
    root = Path(args.run_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    tasks, sparse_index, fixed_index = build_tasks(args)
    manifest = {
        "format_version": 1,
        "experiment": "sparse_m4_m8_and_refit_dense_t77_completion",
        "data_root": str(Path(args.data_root).resolve()),
        "caption_file": str(Path(args.caption_file).resolve()),
        "base_model": str(Path(args.base_model).resolve()),
        "prototype": str(Path(args.prototype).resolve()), "dcs": str(Path(args.dcs).resolve()),
        "gpus": gpus, "sparse_budgets": [4, 8], "bank_seeds": [0, 1],
        "sparse_training": {"micro_batch": 8, "gradient_accumulation": 4, "epochs": 8},
        "dense_training": {
            "families": list(FIXED_FAMILIES), "micro_batch": 8,
            "gradient_accumulation": 4, "epochs": 8,
            "training_seeds": list(args.dense_training_seeds),
            "text_interface": "CLIP T77 truncation and max-length padding",
        },
        "sparse_prompts": ["label", "bank_t77"],
        "fixed_prompts": list(FIXED_PROMPTS),
        "generation_seeds": list(args.generation_seeds),
        "ipc": args.ipc, "strength": args.strength,
        "guidance_scale": 10.0, "num_inference_steps": 50,
        "classifier_repeats": args.classifier_repeats,
        "conditioning_interface": "single CLIP T77 block for every inference prompt",
    }
    manifest_path = root / "run_manifest.json"
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if semantic_manifest(previous) != semantic_manifest(manifest):
            raise RuntimeError(f"Resume configuration differs from {manifest_path}")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (root / "sparse_evaluation_index.json").write_text(
        json.dumps(sparse_index, indent=2) + "\n", encoding="utf-8"
    )
    (root / "fixed_evaluation_index.json").write_text(
        json.dumps(fixed_index, indent=2) + "\n", encoding="utf-8"
    )
    (root / "evaluation_index.json").write_text(
        json.dumps(sparse_index + fixed_index, indent=2) + "\n", encoding="utf-8"
    )

    completed = {name for name, task in tasks.items() if task.complete()}
    running = {}
    while len(completed) < len(tasks):
        now = time.time()
        for gpu, task in list(running.items()):
            code = task.process.poll()
            if code is None:
                continue
            task.handle.write(f"[{time.strftime('%F %T')}] exit code {code}\n")
            task.handle.close()
            task.process = task.handle = None
            del running[gpu]
            if code == 0 and task.complete():
                completed.add(task.name)
                print(f"DONE GPU {gpu}: {task.name} ({len(completed)}/{len(tasks)})", flush=True)
            else:
                archived = task.log.with_name(
                    f"{task.log.name}.failed_{task.attempts}_{time.strftime('%Y%m%dT%H%M%S')}"
                )
                if task.log.exists():
                    task.log.replace(archived)
                if args.max_retries and task.attempts >= args.max_retries:
                    raise RuntimeError(f"Task exhausted retries: {task.name}; see {archived}")
                task.next_ready = time.time() + args.retry_delay_seconds
                print(f"RETRY in {args.retry_delay_seconds}s: {task.name}", flush=True)
        ready = [
            task for task in tasks.values()
            if task.name not in completed and task.process is None and task.next_ready <= now
            and all(dependency in completed for dependency in task.dependencies)
        ]
        ready.sort(key=lambda task: (
            task.stage, {"train": 0, "eval": 1, "generate": 2}[task.kind],
            task.attempts, task.name,
        ))
        for gpu in [value for value in gpus if value not in running]:
            if not ready:
                break
            task = ready.pop(0)
            launch(task, gpu, args)
            running[gpu] = task
        write_state(root, tasks, completed, running)
        time.sleep(5)

    subprocess.run([
        sys.executable, str(HERE / "summarize_sparse_prompt_search.py"),
        "--evaluation-index", str(root / "sparse_evaluation_index.json"),
        "--output-dir", str(root / "summary" / "sparse"),
    ], cwd=REPO_ROOT, check=True)
    write_fixed_summary(fixed_index, root / "summary" / "fixed")
    (root / "COMPLETE").write_text(time.strftime("%F %T\n"), encoding="utf-8")
    print(f"Sparse/fixed T77 completion complete: {root}")


if __name__ == "__main__":
    main()
