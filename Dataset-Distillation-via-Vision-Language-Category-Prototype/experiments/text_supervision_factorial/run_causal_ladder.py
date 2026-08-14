#!/usr/bin/env python3
"""Persistent four-GPU scheduler for the text-supervision causal ladder."""

import argparse
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
EVAL_DIR = REPO_ROOT / "04_evaluation" / "Minimax"
RESULT = re.compile(r"Best, last acc:----\[[^\]]+\]")
PROMPTS = ("label", "correct", "shuffled")
NEW_ROWS = {
    0: ("empty_ft", "constant_ft"),
    1: ("empty_ft", "constant_ft", "label_ft", "unpaired_ft", "matched_ft"),
}
TRAIN_SUPERVISION = {
    "empty_ft": "empty",
    "constant_ft": "constant",
    "label_ft": "label",
    "unpaired_ft": "unpaired",
    "matched_ft": "matched",
}


@dataclass
class Task:
    name: str
    kind: str
    command: list
    cwd: Path
    log: Path
    complete: callable
    dependencies: tuple = ()
    attempts: int = 0
    next_ready: float = 0.0
    process: subprocess.Popen | None = field(default=None, repr=False)
    handle: object | None = field(default=None, repr=False)
    gpu: str | None = None


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--caption-file", required=True)
    parser.add_argument("--base-run-root", required=True, help="Completed original 4x3 run")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--prototype", default=None)
    parser.add_argument("--dcs", default=None)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--generation-seeds", type=int, nargs="+", default=(0, 1))
    parser.add_argument("--classifier-repeats", type=int, default=2)
    parser.add_argument("--classifier-seed", type=int, default=0)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--mixed-precision", choices=("no", "fp16", "bf16"), default="fp16")
    parser.add_argument("--constant-prompt", default="A natural photo.")
    parser.add_argument("--max-parallel-evals", type=int, default=2)
    parser.add_argument("--retry-delay-seconds", type=int, default=120)
    parser.add_argument("--max-retries", type=int, default=0, help="0 retries forever")
    parser.add_argument("--diffusers-src", default="")
    return parser.parse_args()


def complete_model(path):
    return lambda: (path / "model_index.json").is_file() and (path / "training_summary.json").is_file()


def complete_eval(path):
    def check():
        return path.is_file() and bool(RESULT.search(path.read_text(encoding="utf-8", errors="replace")))
    return check


def quote_command(command):
    return " ".join(shlex.quote(str(item)) for item in command)


def validate_base_run(base_run, generation_seeds):
    missing = []
    for row in ("label_ft", "unpaired_ft", "matched_ft"):
        model = base_run / "models" / row
        if not (model / "model_index.json").is_file() or not (model / "training_summary.json").is_file():
            missing.append(str(model))
    for generation_seed in generation_seeds:
        seed_dir = base_run / "evaluation" / f"seed_{generation_seed}"
        for row in ("frozen", "label_ft", "unpaired_ft", "matched_ft"):
            for prompt in PROMPTS:
                log = seed_dir / f"{row}_{prompt}.log"
                if not complete_eval(log)():
                    missing.append(str(log))
    if missing:
        preview = "\n".join(f"  {path}" for path in missing[:20])
        suffix = f"\n  ... and {len(missing) - 20} more" if len(missing) > 20 else ""
        raise RuntimeError(f"BASE_RUN_ROOT is not a complete reusable 4x3 run:\n{preview}{suffix}")


def build_tasks(args, run_root, prototype, dcs):
    tasks = {}
    model_root = run_root / "models"
    synthetic_root = run_root / "synthetic"
    evaluation_root = run_root / "evaluation"
    logs = run_root / "scheduler_logs"
    logs.mkdir(parents=True, exist_ok=True)

    for train_seed, rows in NEW_ROWS.items():
        for row in rows:
            output = model_root / f"train_seed_{train_seed}" / row
            command = [
                "accelerate", "launch", "--num_processes", "1", "--num_machines", "1",
                "--mixed_precision", args.mixed_precision, "--dynamo_backend", "no",
                str(HERE / "train_text_to_image_supervision.py"),
                "--pretrained-model", args.base_model,
                "--train-root", str(Path(args.data_root) / "train"),
                "--caption-file", args.caption_file,
                "--output-dir", str(output),
                "--supervision", TRAIN_SUPERVISION[row],
                "--constant-prompt", args.constant_prompt,
                "--resolution", "512", "--train-batch-size", str(args.train_batch_size),
                "--gradient-accumulation-steps", str(args.gradient_accumulation_steps),
                "--num-train-epochs", "8", "--learning-rate", "1e-5",
                "--lr-scheduler", "constant", "--lr-warmup-steps", "0",
                "--max-grad-norm", "1", "--mixed-precision", args.mixed_precision,
                "--seed", str(train_seed), "--num-workers", str(args.num_workers),
                "--checkpointing-steps", "500", "--checkpoints-total-limit", "2",
                "--loss-log-steps", "50", "--timestep-bins", "10",
                "--random-flip", "--gradient-checkpointing", "--use-ema",
            ]
            if any(output.glob("checkpoint-*")):
                command.extend(("--resume-from-checkpoint", "latest"))
            name = f"train_s{train_seed}_{row}"
            tasks[name] = Task(name, "train", command, REPO_ROOT, logs / f"{name}.log", complete_model(output))

            for generation_seed in args.generation_seeds:
                generation_base = synthetic_root / f"train_seed_{train_seed}" / f"seed_{generation_seed}"
                generation_output = generation_base / f"{row}_label"
                gen_name = f"generate_s{train_seed}_{row}_g{generation_seed}"
                gen_command = [
                    sys.executable, str(HERE / "generate_factorial.py"),
                    "--prototype", str(prototype), "--dcs", str(dcs),
                    "--base-model", args.base_model, "--model", f"{row}={output}",
                    "--supervisions", row, "--prompts", *PROMPTS,
                    "--output-root", str(synthetic_root / f"train_seed_{train_seed}"),
                    "--generation-seeds", str(generation_seed), "--ipc", "10",
                    "--strength", "0.7", "--guidance-scale", "10",
                    "--num-inference-steps", "50", "--shuffle-shift", "1",
                    "--size", "256", "--resume",
                ]
                tasks[gen_name] = Task(
                    gen_name, "generate", gen_command, REPO_ROOT, logs / f"{gen_name}.log",
                    lambda base=generation_base, mode=row: all(
                        (base / f"{mode}_{prompt}" / "complete.json").is_file() for prompt in PROMPTS
                    ),
                    dependencies=(name,),
                )
                for prompt in PROMPTS:
                    condition = f"{row}_{prompt}"
                    eval_log = evaluation_root / f"train_seed_{train_seed}" / f"seed_{generation_seed}" / f"{condition}.log"
                    eval_name = f"eval_s{train_seed}_{row}_{prompt}_g{generation_seed}"
                    eval_command = [
                        sys.executable, "train.py", "-d", "imagenet", "--imagenet_dir",
                        str(generation_base / condition), args.data_root,
                        "-n", "resnet_ap", "--nclass", "10", "--norm_type", "instance",
                        "--ipc", "10", "--tag", f"causal_ladder_s{train_seed}_g{generation_seed}_{condition}",
                        "--slct_type", "random", "--repeat", str(args.classifier_repeats),
                        "--spec", "nette", "--seed", str(args.classifier_seed),
                    ]
                    tasks[eval_name] = Task(
                        eval_name, "eval", eval_command, EVAL_DIR, eval_log,
                        complete_eval(eval_log), dependencies=(gen_name,),
                    )
    return tasks


def write_manifest(args, run_root, prototype, dcs):
    manifest = {
        "format_version": 1,
        "data_root": str(Path(args.data_root).resolve()),
        "base_model": str(Path(args.base_model).resolve()),
        "caption_file": str(Path(args.caption_file).resolve()),
        "base_run_root": str(Path(args.base_run_root).resolve()),
        "prototype": str(prototype.resolve()),
        "dcs": str(dcs.resolve()),
        "training_seeds": [0, 1],
        "new_rows": {str(key): list(value) for key, value in NEW_ROWS.items()},
        "generation_seeds": list(args.generation_seeds),
        "classifier_repeats": args.classifier_repeats,
        "constant_prompt": args.constant_prompt,
        "train_batch_size": args.train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_workers_per_training_job": args.num_workers,
        "mixed_precision": args.mixed_precision,
        "effective_batch_size_per_checkpoint": args.train_batch_size * args.gradient_accumulation_steps,
    }
    path = run_root / "run_manifest.json"
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise RuntimeError(f"Resume configuration differs from {path}")
    else:
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def launch(task, gpu, args):
    task.log.parent.mkdir(parents=True, exist_ok=True)
    task.handle = task.log.open("a", encoding="utf-8", buffering=1)
    task.handle.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] attempt {task.attempts + 1} GPU {gpu}\n")
    command = list(task.command)
    if task.kind == "train":
        if "--main_process_port" not in command:
            command[2:2] = ["--main_process_port", str(29600 + int(gpu))]
        output_index = command.index("--output-dir") + 1
        output = Path(command[output_index])
        if any(output.glob("checkpoint-*")) and "--resume-from-checkpoint" not in command:
            command.extend(("--resume-from-checkpoint", "latest"))
    task.handle.write(quote_command(command) + "\n")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    if args.diffusers_src:
        env["PYTHONPATH"] = args.diffusers_src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    task.process = subprocess.Popen(
        command, cwd=task.cwd, env=env, stdout=task.handle, stderr=subprocess.STDOUT,
        start_new_session=False,
    )
    task.gpu = gpu
    task.attempts += 1
    print(f"LAUNCH GPU {gpu}: {task.name} (attempt {task.attempts})", flush=True)


def main():
    args = parse_args()
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if len(gpus) != 4:
        raise ValueError(f"This scheduler requires exactly four GPU ids, got {gpus}")
    if (args.train_batch_size, args.gradient_accumulation_steps) != (8, 4):
        raise ValueError("VLCP training protocol requires micro-batch 8 and gradient accumulation 4")
    run_root = Path(args.run_root).resolve()
    base_run = Path(args.base_run_root).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    prototype = Path(args.prototype or base_run / "prototypes" / "text_supervision-ipc10-0.7-30-kmexpand1.json").resolve()
    dcs = Path(args.dcs or base_run / "prototypes" / "dcs.json").resolve()
    for required in (Path(args.data_root), Path(args.base_model), Path(args.caption_file), prototype, dcs):
        if not required.exists():
            raise FileNotFoundError(required)
    validate_base_run(base_run, args.generation_seeds)
    write_manifest(args, run_root, prototype, dcs)
    tasks = build_tasks(args, run_root, prototype, dcs)
    completed = {name for name, task in tasks.items() if task.complete()}
    running = {}
    print(f"Causal ladder: {len(completed)}/{len(tasks)} tasks already complete", flush=True)

    while len(completed) < len(tasks):
        now = time.time()
        for gpu, task in list(running.items()):
            code = task.process.poll()
            if code is None:
                continue
            task.handle.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] exit code {code}\n")
            task.handle.close()
            task.process = task.handle = None
            task.gpu = None
            del running[gpu]
            if code == 0 and task.complete():
                completed.add(task.name)
                print(f"DONE GPU {gpu}: {task.name} ({len(completed)}/{len(tasks)})", flush=True)
            else:
                failed_log = task.log.with_name(
                    f"{task.log.name}.failed_attempt_{task.attempts}_{time.strftime('%Y%m%dT%H%M%S')}"
                )
                if task.log.exists():
                    task.log.replace(failed_log)
                if args.max_retries and task.attempts >= args.max_retries:
                    raise RuntimeError(f"Task exhausted retries: {task.name}; see {failed_log}")
                task.next_ready = time.time() + args.retry_delay_seconds
                print(f"RETRY in {args.retry_delay_seconds}s: {task.name}; see {failed_log}", flush=True)

        free_gpus = [gpu for gpu in gpus if gpu not in running]
        running_eval_count = sum(task.kind == "eval" for task in running.values())
        ready = [
            task for task in tasks.values()
            if task.name not in completed
            and task.process is None
            and task.next_ready <= now
            and all(dep in completed for dep in task.dependencies)
        ]
        ready.sort(key=lambda task: ({"train": 0, "generate": 1, "eval": 2}[task.kind], task.attempts, task.name))
        for gpu in free_gpus:
            selected = None
            for task in ready:
                if task.kind == "eval" and running_eval_count >= args.max_parallel_evals:
                    continue
                selected = task
                break
            if selected is None:
                break
            ready.remove(selected)
            launch(selected, gpu, args)
            running[gpu] = selected
            if selected.kind == "eval":
                running_eval_count += 1
        time.sleep(5)

    summary_dir = run_root / "summary"
    command = [
        sys.executable, str(HERE / "summarize_causal_ladder.py"),
        "--base-evaluation-root", str(base_run / "evaluation"),
        "--extension-evaluation-root", str(run_root / "evaluation"),
        "--output-dir", str(summary_dir),
    ]
    subprocess.run(command, check=True, cwd=REPO_ROOT)
    (run_root / "COMPLETE").write_text(time.strftime("%Y-%m-%d %H:%M:%S\n"), encoding="utf-8")
    print(f"Causal ladder complete: {run_root}", flush=True)


if __name__ == "__main__":
    main()
