#!/usr/bin/env python3
"""Persistent four-GPU scheduler for nested sparse unpaired-caption search."""

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

from build_sparse_caption_banks import build_banks


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
EVAL_DIR = REPO_ROOT / "04_evaluation" / "Minimax"
RESULT = re.compile(r"Best, last acc:----\[[^\]]+\]")


@dataclass
class Task:
    name: str
    stage: int
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


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--caption-file", required=True)
    parser.add_argument("--spec", choices=("nette", "woof"), default="nette")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--prototype", required=True)
    parser.add_argument("--dcs", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--budgets", type=int, nargs="+", default=(4, 8, 16, 32))
    parser.add_argument("--bank-seeds", type=int, nargs="+", default=(0, 1))
    parser.add_argument("--training-seed", type=int, default=0)
    parser.add_argument("--generation-seeds", type=int, nargs="+", default=(0, 1))
    parser.add_argument(
        "--prompts", nargs="+", choices=("label", "bank", "bank_t77"),
        default=("label", "bank"),
    )
    parser.add_argument("--ipc", type=int, default=50)
    parser.add_argument("--strength", type=float, default=0.8)
    parser.add_argument("--classifier-repeats", type=int, default=2)
    parser.add_argument("--classifier-seed", type=int, default=0)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--mixed-precision", choices=("no", "fp16", "bf16"), default="fp16")
    parser.add_argument("--retry-delay-seconds", type=int, default=120)
    parser.add_argument("--max-retries", type=int, default=0, help="0 means retry forever")
    parser.add_argument("--diffusers-src", default="")
    return parser.parse_args()


def complete_model(path):
    return lambda: (path / "model_index.json").is_file() and (path / "training_summary.json").is_file()


def complete_eval(path):
    return lambda: path.is_file() and bool(RESULT.search(path.read_text(encoding="utf-8", errors="replace")))


def complete_generation(root, expected, prompts):
    def check():
        for prompt in prompts:
            condition = f"sparse_ft_{prompt}"
            complete = root / condition / "complete.json"
            if not complete.is_file():
                return False
            if int(json.loads(complete.read_text(encoding="utf-8"))["images"]) != expected:
                return False
        return True
    return check


def train_command(args, bank, output):
    command = [
        "accelerate", "launch", "--num_processes", "1", "--num_machines", "1",
        "--mixed_precision", args.mixed_precision, "--dynamo_backend", "no",
        str(HERE / "train_text_to_image_supervision.py"),
        "--pretrained-model", args.base_model,
        "--train-root", str(Path(args.data_root) / "train"),
        "--caption-file", args.caption_file,
        "--output-dir", str(output),
        "--supervision", "sparse_unpaired", "--sparse-bank", str(bank),
        "--resolution", "512", "--train-batch-size", str(args.train_batch_size),
        "--gradient-accumulation-steps", str(args.gradient_accumulation_steps),
        "--num-train-epochs", "8", "--learning-rate", "1e-5",
        "--lr-scheduler", "constant", "--lr-warmup-steps", "0", "--max-grad-norm", "1",
        "--mixed-precision", args.mixed_precision, "--seed", str(args.training_seed),
        "--num-workers", str(args.num_workers), "--checkpointing-steps", "500",
        "--checkpoints-total-limit", "2", "--loss-log-steps", "50", "--timestep-bins", "10",
        "--random-flip", "--gradient-checkpointing", "--use-ema",
    ]
    if output.is_dir() and any(output.glob("checkpoint-*")):
        command.extend(("--resume-from-checkpoint", "latest"))
    return command


def generation_command(args, bank, model, output, generation_seed):
    prompts = getattr(args, "prompts", ("label", "bank"))
    return [
        sys.executable, str(HERE / "generate_factorial.py"),
        "--prototype", args.prototype, "--dcs", args.dcs, "--base-model", args.base_model,
        "--model", f"sparse_ft={model}", "--supervisions", "sparse_ft",
        "--prompts", *prompts, "--prompt-bank", str(bank),
        "--output-root", str(output), "--generation-seeds", str(generation_seed),
        "--ipc", str(args.ipc), "--strength", str(args.strength),
        "--guidance-scale", "10", "--num-inference-steps", "50", "--shuffle-shift", "1",
        "--size", "256", "--resume",
    ]


def eval_command(args, synthetic, tag):
    return [
        sys.executable, "train.py", "-d", "imagenet", "--imagenet_dir",
        str(synthetic), str(Path(args.data_root).resolve()), "-n", "resnet_ap",
        "--nclass", "10", "--norm_type", "instance", "--ipc", str(args.ipc),
        "--tag", tag, "--slct_type", "random", "--repeat", str(args.classifier_repeats),
        "--spec", getattr(args, "spec", "nette"), "--seed", str(args.classifier_seed),
    ]


def build_tasks(args):
    root = Path(args.run_root).resolve()
    prompts = getattr(args, "prompts", ("label", "bank"))
    tasks, index = {}, []
    for bank_seed in sorted(set(args.bank_seeds)):
        for budget in sorted(set(args.budgets)):
            token = f"bank{bank_seed}_m{budget}"
            bank = root / "caption_banks" / f"bank_seed_{bank_seed}" / f"m_{budget}.json"
            model = root / "models" / f"bank_seed_{bank_seed}" / f"m_{budget}" / "sparse_ft"
            train_name = f"train_{token}"
            tasks[train_name] = Task(
                train_name, 1, "train", train_command(args, bank, model), REPO_ROOT,
                root / "scheduler_logs" / f"{train_name}.log", complete_model(model),
            )
            for generation_seed in args.generation_seeds:
                output = root / "synthetic" / f"bank_seed_{bank_seed}" / f"m_{budget}"
                seed_root = output / f"seed_{generation_seed}"
                gen_name = f"gen_{token}_g{generation_seed}"
                tasks[gen_name] = Task(
                    gen_name, 2, "generate",
                    generation_command(args, bank, model, output, generation_seed), REPO_ROOT,
                    root / "scheduler_logs" / f"{gen_name}.log",
                    complete_generation(seed_root, 10 * args.ipc, prompts), dependencies=(train_name,),
                )
                for prompt in prompts:
                    condition = f"sparse_ft_{prompt}"
                    eval_name = f"eval_{token}_g{generation_seed}_{prompt}"
                    log = root / "evaluation" / f"bank_seed_{bank_seed}" / f"m_{budget}" / f"seed_{generation_seed}" / f"{condition}.log"
                    tasks[eval_name] = Task(
                        eval_name, 2, "eval", eval_command(args, seed_root / condition, eval_name),
                        EVAL_DIR, log, complete_eval(log), dependencies=(gen_name,),
                    )
                    index.append({
                        "method": "random_sparse_unpaired_marginal",
                        "bank_seed": bank_seed, "budget": budget,
                        "training_seed": args.training_seed,
                        "generation_seed": generation_seed,
                        "prompt": prompt, "ipc": args.ipc, "strength": args.strength,
                        "spec": getattr(args, "spec", "nette"),
                        "evaluation_log": str(log),
                    })
    return tasks, index


def quote(command):
    return " ".join(shlex.quote(str(item)) for item in command)


def launch(task, gpu, args):
    command = list(task.command)
    if task.kind == "train":
        command[2:2] = ["--main_process_port", str(29600 + int(gpu))]
        output = Path(command[command.index("--output-dir") + 1])
        if output.is_dir() and any(output.glob("checkpoint-*")) and "--resume-from-checkpoint" not in command:
            command.extend(("--resume-from-checkpoint", "latest"))
        elif output.is_dir() and any(output.iterdir()) and not complete_model(output)():
            archived = output.with_name(f"{output.name}.incomplete_{time.strftime('%Y%m%dT%H%M%S')}")
            output.replace(archived)
            print(f"ARCHIVE non-resumable output: {output} -> {archived}", flush=True)
    task.log.parent.mkdir(parents=True, exist_ok=True)
    task.handle = task.log.open("a", encoding="utf-8", buffering=1)
    task.handle.write(f"\n[{time.strftime('%F %T')}] attempt {task.attempts + 1} GPU {gpu}\n{quote(command)}\n")
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": gpu, "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false"})
    if args.diffusers_src:
        env["PYTHONPATH"] = args.diffusers_src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    task.process = subprocess.Popen(command, cwd=task.cwd, env=env, stdout=task.handle, stderr=subprocess.STDOUT)
    task.attempts += 1
    print(f"LAUNCH GPU {gpu}: {task.name} (attempt {task.attempts})", flush=True)


def write_state(root, tasks, completed, running):
    active = {task.name for task in running.values()}
    payload = {
        "updated_at": time.strftime("%F %T"),
        "completed": sorted(completed),
        "running": {gpu: task.name for gpu, task in running.items()},
        "pending": sorted(name for name in tasks if name not in completed and name not in active),
    }
    (root / "scheduler_state.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if not gpus:
        raise ValueError("At least one GPU is required")
    if (args.train_batch_size, args.gradient_accumulation_steps) != (8, 4):
        raise ValueError("VLCP training protocol requires micro-batch 8 and gradient accumulation 4")
    for path in (args.data_root, args.caption_file, args.base_model, args.prototype, args.dcs):
        if not Path(path).exists():
            raise FileNotFoundError(path)
    root = Path(args.run_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    build_banks(
        Path(args.data_root) / "train", args.caption_file, root / "caption_banks",
        args.budgets, args.bank_seeds,
    )
    tasks, index = build_tasks(args)
    manifest = {
        "format_version": 1,
        "experiment": "random_sparse_unpaired_caption_marginal",
        "data_root": str(Path(args.data_root).resolve()),
        "caption_file": str(Path(args.caption_file).resolve()),
        "spec": args.spec,
        "base_model": str(Path(args.base_model).resolve()),
        "prototype": str(Path(args.prototype).resolve()),
        "dcs": str(Path(args.dcs).resolve()),
        "gpus": gpus, "budgets": sorted(set(args.budgets)),
        "bank_seeds": sorted(set(args.bank_seeds)),
        "training_seed": args.training_seed,
        "generation_seeds": list(args.generation_seeds),
        "prompts": list(args.prompts),
        "ipc": args.ipc, "strength": args.strength,
        "classifier_repeats": args.classifier_repeats,
        "training_text_interface": "CLIP T77 truncation and max-length padding",
        "inference_text_interfaces": {
            "label": "CLIP T77 truncation and max-length padding",
            "bank": "legacy multi-chunk conditioning",
            "bank_t77": "CLIP T77 truncation and max-length padding",
        },
        "no_walltime_limit": True,
    }
    manifest_path = root / "run_manifest.json"
    if manifest_path.is_file() and json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
        raise RuntimeError(f"Resume configuration differs from {manifest_path}")
    if not manifest_path.is_file():
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (root / "evaluation_index.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    completed = {name for name, task in tasks.items() if task.complete()}
    running = {}
    print(f"Sparse search: {len(completed)}/{len(tasks)} tasks already complete; no walltime limit", flush=True)
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
                archived = task.log.with_name(f"{task.log.name}.failed_{task.attempts}_{time.strftime('%Y%m%dT%H%M%S')}")
                if task.log.exists():
                    task.log.replace(archived)
                if args.max_retries and task.attempts >= args.max_retries:
                    raise RuntimeError(f"Task exhausted retries: {task.name}; see {archived}")
                task.next_ready = time.time() + args.retry_delay_seconds
                print(f"RETRY in {args.retry_delay_seconds}s: {task.name}; see {archived}", flush=True)
        free = [gpu for gpu in gpus if gpu not in running]
        ready = [
            task for task in tasks.values()
            if task.name not in completed and task.process is None and task.next_ready <= now
            and all(dependency in completed for dependency in task.dependencies)
        ]
        ready.sort(key=lambda task: (task.stage, {"train": 0, "eval": 1, "generate": 2}[task.kind], task.attempts, task.name))
        for gpu in free:
            if not ready:
                break
            selected = ready.pop(0)
            launch(selected, gpu, args)
            running[gpu] = selected
        write_state(root, tasks, completed, running)
        time.sleep(5)
    subprocess.run([
        sys.executable, str(HERE / "summarize_sparse_prompt_search.py"),
        "--evaluation-index", str(root / "evaluation_index.json"),
        "--output-dir", str(root / "summary"),
    ], cwd=REPO_ROOT, check=True)
    (root / "COMPLETE").write_text(time.strftime("%F %T\n"), encoding="utf-8")
    print(f"Sparse prompt search complete: {root}")


if __name__ == "__main__":
    main()
