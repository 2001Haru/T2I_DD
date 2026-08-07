#!/usr/bin/env python3
"""Resume-safe two-GPU scheduler for seed, IPC, and ImageWoof generality tests."""

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

from subset_specs import validate_subset


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DISTILLATION_DIR = REPO_ROOT / "03_distiilation"
EVAL_DIR = REPO_ROOT / "04_evaluation" / "Minimax"
RESULT = re.compile(r"Best, last acc:----\[[^\]]+\]")
TRAIN_SUPERVISION = {"empty_ft": "empty", "unpaired_ft": "unpaired", "matched_ft": "matched"}
PHASE_NUMBER = {"nette_seeds": 1, "nette_ipc": 2, "woof": 3}


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
    parser.add_argument("--nette-data-root", required=True)
    parser.add_argument("--nette-caption-file", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--base-run-root", required=True, help="Original seed-0 4x3 run")
    parser.add_argument("--causal-run-root", required=True, help="Completed seed-0/1 causal ladder")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--nette-prototype", default=None)
    parser.add_argument("--nette-dcs", default=None)
    parser.add_argument("--woof-data-root", default=None)
    parser.add_argument("--woof-caption-file", default=None)
    parser.add_argument("--phases", nargs="+", choices=tuple(PHASE_NUMBER), default=("nette_seeds", "nette_ipc"))
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--new-training-seeds", type=int, nargs="+", default=(2, 3))
    parser.add_argument("--ipc-training-seeds", type=int, nargs="+", default=(0, 1))
    parser.add_argument("--woof-training-seeds", type=int, nargs="+", default=(0, 1))
    parser.add_argument("--generation-seeds", type=int, nargs="+", default=(0, 1))
    parser.add_argument("--ipc-values", type=int, nargs="+", default=(20, 50))
    parser.add_argument("--classifier-repeats", type=int, default=2)
    parser.add_argument("--classifier-seed", type=int, default=0)
    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--mixed-precision", choices=("no", "fp16", "bf16"), default="fp16")
    parser.add_argument("--max-parallel-evals", type=int, default=1)
    parser.add_argument("--retry-delay-seconds", type=int, default=120)
    parser.add_argument("--max-retries", type=int, default=0, help="0 retries forever")
    parser.add_argument("--diffusers-src", default="")
    return parser.parse_args()


def complete_model(path):
    return lambda: (path / "model_index.json").is_file() and (path / "training_summary.json").is_file()


def complete_eval(path):
    return lambda: path.is_file() and bool(RESULT.search(path.read_text(encoding="utf-8", errors="replace")))


def complete_generation(root, conditions, expected):
    def check():
        for condition in conditions:
            path = root / condition / "complete.json"
            if not path.is_file():
                return False
            if int(json.loads(path.read_text(encoding="utf-8"))["images"]) != expected:
                return False
        return True
    return check


def validate_caption_coverage(data_root, caption_file):
    data_root = Path(data_root).resolve()
    images = {
        path.relative_to(data_root / "train").as_posix()
        for path in (data_root / "train").rglob("*")
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    }
    captions = set()
    with Path(caption_file).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                captions.add(str(json.loads(line)["file_name"]).replace("\\", "/"))
    if images == captions:
        return
    # Legacy ImageWoof metadata stores basenames only.
    if {Path(path).name for path in images} == captions and len(captions) == len(images):
        return
    raise RuntimeError(
        f"Caption/image mismatch under {data_root}: {len(images - captions)} direct-path missing, "
        f"{len(captions - images)} direct-path unknown"
    )


def existing_model(args, training_seed, mode):
    base = Path(args.base_run_root).resolve()
    causal = Path(args.causal_run_root).resolve()
    run = Path(args.run_root).resolve()
    if training_seed == 0 and mode == "matched_ft":
        return base / "models" / mode
    if training_seed in (0, 1):
        return causal / "models" / f"train_seed_{training_seed}" / mode
    return run / "models" / "nette" / f"train_seed_{training_seed}" / mode


def prototype_paths(run_root, spec, ipc):
    root = run_root / "artifacts" / spec / f"ipc{ipc}"
    return root / f"{spec}-ipc{ipc}-0.7-30-kmexpand1.json", root / "dcs.json"


def train_command(args, data_root, caption_file, output, supervision, seed):
    command = [
        "accelerate", "launch", "--num_processes", "1", "--num_machines", "1",
        "--mixed_precision", args.mixed_precision, "--dynamo_backend", "no",
        str(HERE / "train_text_to_image_supervision.py"),
        "--pretrained-model", args.base_model, "--train-root", str(Path(data_root) / "train"),
        "--caption-file", str(caption_file), "--output-dir", str(output),
        "--supervision", supervision, "--resolution", "512",
        "--train-batch-size", str(args.train_batch_size),
        "--gradient-accumulation-steps", str(args.gradient_accumulation_steps),
        "--num-train-epochs", "8", "--learning-rate", "1e-5", "--lr-scheduler", "constant",
        "--lr-warmup-steps", "0", "--max-grad-norm", "1", "--mixed-precision", args.mixed_precision,
        "--seed", str(seed), "--num-workers", str(args.num_workers), "--checkpointing-steps", "500",
        "--checkpoints-total-limit", "2", "--loss-log-steps", "50", "--timestep-bins", "10",
        "--random-flip", "--gradient-checkpointing", "--use-ema",
    ]
    if any(output.glob("checkpoint-*")):
        command.extend(("--resume-from-checkpoint", "latest"))
    return command


def prototype_command(args, spec, data_root, caption_file, ipc, output_root, dcs):
    label = DISTILLATION_DIR / "label-prompt" / ("class_nette.txt" if spec == "nette" else "imagenet_woof_classes.txt")
    return [
        sys.executable, "gen_prototype.py", "--batch_size", "10", "--spec", spec,
        "--contamination", "0.1", "--data_dir", str(data_root), "--dataset", "imagenet",
        "--diffusion_checkpoints_path", args.base_model, "--ipc", str(ipc), "--km_expand", "1",
        "--label_file_path", str(label), "--save_prototype_path", str(output_root),
        "--save_text_prototype_path", str(dcs), "--seed", "0", "--metajson_file", str(caption_file),
        "--threshold", "0.7", "--tpk", "30", "--num_workers", str(args.num_workers),
    ]


def generation_command(args, prototype, dcs, model, mode, prompts, output_root, generation_seed, ipc):
    command = [
        sys.executable, str(HERE / "generate_factorial.py"), "--prototype", str(prototype),
        "--dcs", str(dcs), "--base-model", args.base_model,
    ]
    if mode != "frozen":
        command.extend(("--model", f"{mode}={model}"))
    command.extend((
        "--supervisions", mode, "--prompts", *prompts, "--output-root", str(output_root),
        "--generation-seeds", str(generation_seed), "--ipc", str(ipc), "--strength", "0.7",
        "--guidance-scale", "10", "--num-inference-steps", "50", "--shuffle-shift", "1",
        "--size", "256", "--resume",
    ))
    return command


def eval_command(args, synthetic, data_root, ipc, spec, tag):
    return [
        sys.executable, "train.py", "-d", "imagenet", "--imagenet_dir", str(synthetic), str(data_root),
        "-n", "resnet_ap", "--nclass", "10", "--norm_type", "instance", "--ipc", str(ipc),
        "--tag", tag, "--slct_type", "random", "--repeat", str(args.classifier_repeats),
        "--spec", spec, "--seed", str(args.classifier_seed),
    ]


def add_eval(tasks, index, args, name, stage, synthetic, data_root, ipc, spec, log, dependency, metadata):
    tasks[name] = Task(
        name, stage, "eval", eval_command(args, synthetic, data_root, ipc, spec, name), EVAL_DIR, log,
        complete_eval(log), dependencies=(dependency,),
    )
    index.append({**metadata, "ipc": ipc, "spec": spec, "evaluation_log": str(log), "source": "new"})


def add_generation_group(tasks, index, args, stage, prefix, data_root, spec, ipc, prototype, dcs,
                         model, mode, prompts, output_root, generation_seed, training_seed, dependency=()):
    seed_root = output_root / f"seed_{generation_seed}"
    conditions = [f"{mode}_{prompt}" for prompt in prompts]
    name = f"gen_{prefix}_{mode}_g{generation_seed}"
    tasks[name] = Task(
        name, stage, "generate",
        generation_command(args, prototype, dcs, model, mode, prompts, output_root, generation_seed, ipc),
        REPO_ROOT, Path(args.run_root) / "scheduler_logs" / f"{name}.log",
        complete_generation(seed_root, conditions, 10 * ipc), dependencies=tuple(dependency),
    )
    for prompt, condition in zip(prompts, conditions):
        eval_name = f"eval_{prefix}_{condition}_g{generation_seed}"
        log = Path(args.run_root) / "evaluation" / prefix / f"seed_{generation_seed}" / f"{condition}.log"
        add_eval(tasks, index, args, eval_name, stage, seed_root / condition, data_root, ipc, spec, log, name, {
            "phase": stage, "training_seed": training_seed, "generation_seed": generation_seed,
            "supervision": mode, "prompt": prompt,
        })


def add_reused_nette_seed_rows(args, index):
    base = Path(args.base_run_root).resolve()
    causal = Path(args.causal_run_root).resolve()
    for training_seed in (0, 1):
        for generation_seed in args.generation_seeds:
            for mode, prompts in (("empty_ft", ("label",)), ("matched_ft", ("correct", "shuffled"))):
                for prompt in prompts:
                    if training_seed == 0 and mode == "matched_ft":
                        log = base / "evaluation" / f"seed_{generation_seed}" / f"{mode}_{prompt}.log"
                    else:
                        log = causal / "evaluation" / f"train_seed_{training_seed}" / f"seed_{generation_seed}" / f"{mode}_{prompt}.log"
                    if not complete_eval(log)():
                        raise RuntimeError(f"Required reusable ImageNette result is incomplete: {log}")
                    index.append({
                        "phase": 1, "training_seed": training_seed, "generation_seed": generation_seed,
                        "supervision": mode, "prompt": prompt, "ipc": 10, "spec": "nette",
                        "evaluation_log": str(log), "source": "reused",
                    })


def build_tasks(args):
    tasks, index = {}, []
    run_root = Path(args.run_root).resolve()
    logs = run_root / "scheduler_logs"
    logs.mkdir(parents=True, exist_ok=True)
    nette_root = Path(args.nette_data_root).resolve()
    nette_caption = Path(args.nette_caption_file).resolve()
    base = Path(args.base_run_root).resolve()
    nette_p10 = Path(args.nette_prototype or base / "prototypes" / "text_supervision-ipc10-0.7-30-kmexpand1.json").resolve()
    nette_d10 = Path(args.nette_dcs or base / "prototypes" / "dcs.json").resolve()

    if "nette_seeds" in args.phases:
        for artifact in (nette_p10, nette_d10):
            if not artifact.is_file():
                raise FileNotFoundError(artifact)
        add_reused_nette_seed_rows(args, index)
        for training_seed in args.new_training_seeds:
            for mode in ("empty_ft", "matched_ft"):
                model = existing_model(args, training_seed, mode)
                train_name = f"train_nette_s{training_seed}_{mode}"
                tasks[train_name] = Task(
                    train_name, 1, "train",
                    train_command(args, nette_root, nette_caption, model, TRAIN_SUPERVISION[mode], training_seed),
                    REPO_ROOT, logs / f"{train_name}.log", complete_model(model),
                )
                prompts = ("label",) if mode == "empty_ft" else ("correct", "shuffled")
                for generation_seed in args.generation_seeds:
                    prefix = f"nette_ipc10_s{training_seed}"
                    output = run_root / "synthetic" / "nette" / "ipc10" / f"train_seed_{training_seed}"
                    add_generation_group(
                        tasks, index, args, 1, prefix, nette_root, "nette", 10, nette_p10, nette_d10,
                        model, mode, prompts, output, generation_seed, training_seed, dependency=(train_name,),
                    )

    if "nette_ipc" in args.phases:
        for ipc in args.ipc_values:
            prototype, dcs = prototype_paths(run_root, "nette", ipc)
            artifact_name = f"artifact_nette_ipc{ipc}"
            tasks[artifact_name] = Task(
                artifact_name, 2, "artifact",
                prototype_command(args, "nette", nette_root, nette_caption, ipc, prototype.parent, dcs),
                DISTILLATION_DIR, logs / f"{artifact_name}.log",
                lambda p=prototype, d=dcs: p.is_file() and d.is_file(),
            )
            for generation_seed in args.generation_seeds:
                frozen_output = run_root / "synthetic" / "nette" / f"ipc{ipc}" / "frozen"
                add_generation_group(
                    tasks, index, args, 2, f"nette_ipc{ipc}_frozen", nette_root, "nette", ipc,
                    prototype, dcs, args.base_model, "frozen", ("label",), frozen_output,
                    generation_seed, None, dependency=(artifact_name,),
                )
                for training_seed in args.ipc_training_seeds:
                    for mode, prompts in (("empty_ft", ("label",)), ("matched_ft", ("correct", "shuffled"))):
                        model = existing_model(args, training_seed, mode)
                        model_dependencies = [artifact_name]
                        pending_train = f"train_nette_s{training_seed}_{mode}"
                        if not complete_model(model)():
                            if pending_train not in tasks:
                                raise RuntimeError(f"Missing reusable checkpoint: {model}")
                            model_dependencies.append(pending_train)
                        output = run_root / "synthetic" / "nette" / f"ipc{ipc}" / f"train_seed_{training_seed}"
                        add_generation_group(
                            tasks, index, args, 2, f"nette_ipc{ipc}_s{training_seed}", nette_root, "nette", ipc,
                            prototype, dcs, model, mode, prompts, output, generation_seed, training_seed,
                            dependency=tuple(model_dependencies),
                        )

    if "woof" in args.phases:
        woof_root = Path(args.woof_data_root).resolve()
        woof_caption = Path(args.woof_caption_file).resolve()
        prototype, dcs = prototype_paths(run_root, "woof", 10)
        artifact_name = "artifact_woof_ipc10"
        tasks[artifact_name] = Task(
            artifact_name, 3, "artifact",
            prototype_command(args, "woof", woof_root, woof_caption, 10, prototype.parent, dcs),
            DISTILLATION_DIR, logs / f"{artifact_name}.log",
            lambda p=prototype, d=dcs: p.is_file() and d.is_file(),
        )
        train_names = {}
        for training_seed in args.woof_training_seeds:
            for mode in ("empty_ft", "unpaired_ft", "matched_ft"):
                model = run_root / "models" / "woof" / f"train_seed_{training_seed}" / mode
                name = f"train_woof_s{training_seed}_{mode}"
                train_names[(training_seed, mode)] = name
                tasks[name] = Task(
                    name, 3, "train",
                    train_command(args, woof_root, woof_caption, model, TRAIN_SUPERVISION[mode], training_seed),
                    REPO_ROOT, logs / f"{name}.log", complete_model(model),
                )
        for generation_seed in args.generation_seeds:
            output = run_root / "synthetic" / "woof" / "ipc10" / "frozen"
            add_generation_group(
                tasks, index, args, 3, "woof_ipc10_frozen", woof_root, "woof", 10, prototype, dcs,
                args.base_model, "frozen", ("label", "correct", "shuffled"), output, generation_seed, None,
                dependency=(artifact_name,),
            )
            for training_seed in args.woof_training_seeds:
                for mode, prompts in (
                    ("empty_ft", ("label",)),
                    ("unpaired_ft", ("label", "correct", "shuffled")),
                    ("matched_ft", ("label", "correct", "shuffled")),
                ):
                    model = run_root / "models" / "woof" / f"train_seed_{training_seed}" / mode
                    output = run_root / "synthetic" / "woof" / "ipc10" / f"train_seed_{training_seed}"
                    add_generation_group(
                        tasks, index, args, 3, f"woof_ipc10_s{training_seed}", woof_root, "woof", 10,
                        prototype, dcs, model, mode, prompts, output, generation_seed, training_seed,
                        dependency=(artifact_name, train_names[(training_seed, mode)]),
                    )
    return tasks, index


def write_manifest(args, index):
    run_root = Path(args.run_root).resolve()
    payload = {
        "format_version": 1, "phases": list(args.phases),
        "nette_data_root": str(Path(args.nette_data_root).resolve()),
        "nette_caption_file": str(Path(args.nette_caption_file).resolve()),
        "woof_data_root": str(Path(args.woof_data_root).resolve()) if args.woof_data_root else None,
        "woof_caption_file": str(Path(args.woof_caption_file).resolve()) if args.woof_caption_file else None,
        "base_model": str(Path(args.base_model).resolve()),
        "base_run_root": str(Path(args.base_run_root).resolve()),
        "causal_run_root": str(Path(args.causal_run_root).resolve()),
        "new_training_seeds": list(args.new_training_seeds),
        "ipc_training_seeds": list(args.ipc_training_seeds),
        "woof_training_seeds": list(args.woof_training_seeds),
        "generation_seeds": list(args.generation_seeds), "ipc_values": list(args.ipc_values),
        "classifier_repeats": args.classifier_repeats,
        "effective_training_batch": args.train_batch_size * args.gradient_accumulation_steps,
    }
    path = run_root / "run_manifest.json"
    if path.is_file() and json.loads(path.read_text(encoding="utf-8")) != payload:
        raise RuntimeError(f"Resume configuration differs from {path}")
    if not path.is_file():
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (run_root / "evaluation_index.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def quote(command):
    return " ".join(shlex.quote(str(item)) for item in command)


def launch(task, gpu, args):
    command = list(task.command)
    if task.kind == "train":
        if "--main_process_port" not in command:
            command[2:2] = ["--main_process_port", str(29600 + int(gpu))]
        output = Path(command[command.index("--output-dir") + 1])
        checkpoints = list(output.glob("checkpoint-*")) if output.is_dir() else []
        if checkpoints and "--resume-from-checkpoint" not in command:
            command.extend(("--resume-from-checkpoint", "latest"))
        elif output.is_dir() and any(output.iterdir()) and not complete_model(output)():
            archived = output.with_name(f"{output.name}.incomplete_{time.strftime('%Y%m%dT%H%M%S')}")
            output.replace(archived)
            print(f"ARCHIVE non-resumable training output: {output} -> {archived}", flush=True)
    task.log.parent.mkdir(parents=True, exist_ok=True)
    task.handle = task.log.open("a", encoding="utf-8", buffering=1)
    task.handle.write(f"\n[{time.strftime('%F %T')}] attempt {task.attempts + 1} GPU {gpu}\n{quote(command)}\n")
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": gpu, "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false"})
    if args.diffusers_src:
        env["PYTHONPATH"] = args.diffusers_src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    task.process = subprocess.Popen(command, cwd=task.cwd, env=env, stdout=task.handle, stderr=subprocess.STDOUT)
    task.attempts += 1
    print(f"LAUNCH stage {task.stage} GPU {gpu}: {task.name} (attempt {task.attempts})", flush=True)


def main():
    args = parse_args()
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not gpus:
        raise ValueError("At least one GPU is required")
    if args.train_batch_size * args.gradient_accumulation_steps != 32:
        raise ValueError("Each one-GPU fine-tune must keep effective batch size 32")
    run_root = Path(args.run_root).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    validate_subset(args.nette_data_root, "nette", args.nette_caption_file)
    validate_caption_coverage(args.nette_data_root, args.nette_caption_file)
    if "woof" in args.phases:
        if not args.woof_data_root or not args.woof_caption_file:
            raise ValueError("The woof phase requires --woof-data-root and --woof-caption-file")
        validate_subset(args.woof_data_root, "woof", args.woof_caption_file)
        validate_caption_coverage(args.woof_data_root, args.woof_caption_file)
    for path in (args.base_model, args.base_run_root, args.causal_run_root):
        if not Path(path).exists():
            raise FileNotFoundError(path)
    tasks, index = build_tasks(args)
    write_manifest(args, index)
    completed = {name for name, task in tasks.items() if task.complete()}
    running = {}
    print(f"Generality pipeline: {len(completed)}/{len(tasks)} tasks already complete", flush=True)
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
        incomplete_stages = [task.stage for name, task in tasks.items() if name not in completed]
        active_stage = min(incomplete_stages)
        free = [gpu for gpu in gpus if gpu not in running]
        evals = sum(task.kind == "eval" for task in running.values())
        ready = [
            task for task in tasks.values()
            if task.stage == active_stage and task.name not in completed and task.process is None
            and task.next_ready <= now and all(dep in completed for dep in task.dependencies)
        ]
        ready.sort(key=lambda task: ({"train": 0, "artifact": 1, "generate": 2, "eval": 3}[task.kind], task.attempts, task.name))
        for gpu in free:
            selected = next((task for task in ready if task.kind != "eval" or evals < args.max_parallel_evals), None)
            if selected is None:
                break
            ready.remove(selected)
            launch(selected, gpu, args)
            running[gpu] = selected
            if selected.kind == "eval":
                evals += 1
        time.sleep(5)
    subprocess.run([
        sys.executable, str(HERE / "summarize_generality.py"), "--evaluation-index",
        str(run_root / "evaluation_index.json"), "--output-dir", str(run_root / "summary"),
    ], cwd=REPO_ROOT, check=True)
    (run_root / "COMPLETE").write_text(time.strftime("%F %T\n"), encoding="utf-8")
    print(f"Generality pipeline complete: {run_root}")


if __name__ == "__main__":
    main()
