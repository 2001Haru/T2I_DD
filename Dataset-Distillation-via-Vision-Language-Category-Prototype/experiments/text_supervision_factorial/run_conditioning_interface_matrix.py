#!/usr/bin/env python3
"""Persistent four-GPU scheduler for the preregistered A/B/C conditioning-interface matrix."""

import argparse
import json
import signal
import subprocess
import sys
import time
from pathlib import Path

from run_generality import (
    DISTILLATION_DIR,
    EVAL_DIR,
    REPO_ROOT,
    Task,
    complete_eval,
    complete_generation,
    complete_model,
    eval_command,
    launch,
    prototype_command,
    train_command,
    validate_caption_coverage,
    write_scheduler_state,
)
from subset_specs import validate_subset


HERE = Path(__file__).resolve().parent
CORE_STRENGTHS = (0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00)
CONTROL_STRENGTHS = (0.70, 0.80, 0.90, 1.00)
SHUFFLE_SHIFTS = (1, 2, 4, 7)
WOOF_TRAIN_SUPERVISION = {
    "empty_ft": "empty",
    "constant_ft": "constant",
    "label_ft": "label",
    "unpaired_ft": "unpaired",
    "matched_ft": "matched",
}
WOOF_PHASES = ("ladder", "curve_ipc10_20", "curve_ipc50")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nette-data-root", required=True)
    parser.add_argument("--nette-caption-file", required=True)
    parser.add_argument("--woof-data-root")
    parser.add_argument("--woof-caption-file")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--base-run-root", required=True)
    parser.add_argument("--causal-run-root", required=True)
    parser.add_argument("--generality-run-root", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--reuse-index", action="append", default=[])
    parser.add_argument(
        "--allow-d-regeneration", action="store_true",
        help="Allow Matrix D to regenerate existing Label/Correct IPC50 cells instead of requiring reuse",
    )
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--matrices", nargs="+", choices=("A", "B", "C", "D"), default=("A", "B", "C"))
    parser.add_argument("--woof-phases", nargs="+", choices=WOOF_PHASES, default=WOOF_PHASES)
    parser.add_argument("--training-seeds", type=int, nargs="+", default=(0, 1))
    parser.add_argument("--generation-seeds", type=int, nargs="+", default=(0, 1))
    parser.add_argument("--classifier-repeats", type=int, default=2)
    parser.add_argument("--classifier-seed", type=int, default=0)
    parser.add_argument("--guidance-scale", type=float, default=10.0)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--mixed-precision", choices=("no", "fp16", "bf16"), default="fp16")
    parser.add_argument("--max-parallel-evals", type=int, default=4)
    parser.add_argument("--retry-delay-seconds", type=int, default=120)
    parser.add_argument("--max-retries", type=int, default=0, help="0 retries forever")
    parser.add_argument(
        "--max-walltime-hours", type=float, default=0.0,
        help="Stop launching new tasks after this many hours, then wait for active tasks and exit cleanly; 0 disables",
    )
    parser.add_argument("--diffusers-src", default="")
    return parser.parse_args()


def strength_token(value):
    if value is None:
        return "pure_noise"
    return f"strength_{float(value):g}".replace(".", "p")


def cell_key(row):
    shift = int(row.get("shuffle_shift", 1)) if row["prompt"] == "shuffled" else None
    strength = row.get("strength", 0.7)
    strength = None if row.get("visual_mode", "prototype") == "pure_noise" else round(float(strength), 8)
    training_seed = row.get("training_seed")
    return (
        row["spec"], int(row["ipc"]), row.get("visual_mode", "prototype"), strength,
        row["supervision"], training_seed, int(row["generation_seed"]), row["prompt"], shift,
    )


def load_reuse_catalog(paths):
    catalog = {}
    for index_path in paths:
        index_path = Path(index_path).resolve()
        if not index_path.is_file():
            raise FileNotFoundError(index_path)
        for raw in json.loads(index_path.read_text(encoding="utf-8")):
            if not complete_eval(Path(raw["evaluation_log"]))():
                continue
            row = dict(raw)
            row.setdefault("strength", 0.7)
            row.setdefault("visual_mode", "prototype")
            row.setdefault("shuffle_shift", 1)
            key = cell_key(row)
            if key not in catalog:
                catalog[key] = row
    return catalog


def nette_artifacts(args, ipc):
    if ipc == 10:
        root = Path(args.base_run_root).resolve() / "prototypes"
        return root / "text_supervision-ipc10-0.7-30-kmexpand1.json", root / "dcs.json"
    root = Path(args.generality_run_root).resolve() / "artifacts" / "nette" / f"ipc{ipc}"
    return root / f"nette-ipc{ipc}-0.7-30-kmexpand1.json", root / "dcs.json"


def nette_model(args, supervision, training_seed):
    if supervision == "frozen":
        return Path(args.base_model).resolve()
    if supervision == "matched_ft" and training_seed == 0:
        return Path(args.base_run_root).resolve() / "models" / "matched_ft"
    return Path(args.causal_run_root).resolve() / "models" / f"train_seed_{training_seed}" / supervision


def generation_command(args, prototype, dcs, model, supervision, prompts, output_root,
                       generation_seed, ipc, strength, visual_mode, shuffle_shift):
    command = [
        sys.executable, str(HERE / "generate_factorial.py"),
        "--prototype", str(prototype), "--dcs", str(dcs), "--base-model", args.base_model,
    ]
    if supervision != "frozen":
        command.extend(("--model", f"{supervision}={model}"))
    command.extend((
        "--supervisions", supervision, "--prompts", *prompts,
        "--output-root", str(output_root), "--generation-seeds", str(generation_seed),
        "--ipc", str(ipc), "--strength", str(strength if strength is not None else 1.0),
        "--visual-mode", visual_mode, "--guidance-scale", str(args.guidance_scale),
        "--num-inference-steps", str(args.num_inference_steps), "--shuffle-shift", str(shuffle_shift),
        "--size", "256", "--resume",
    ))
    return command


def add_generation(tasks, index, args, reuse, matrix, spec, data_root, ipc, prototype, dcs,
                   model, supervision, training_seed, generation_seed, strength, visual_mode,
                   prompts, shuffle_shift, dependencies=(), stage=None, phase=None):
    visual_token = strength_token(strength)
    seed_token = "frozen" if training_seed is None else f"train_seed_{training_seed}"
    shift_token = f"shift_{shuffle_shift}"
    output_root = (
        Path(args.run_root).resolve() / "synthetic" / matrix / spec / f"ipc{ipc}" /
        visual_token / seed_token / shift_token
    )
    missing = []
    for prompt in prompts:
        metadata = {
            "matrix": matrix, "spec": spec, "ipc": ipc, "visual_mode": visual_mode,
            "strength": strength, "guidance_scale": args.guidance_scale,
            "num_inference_steps": args.num_inference_steps, "supervision": supervision,
            "training_seed": training_seed, "generation_seed": generation_seed,
            "prompt": prompt, "shuffle_shift": shuffle_shift if prompt == "shuffled" else None,
            "phase": phase,
        }
        reused = reuse.get(cell_key(metadata))
        if reused:
            index.append({**metadata, "evaluation_log": reused["evaluation_log"], "source": "reused"})
        else:
            missing.append(prompt)
    if not missing:
        return

    prefix = (
        f"{matrix}_{spec}_ipc{ipc}_{visual_token}_{seed_token}_{supervision}_"
        f"g{generation_seed}_s{shuffle_shift}_" +
        "-".join(missing)
    )
    generation_name = f"gen_{prefix}"
    seed_root = output_root / f"seed_{generation_seed}"
    conditions = [f"{supervision}_{prompt}" for prompt in missing]
    tasks[generation_name] = Task(
        generation_name, ord(matrix) if stage is None else stage, "generate",
        generation_command(
            args, prototype, dcs, model, supervision, missing, output_root,
            generation_seed, ipc, strength, visual_mode, shuffle_shift,
        ),
        REPO_ROOT, Path(args.run_root) / "scheduler_logs" / f"{generation_name}.log",
        complete_generation(seed_root, conditions, 10 * ipc), dependencies=tuple(dependencies),
    )
    for prompt, condition in zip(missing, conditions):
        eval_name = f"eval_{prefix}_{prompt}"
        log = (
            Path(args.run_root) / "evaluation" / matrix / spec / f"ipc{ipc}" / visual_token /
            seed_token / shift_token / f"seed_{generation_seed}" / f"{condition}.log"
        )
        synthetic = seed_root / condition
        tasks[eval_name] = Task(
            eval_name, ord(matrix) if stage is None else stage, "eval",
            eval_command(args, synthetic, data_root, ipc, spec, eval_name),
            EVAL_DIR, log, complete_eval(log), dependencies=(generation_name,),
        )
        index.append({
            "matrix": matrix, "spec": spec, "ipc": ipc, "visual_mode": visual_mode,
            "strength": strength, "guidance_scale": args.guidance_scale,
            "num_inference_steps": args.num_inference_steps, "supervision": supervision,
            "training_seed": training_seed, "generation_seed": generation_seed,
            "prompt": prompt, "shuffle_shift": shuffle_shift if prompt == "shuffled" else None,
            "phase": phase,
            "synthetic_dir": str(synthetic), "evaluation_log": str(log), "source": "new",
        })


def add_prompt_grid(tasks, index, args, reuse, matrix, spec, data_root, ipc, prototype, dcs,
                    model, supervision, training_seed, generation_seed, strength, visual_mode,
                    extra_shifts=(), dependencies=(), stage=None, phase=None):
    add_generation(
        tasks, index, args, reuse, matrix, spec, data_root, ipc, prototype, dcs, model,
        supervision, training_seed, generation_seed, strength, visual_mode,
        ("label", "correct", "shuffled"), 1, dependencies, stage, phase,
    )
    for shift in extra_shifts:
        add_generation(
            tasks, index, args, reuse, matrix, spec, data_root, ipc, prototype, dcs, model,
            supervision, training_seed, generation_seed, strength, visual_mode,
            ("shuffled",), shift, dependencies, stage, phase,
        )


def build_tasks(args, reuse):
    tasks, index = {}, []
    run_root = Path(args.run_root).resolve()
    logs = run_root / "scheduler_logs"
    logs.mkdir(parents=True, exist_ok=True)
    nette_root = Path(args.nette_data_root).resolve()
    woof_root = Path(args.woof_data_root).resolve() if args.woof_data_root else None

    # Targeted ImageNette follow-up: existing Correct/Label cells are reused and
    # only missing IPC50 Shuffled-S1 cells are generated in practice.
    if "D" in args.matrices:
        ipc = 50
        prototype, dcs = nette_artifacts(args, ipc)
        for artifact in (prototype, dcs):
            if not artifact.is_file():
                raise FileNotFoundError(artifact)
        for training_seed in args.training_seeds:
            model = nette_model(args, "matched_ft", training_seed)
            if not complete_model(model)():
                raise RuntimeError(f"Missing Matched-FT checkpoint: {model}")
            for strength in CONTROL_STRENGTHS:
                for generation_seed in args.generation_seeds:
                    add_prompt_grid(
                        tasks, index, args, reuse, "D", "nette", nette_root, ipc, prototype, dcs,
                        model, "matched_ft", training_seed, generation_seed, strength, "prototype",
                        stage=1, phase="nette_ipc50_correspondence",
                    )

    if "A" in args.matrices:
        for ipc in (10, 20, 50):
            prototype, dcs = nette_artifacts(args, ipc)
            for artifact in (prototype, dcs):
                if not artifact.is_file():
                    raise FileNotFoundError(artifact)
            for training_seed in args.training_seeds:
                model = nette_model(args, "matched_ft", training_seed)
                if not complete_model(model)():
                    raise RuntimeError(f"Missing Matched-FT checkpoint: {model}")
                for strength in CORE_STRENGTHS:
                    extra = (2, 4, 7) if strength in CONTROL_STRENGTHS else ()
                    for generation_seed in args.generation_seeds:
                        add_prompt_grid(
                            tasks, index, args, reuse, "A", "nette", nette_root, ipc, prototype, dcs,
                            model, "matched_ft", training_seed, generation_seed, strength, "prototype", extra,
                        )

    if "B" in args.matrices:
        for ipc in (10, 50):
            prototype, dcs = nette_artifacts(args, ipc)
            for artifact in (prototype, dcs):
                if not artifact.is_file():
                    raise FileNotFoundError(artifact)
            regimes = [("frozen", None)] + [("empty_ft", seed) for seed in args.training_seeds]
            for supervision, training_seed in regimes:
                model = nette_model(args, supervision, training_seed)
                if supervision != "frozen" and not complete_model(model)():
                    raise RuntimeError(f"Missing {supervision} checkpoint: {model}")
                for strength in (*CONTROL_STRENGTHS, None):
                    visual_mode = "pure_noise" if strength is None else "prototype"
                    for generation_seed in args.generation_seeds:
                        add_prompt_grid(
                            tasks, index, args, reuse, "B", "nette", nette_root, ipc, prototype, dcs,
                            model, supervision, training_seed, generation_seed, strength, visual_mode,
                        )

    if "C" in args.matrices:
        woof_caption = Path(args.woof_caption_file).resolve()
        train_names = {}
        if "ladder" in args.woof_phases:
            required_supervisions = tuple(WOOF_TRAIN_SUPERVISION)
        elif "curve_ipc10_20" in args.woof_phases:
            required_supervisions = ("empty_ft", "matched_ft")
        else:
            required_supervisions = ("matched_ft",)
        for supervision in required_supervisions:
            train_supervision = WOOF_TRAIN_SUPERVISION[supervision]
            for training_seed in args.training_seeds:
                model = run_root / "models" / "woof" / f"train_seed_{training_seed}" / supervision
                name = f"train_C_woof_s{training_seed}_{supervision}"
                train_names[(supervision, training_seed)] = name
                tasks[name] = Task(
                    name, 2, "train",
                    train_command(args, woof_root, woof_caption, model, train_supervision, training_seed),
                    REPO_ROOT, logs / f"{name}.log", complete_model(model),
                )

        required_ipcs = {10}
        if "curve_ipc10_20" in args.woof_phases:
            required_ipcs.add(20)
        if "curve_ipc50" in args.woof_phases:
            required_ipcs.add(50)
        artifacts = {}
        for ipc in sorted(required_ipcs):
            artifact_root = run_root / "artifacts" / "woof" / f"ipc{ipc}"
            prototype = artifact_root / f"woof-ipc{ipc}-0.7-30-kmexpand1.json"
            dcs = artifact_root / "dcs.json"
            artifact_name = f"artifact_C_woof_ipc{ipc}"
            artifacts[ipc] = (prototype, dcs, artifact_name)
            tasks[artifact_name] = Task(
                artifact_name, 2 if ipc == 10 else (3 if ipc == 20 else 4), "artifact",
                prototype_command(args, "woof", woof_root, woof_caption, ipc, artifact_root, dcs),
                DISTILLATION_DIR, logs / f"{artifact_name}.log",
                lambda p=prototype, d=dcs: p.is_file() and d.is_file(),
            )

        # C1: the full causal ladder at one established visual interface and at
        # pure noise. This is the minimum complete ImageWoof replication.
        if "ladder" in args.woof_phases:
            prototype, dcs, artifact_name = artifacts[10]
            regimes = [("frozen", None)] + [
                (supervision, seed)
                for supervision in WOOF_TRAIN_SUPERVISION
                for seed in args.training_seeds
            ]
            for supervision, training_seed in regimes:
                if supervision == "frozen":
                    model = Path(args.base_model).resolve()
                    dependencies = (artifact_name,)
                else:
                    model = run_root / "models" / "woof" / f"train_seed_{training_seed}" / supervision
                    dependencies = (train_names[(supervision, training_seed)], artifact_name)
                for strength in (0.70, None):
                    visual_mode = "pure_noise" if strength is None else "prototype"
                    for generation_seed in args.generation_seeds:
                        add_prompt_grid(
                            tasks, index, args, reuse, "C", "woof", woof_root, 10, prototype, dcs,
                            model, supervision, training_seed, generation_seed, strength, visual_mode,
                            dependencies=dependencies, stage=2, phase="woof_causal_ladder",
                        )

        # C2/C3: Matched-FT strength curves. IPC10 omits 0.70 because C1 already
        # owns that exact cell; IPC20/50 use all four preregistered strengths.
        curve_specs = []
        if "curve_ipc10_20" in args.woof_phases:
            curve_specs.extend((
                (10, (0.80, 0.90, 1.00), 3, ("frozen", "empty_ft", "matched_ft")),
                (20, CONTROL_STRENGTHS, 3, ("matched_ft",)),
            ))
        if "curve_ipc50" in args.woof_phases:
            curve_specs.append((50, CONTROL_STRENGTHS, 4, ("matched_ft",)))
        for ipc, strengths, stage, supervisions in curve_specs:
            prototype, dcs, artifact_name = artifacts[ipc]
            regimes = [
                (supervision, None if supervision == "frozen" else training_seed)
                for supervision in supervisions
                for training_seed in ((None,) if supervision == "frozen" else args.training_seeds)
            ]
            for supervision, training_seed in regimes:
                if supervision == "frozen":
                    model = Path(args.base_model).resolve()
                    dependencies = (artifact_name,)
                else:
                    model = run_root / "models" / "woof" / f"train_seed_{training_seed}" / supervision
                    dependencies = (train_names[(supervision, training_seed)], artifact_name)
                for strength in strengths:
                    for generation_seed in args.generation_seeds:
                        add_prompt_grid(
                            tasks, index, args, reuse, "C", "woof", woof_root, ipc, prototype, dcs,
                            model, supervision, training_seed, generation_seed, strength, "prototype",
                            dependencies=dependencies, stage=stage,
                            phase=f"woof_strength_curve_ipc{ipc}",
                        )
    return tasks, index


def write_manifest(args, index):
    run_root = Path(args.run_root).resolve()
    payload = {
        "format_version": 1,
        "question": "Prompt utility across visual interfaces, budgets, checkpoints, and datasets",
        "matrices": list(args.matrices), "g_condition_removed": True,
        "matrix_A": {
            "spec": "nette", "supervision": "matched_ft", "ipc": [10, 20, 50],
            "strengths": list(CORE_STRENGTHS), "base_shuffle_shift": 1,
            "extra_shuffle_shifts": list(SHUFFLE_SHIFTS[1:]),
            "extra_shift_strengths": list(CONTROL_STRENGTHS),
        },
        "matrix_B": {
            "spec": "nette", "supervisions": ["frozen", "empty_ft"], "ipc": [10, 50],
            "strengths": list(CONTROL_STRENGTHS), "pure_noise": True, "shuffle_shift": 1,
        },
        "matrix_C": {
            "spec": "woof", "phases": list(args.woof_phases),
            "causal_ladder_supervisions": ["frozen", *WOOF_TRAIN_SUPERVISION],
            "causal_ladder_ipc": 10, "causal_ladder_interfaces": [0.7, "pure_noise"],
            "curve_ipc10_supervisions": ["frozen", "empty_ft", "matched_ft"],
            "curve_ipc20_50_supervision": "matched_ft", "curve_ipc": [10, 20, 50],
            "curve_strengths": list(CONTROL_STRENGTHS), "shuffle_shift": 1,
        },
        "matrix_D": {
            "spec": "nette", "supervision": "matched_ft", "ipc": 50,
            "strengths": list(CONTROL_STRENGTHS), "shuffle_shift": 1,
        },
        "prompts": ["label", "correct", "shuffled"],
        "training_seeds": list(args.training_seeds), "generation_seeds": list(args.generation_seeds),
        "classifier_repeats": args.classifier_repeats, "classifier_seed": args.classifier_seed,
        "guidance_scale": args.guidance_scale, "num_inference_steps": args.num_inference_steps,
        "base_model": str(Path(args.base_model).resolve()),
        "nette_data_root": str(Path(args.nette_data_root).resolve()),
        "nette_caption_file": str(Path(args.nette_caption_file).resolve()),
        "woof_data_root": str(Path(args.woof_data_root).resolve()) if args.woof_data_root else None,
        "woof_caption_file": str(Path(args.woof_caption_file).resolve()) if args.woof_caption_file else None,
        "base_run_root": str(Path(args.base_run_root).resolve()),
        "causal_run_root": str(Path(args.causal_run_root).resolve()),
        "generality_run_root": str(Path(args.generality_run_root).resolve()),
        "reuse_indexes": [str(Path(path).resolve()) for path in args.reuse_index],
    }
    path = run_root / "run_manifest.json"
    if path.is_file() and json.loads(path.read_text(encoding="utf-8")) != payload:
        raise RuntimeError(f"Resume configuration differs from {path}")
    if not path.is_file():
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (run_root / "evaluation_index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def choose_ready(tasks, completed, running, now, evals, max_evals):
    running_names = {task.name for task in running.values()}
    ready = [
        task for task in tasks.values()
        if task.name not in completed and task.name not in running_names and task.next_ready <= now
        and all(dependency in completed for dependency in task.dependencies)
    ]
    if not ready:
        return None
    # Honor scientific phase priority. Later phases may fill otherwise idle GPUs
    # only when no task from the current phase is ready.
    earliest_stage = min(task.stage for task in ready)
    ready = [task for task in ready if task.stage == earliest_stage]
    trains = sorted((task for task in ready if task.kind == "train"), key=lambda task: task.name)
    if trains:
        return trains[0]
    artifacts = sorted((task for task in ready if task.kind == "artifact"), key=lambda task: task.name)
    if artifacts:
        return artifacts[0]
    evaluations = sorted((task for task in ready if task.kind == "eval"), key=lambda task: (task.stage, task.name))
    if evaluations and evals < max_evals:
        return evaluations[0]
    generations = sorted((task for task in ready if task.kind == "generate"), key=lambda task: (task.stage, task.name))
    return generations[0] if generations else None


def main():
    args = parse_args()
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not gpus:
        raise ValueError("At least one GPU is required")
    if args.max_parallel_evals < 1:
        raise ValueError("--max-parallel-evals must be at least 1")
    if args.train_batch_size * args.gradient_accumulation_steps != 32:
        raise ValueError("Each one-GPU Woof fine-tune must keep effective batch size 32")
    validate_subset(args.nette_data_root, "nette", args.nette_caption_file)
    validate_caption_coverage(args.nette_data_root, args.nette_caption_file)
    if "C" in args.matrices:
        if not args.woof_data_root or not args.woof_caption_file:
            raise ValueError("Matrix C requires --woof-data-root and --woof-caption-file")
        validate_subset(args.woof_data_root, "woof", args.woof_caption_file)
        validate_caption_coverage(args.woof_data_root, args.woof_caption_file)
    for path in (args.base_model, args.base_run_root, args.causal_run_root, args.generality_run_root):
        if not Path(path).exists():
            raise FileNotFoundError(path)
    run_root = Path(args.run_root).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    reuse = load_reuse_catalog(args.reuse_index)
    tasks, index = build_tasks(args, reuse)
    if "D" in args.matrices and not args.allow_d_regeneration:
        reusable_d_controls = sum(
            row["matrix"] == "D" and row["prompt"] in {"label", "correct"}
            and row["source"] == "reused"
            for row in index
        )
        expected_d_controls = len(CONTROL_STRENGTHS) * len(args.training_seeds) * len(args.generation_seeds) * 2
        if reusable_d_controls != expected_d_controls:
            raise RuntimeError(
                f"Matrix D expected {expected_d_controls} reusable Label/Correct cells but found "
                f"{reusable_d_controls}. Add the completed strength-sweep evaluation_index.json, or pass "
                "--allow-d-regeneration intentionally."
            )
    write_manifest(args, index)
    completed = {name for name, task in tasks.items() if task.complete()}
    running = {}
    print(
        f"A/B/C matrix: {len(index)} evaluation cells ({sum(row['source'] == 'reused' for row in index)} reused), "
        f"{len(completed)}/{len(tasks)} tasks complete on GPUs {gpus}; "
        f"max parallel evals={min(args.max_parallel_evals, len(gpus))}",
        flush=True,
    )
    write_scheduler_state(run_root, tasks, completed, running)
    started_at = time.time()
    deadline = started_at + args.max_walltime_hours * 3600 if args.max_walltime_hours > 0 else None
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
                print(f"RETRY in {args.retry_delay_seconds}s: {task.name}; see {archived}", flush=True)
        if deadline is None or now < deadline:
            free = [gpu for gpu in gpus if gpu not in running]
            for gpu in free:
                evals = sum(task.kind == "eval" for task in running.values())
                selected = choose_ready(tasks, completed, running, now, evals, args.max_parallel_evals)
                if selected is None:
                    break
                launch(selected, gpu, args)
                running[gpu] = selected
        elif not running:
            print(
                f"Walltime reached after {(now - started_at) / 3600:.2f}h; "
                "active tasks finished and scheduler state is safe to resume.",
                flush=True,
            )
            write_scheduler_state(run_root, tasks, completed, running)
            subprocess.run([
                sys.executable, str(HERE / "summarize_conditioning_interface_matrix.py"),
                "--evaluation-index", str(run_root / "evaluation_index.json"),
                "--output-dir", str(run_root / "summary_partial"), "--allow-incomplete",
            ], cwd=REPO_ROOT, check=True)
            return
        write_scheduler_state(run_root, tasks, completed, running)
        time.sleep(5)
    subprocess.run([
        sys.executable, str(HERE / "summarize_conditioning_interface_matrix.py"),
        "--evaluation-index", str(run_root / "evaluation_index.json"),
        "--output-dir", str(run_root / "summary"),
    ], cwd=REPO_ROOT, check=True)
    (run_root / "COMPLETE").write_text(time.strftime("%F %T\n"), encoding="utf-8")
    print(f"Conditioning-interface matrices {','.join(args.matrices)} complete: {run_root}")


if __name__ == "__main__":
    main()
