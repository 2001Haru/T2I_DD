#!/usr/bin/env python3
"""Resume-safe scheduler for prompt utility across prototype initialization strengths."""

import argparse
import json
import signal
import subprocess
import sys
import time
from pathlib import Path

from run_generality import (
    EVAL_DIR,
    REPO_ROOT,
    Task,
    complete_eval,
    complete_generation,
    eval_command,
    launch,
    write_scheduler_state,
)


HERE = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nette-data-root", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--base-run-root", required=True, help="Original training-seed-0 factorial")
    parser.add_argument("--causal-run-root", required=True, help="Causal-ladder run containing training seed 1")
    parser.add_argument("--generality-run-root", required=True, help="Run containing IPC50 artifacts/results")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--strengths", type=float, nargs="+", default=(0.7, 0.8, 0.9, 1.0))
    parser.add_argument("--ipc-values", type=int, nargs="+", default=(10, 50))
    parser.add_argument("--training-seeds", type=int, nargs="+", default=(0, 1))
    parser.add_argument("--generation-seeds", type=int, nargs="+", default=(0, 1))
    parser.add_argument("--prompts", nargs="+", choices=("label", "correct", "shuffled"), default=("label", "correct"))
    parser.add_argument("--guidance-scale", type=float, default=10.0)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--classifier-repeats", type=int, default=2)
    parser.add_argument("--classifier-seed", type=int, default=0)
    parser.add_argument("--max-parallel-evals", type=int, default=1)
    parser.add_argument("--retry-delay-seconds", type=int, default=120)
    parser.add_argument("--max-retries", type=int, default=0, help="0 retries forever")
    parser.add_argument("--diffusers-src", default="")
    parser.add_argument("--disable-reuse-0p7", action="store_true")
    return parser.parse_args()


def strength_token(value):
    return f"{float(value):g}".replace(".", "p")


def model_path(args, training_seed):
    if training_seed == 0:
        return Path(args.base_run_root).resolve() / "models" / "matched_ft"
    if training_seed == 1:
        return Path(args.causal_run_root).resolve() / "models" / "train_seed_1" / "matched_ft"
    raise ValueError(f"No matched_ft checkpoint mapping for training seed {training_seed}")


def artifact_paths(args, ipc):
    if ipc == 10:
        root = Path(args.base_run_root).resolve() / "prototypes"
        return root / "text_supervision-ipc10-0.7-30-kmexpand1.json", root / "dcs.json"
    root = Path(args.generality_run_root).resolve() / "artifacts" / "nette" / f"ipc{ipc}"
    return root / f"nette-ipc{ipc}-0.7-30-kmexpand1.json", root / "dcs.json"


def generation_command(args, prototype, dcs, model, prompts, output_root, generation_seed, ipc, strength):
    return [
        sys.executable, str(HERE / "generate_factorial.py"),
        "--prototype", str(prototype), "--dcs", str(dcs), "--base-model", args.base_model,
        "--model", f"matched_ft={model}", "--supervisions", "matched_ft",
        "--prompts", *prompts, "--output-root", str(output_root),
        "--generation-seeds", str(generation_seed), "--ipc", str(ipc),
        "--strength", str(strength), "--guidance-scale", str(args.guidance_scale),
        "--num-inference-steps", str(args.num_inference_steps), "--shuffle-shift", "1",
        "--size", "256", "--resume",
    ]


def expected_reuse(args, ipc, training_seed, generation_seed, prompt):
    """Return a prior synthetic/log pair when the exact 0.7 cell exists."""
    if ipc == 10:
        if training_seed == 0:
            root = Path(args.base_run_root).resolve()
            synthetic = root / "synthetic" / f"seed_{generation_seed}" / f"matched_ft_{prompt}"
            log = root / "evaluation" / f"seed_{generation_seed}" / f"matched_ft_{prompt}.log"
        elif training_seed == 1:
            root = Path(args.causal_run_root).resolve()
            synthetic = root / "synthetic" / "train_seed_1" / f"seed_{generation_seed}" / f"matched_ft_{prompt}"
            log = root / "evaluation" / "train_seed_1" / f"seed_{generation_seed}" / f"matched_ft_{prompt}.log"
        else:
            return None
    else:
        root = Path(args.generality_run_root).resolve()
        synthetic = root / "synthetic" / "nette" / f"ipc{ipc}" / f"train_seed_{training_seed}" / f"seed_{generation_seed}" / f"matched_ft_{prompt}"
        log = root / "evaluation" / f"nette_ipc{ipc}_s{training_seed}" / f"seed_{generation_seed}" / f"matched_ft_{prompt}.log"
    return synthetic, log


def reusable_cell(args, ipc, training_seed, generation_seed, prompt):
    if args.disable_reuse_0p7:
        return None
    candidate = expected_reuse(args, ipc, training_seed, generation_seed, prompt)
    if candidate is None:
        return None
    synthetic, log = candidate
    manifest_path = synthetic / "manifest.json"
    if not complete_eval(log)() or not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = {
        "ipc": ipc,
        "generation_seed": generation_seed,
        "strength": 0.7,
        "guidance_scale": args.guidance_scale,
        "num_inference_steps": args.num_inference_steps,
        "supervision_mode": "matched_ft",
        "prompt_mode": prompt,
    }
    if any(manifest.get(key) != value for key, value in required.items()):
        return None
    if not complete_generation(synthetic.parent, [synthetic.name], 10 * ipc)():
        return None
    return synthetic, log


def build_tasks(args):
    tasks, index = {}, []
    run_root = Path(args.run_root).resolve()
    for ipc in args.ipc_values:
        prototype, dcs = artifact_paths(args, ipc)
        for artifact in (prototype, dcs):
            if not artifact.is_file():
                raise FileNotFoundError(artifact)
        for training_seed in args.training_seeds:
            model = model_path(args, training_seed)
            if not (model / "model_index.json").is_file():
                raise FileNotFoundError(model / "model_index.json")
            for strength in args.strengths:
                token = strength_token(strength)
                for generation_seed in args.generation_seeds:
                    missing = []
                    for prompt in args.prompts:
                        reused = reusable_cell(args, ipc, training_seed, generation_seed, prompt) if abs(strength - 0.7) < 1e-9 else None
                        metadata = {
                            "spec": "nette", "ipc": ipc, "strength": strength,
                            "guidance_scale": args.guidance_scale, "num_inference_steps": args.num_inference_steps,
                            "training_seed": training_seed, "generation_seed": generation_seed,
                            "supervision": "matched_ft", "prompt": prompt,
                        }
                        if reused:
                            synthetic, log = reused
                            index.append({**metadata, "synthetic_dir": str(synthetic), "evaluation_log": str(log), "source": "reused_0p7"})
                        else:
                            missing.append(prompt)
                    if not missing:
                        continue
                    prefix = f"ipc{ipc}_s{training_seed}_str{token}_g{generation_seed}"
                    output_root = run_root / "synthetic" / "nette" / f"ipc{ipc}" / f"strength_{token}" / f"train_seed_{training_seed}"
                    seed_root = output_root / f"seed_{generation_seed}"
                    conditions = [f"matched_ft_{prompt}" for prompt in missing]
                    generation_name = f"gen_{prefix}"
                    tasks[generation_name] = Task(
                        generation_name, 1, "generate",
                        generation_command(args, prototype, dcs, model, missing, output_root, generation_seed, ipc, strength),
                        REPO_ROOT, run_root / "scheduler_logs" / f"{generation_name}.log",
                        complete_generation(seed_root, conditions, 10 * ipc),
                    )
                    for prompt, condition in zip(missing, conditions):
                        eval_name = f"eval_{prefix}_{prompt}"
                        log = run_root / "evaluation" / "nette" / f"ipc{ipc}" / f"strength_{token}" / f"train_seed_{training_seed}" / f"seed_{generation_seed}" / f"{condition}.log"
                        synthetic = seed_root / condition
                        command = eval_command(args, synthetic, args.nette_data_root, ipc, "nette", eval_name)
                        tasks[eval_name] = Task(
                            eval_name, 1, "eval", command, EVAL_DIR, log, complete_eval(log),
                            dependencies=(generation_name,),
                        )
                        index.append({
                            "spec": "nette", "ipc": ipc, "strength": strength,
                            "guidance_scale": args.guidance_scale, "num_inference_steps": args.num_inference_steps,
                            "training_seed": training_seed, "generation_seed": generation_seed,
                            "supervision": "matched_ft", "prompt": prompt,
                            "synthetic_dir": str(synthetic), "evaluation_log": str(log), "source": "new",
                        })
    return tasks, index


def write_configuration(args, index):
    run_root = Path(args.run_root).resolve()
    payload = {
        "format_version": 1,
        "question": "How prototype initialization strength changes the marginal downstream value of DCS text",
        "nette_data_root": str(Path(args.nette_data_root).resolve()),
        "base_model": str(Path(args.base_model).resolve()),
        "base_run_root": str(Path(args.base_run_root).resolve()),
        "causal_run_root": str(Path(args.causal_run_root).resolve()),
        "generality_run_root": str(Path(args.generality_run_root).resolve()),
        "strengths": list(args.strengths), "ipc_values": list(args.ipc_values),
        "training_seeds": list(args.training_seeds), "generation_seeds": list(args.generation_seeds),
        "prompts": list(args.prompts), "guidance_scale": args.guidance_scale,
        "num_inference_steps": args.num_inference_steps,
        "classifier_repeats": args.classifier_repeats, "classifier_seed": args.classifier_seed,
        "reuse_0p7": not args.disable_reuse_0p7,
    }
    path = run_root / "run_manifest.json"
    if path.is_file() and json.loads(path.read_text(encoding="utf-8")) != payload:
        raise RuntimeError(f"Resume configuration differs from {path}")
    if not path.is_file():
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (run_root / "evaluation_index.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not gpus:
        raise ValueError("At least one GPU is required")
    for path in (args.nette_data_root, args.base_model, args.base_run_root, args.causal_run_root, args.generality_run_root):
        if not Path(path).exists():
            raise FileNotFoundError(path)
    if len(set(args.strengths)) != len(args.strengths) or any(not 0 < value <= 1 for value in args.strengths):
        raise ValueError("Strengths must be unique and in (0, 1]")
    run_root = Path(args.run_root).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    tasks, index = build_tasks(args)
    write_configuration(args, index)
    completed = {name for name, task in tasks.items() if task.complete()}
    running = {}
    reused = sum(row["source"] == "reused_0p7" for row in index)
    print(f"Strength interaction: {len(index)} cells, {reused} reused, {len(completed)}/{len(tasks)} tasks complete", flush=True)
    write_scheduler_state(run_root, tasks, completed, running)
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
        evals = sum(task.kind == "eval" for task in running.values())
        ready = [
            task for task in tasks.values()
            if task.name not in completed and task.process is None and task.next_ready <= now
            and all(dependency in completed for dependency in task.dependencies)
        ]
        ready.sort(key=lambda task: ({"generate": 0, "eval": 1}[task.kind], task.attempts, task.name))
        for gpu in free:
            selected = next((task for task in ready if task.kind != "eval" or evals < args.max_parallel_evals), None)
            if selected is None:
                break
            ready.remove(selected)
            launch(selected, gpu, args)
            running[gpu] = selected
            evals += selected.kind == "eval"
        write_scheduler_state(run_root, tasks, completed, running)
        time.sleep(5)
    subprocess.run([
        sys.executable, str(HERE / "summarize_strength_interaction.py"),
        "--evaluation-index", str(run_root / "evaluation_index.json"),
        "--output-dir", str(run_root / "summary"),
    ], cwd=REPO_ROOT, check=True)
    (run_root / "COMPLETE").write_text(time.strftime("%F %T\n"), encoding="utf-8")
    print(f"Strength interaction complete: {run_root}", flush=True)


if __name__ == "__main__":
    main()
