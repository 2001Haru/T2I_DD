#!/usr/bin/env python3
"""Run the paired Sparse-m4/Matched-FT inference-interface transfer test."""

import argparse
import hashlib
import json
import signal
import subprocess
import sys
import time
from pathlib import Path

from run_generality import EVAL_DIR, REPO_ROOT, Task, complete_eval, eval_command
from run_sparse_prompt_search import launch, write_state


HERE = Path(__file__).resolve().parent
PROMPTS = ("label", "correct", "shuffled", "bank")
FAMILIES = ("sparse_m4_ft", "matched_ft")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--prototype", required=True)
    parser.add_argument("--dcs", required=True)
    parser.add_argument("--prompt-bank", required=True)
    parser.add_argument("--sparse-seed0-model", required=True)
    parser.add_argument("--sparse-seed1-model", required=True)
    parser.add_argument("--matched-seed0-model", required=True)
    parser.add_argument("--matched-seed1-model", required=True)
    parser.add_argument("--bank-copy", action="append", default=[])
    parser.add_argument("--reuse-index", action="append", default=[])
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--training-seeds", type=int, nargs="+", default=(0, 1))
    parser.add_argument("--generation-seeds", type=int, nargs="+", default=(0, 1))
    parser.add_argument("--ipc", type=int, default=50)
    parser.add_argument("--strength", type=float, default=0.8)
    parser.add_argument("--classifier-repeats", type=int, default=2)
    parser.add_argument("--classifier-seed", type=int, default=0)
    parser.add_argument("--guidance-scale", type=float, default=10.0)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--retry-delay-seconds", type=int, default=120)
    parser.add_argument("--max-retries", type=int, default=0)
    parser.add_argument("--diffusers-src", default="")
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_identity(path):
    path = Path(path).resolve()
    identity = {"path": str(path)}
    for name in ("model_index.json", "training_summary.json"):
        candidate = path / name
        identity[f"{name}_sha256"] = sha256(candidate) if candidate.is_file() else None
    return identity


def model_path(args, family, seed):
    return Path(getattr(args, f"{family.split('_')[0]}_seed{seed}_model")).resolve()


def condition(family, prompt):
    supervision = "sparse_ft" if family == "sparse_m4_ft" else "matched_ft"
    return f"{supervision}_{prompt}"


def complete_generation(root, family, prompts, expected):
    def check():
        for prompt in prompts:
            path = root / condition(family, prompt) / "complete.json"
            if not path.is_file() or int(json.loads(path.read_text(encoding="utf-8"))["images"]) != expected:
                return False
        return True
    return check


def family_from_row(row):
    if row.get("supervision") == "matched_ft":
        return "matched_ft"
    if row.get("supervision") == "sparse_ft" or str(row.get("method", "")).startswith("random_sparse"):
        if int(row.get("budget", 4)) == 4 and int(row.get("bank_seed", 0)) == 0:
            return "sparse_m4_ft"
    return None


def reuse_key(row):
    family = row.get("checkpoint_family") or family_from_row(row)
    if family not in FAMILIES:
        return None
    if row.get("spec", "nette") != "nette" or int(row.get("ipc", 50)) != 50:
        return None
    if abs(float(row.get("strength", 0.8)) - 0.8) > 1e-8:
        return None
    seed = row.get("training_seed")
    if seed is None:
        return None
    prompt = {
        "shuffled_s1": "shuffled",
        "sparse_bank": "bank",
    }.get(row.get("prompt"), row.get("prompt"))
    if prompt not in PROMPTS:
        return None
    return family, int(seed), int(row["generation_seed"]), prompt


def load_reuse(paths):
    catalog = {}
    for path in paths:
        path = Path(path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload if isinstance(payload, list) else payload.get("rows", payload.get("evaluation_index", []))
        for row in rows:
            key = reuse_key(row)
            log = Path(row.get("evaluation_log", ""))
            if key and complete_eval(log)():
                catalog.setdefault(key, dict(row))
    return catalog


def generation_command(args, family, model, output, generation_seed, prompts):
    supervision = "sparse_ft" if family == "sparse_m4_ft" else "matched_ft"
    return [
        sys.executable, str(HERE / "generate_factorial.py"),
        "--prototype", args.prototype, "--dcs", args.dcs,
        "--base-model", args.base_model, "--model", f"{supervision}={model}",
        "--supervisions", supervision, "--prompts", *prompts,
        "--prompt-bank", args.prompt_bank, "--output-root", str(output),
        "--generation-seeds", str(generation_seed), "--ipc", str(args.ipc),
        "--strength", str(args.strength), "--guidance-scale", str(args.guidance_scale),
        "--num-inference-steps", str(args.num_inference_steps), "--shuffle-shift", "1",
        "--size", "256", "--resume",
    ]


def build_tasks(args, reuse):
    root = Path(args.run_root).resolve()
    tasks, index = {}, []
    for family in FAMILIES:
        for training_seed in sorted(set(args.training_seeds)):
            model = model_path(args, family, training_seed)
            for generation_seed in sorted(set(args.generation_seeds)):
                missing = []
                for prompt in PROMPTS:
                    metadata = {
                        "experiment": "sparse_interface_transfer", "spec": "nette",
                        "ipc": args.ipc, "strength": args.strength,
                        "checkpoint_family": family, "training_seed": training_seed,
                        "generation_seed": generation_seed, "prompt": prompt,
                    }
                    reused = reuse.get((family, training_seed, generation_seed, prompt))
                    if reused:
                        index.append({**metadata, "evaluation_log": reused["evaluation_log"], "source": "reused"})
                    else:
                        missing.append(prompt)
                if not missing:
                    continue
                token = f"{family}_t{training_seed}_g{generation_seed}"
                output = root / "synthetic" / family / f"train_seed_{training_seed}"
                seed_root = output / f"seed_{generation_seed}"
                gen_name = f"gen_{token}_{'-'.join(missing)}"
                tasks[gen_name] = Task(
                    gen_name, 1, "generate",
                    generation_command(args, family, model, output, generation_seed, missing),
                    REPO_ROOT, root / "scheduler_logs" / f"{gen_name}.log",
                    complete_generation(seed_root, family, missing, 10 * args.ipc),
                )
                for prompt in missing:
                    eval_name = f"eval_{token}_{prompt}"
                    log = root / "evaluation" / family / f"train_seed_{training_seed}" / f"seed_{generation_seed}" / f"{prompt}.log"
                    tasks[eval_name] = Task(
                        eval_name, 1, "eval",
                        eval_command(args, seed_root / condition(family, prompt), args.data_root, args.ipc, "nette", eval_name),
                        EVAL_DIR, log, complete_eval(log), dependencies=(gen_name,),
                    )
                    index.append({
                        "experiment": "sparse_interface_transfer", "spec": "nette",
                        "ipc": args.ipc, "strength": args.strength,
                        "checkpoint_family": family, "training_seed": training_seed,
                        "generation_seed": generation_seed, "prompt": prompt,
                        "evaluation_log": str(log), "source": "new",
                    })
    return tasks, sorted(index, key=lambda row: (
        row["checkpoint_family"], row["training_seed"], row["generation_seed"], row["prompt"]
    ))


def run_scheduler(args, tasks):
    root = Path(args.run_root).resolve()
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    completed = {name for name, task in tasks.items() if task.complete()}
    running = {}
    print(f"Sparse-interface transfer: {len(completed)}/{len(tasks)} tasks complete", flush=True)
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
                failed = task.log.with_name(f"{task.log.name}.failed_{task.attempts}_{time.strftime('%Y%m%dT%H%M%S')}")
                if task.log.exists():
                    task.log.replace(failed)
                if args.max_retries and task.attempts >= args.max_retries:
                    raise RuntimeError(f"Task exhausted retries: {task.name}; see {failed}")
                task.next_ready = time.time() + args.retry_delay_seconds
                print(f"RETRY in {args.retry_delay_seconds}s: {task.name}; see {failed}", flush=True)
        ready = [
            task for task in tasks.values()
            if task.name not in completed and task.process is None and task.next_ready <= now
            and all(dependency in completed for dependency in task.dependencies)
        ]
        ready.sort(key=lambda task: ({"eval": 0, "generate": 1}[task.kind], task.attempts, task.name))
        for gpu in [item for item in gpus if item not in running]:
            if not ready:
                break
            task = ready.pop(0)
            launch(task, gpu, args)
            running[gpu] = task
        write_state(root, tasks, completed, running)
        time.sleep(5)


def main():
    args = parse_args()
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
    required = [args.data_root, args.base_model, args.prototype, args.dcs, args.prompt_bank]
    required += [str(model_path(args, family, seed)) for family in FAMILIES for seed in args.training_seeds]
    for path in required:
        if not Path(path).exists():
            raise FileNotFoundError(path)
    canonical_hash = sha256(args.prompt_bank)
    mismatched = [path for path in args.bank_copy if sha256(path) != canonical_hash]
    if mismatched:
        raise RuntimeError(f"Sparse m4 banks differ from canonical bank: {mismatched}")
    root = Path(args.run_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    reuse = load_reuse(args.reuse_index)
    tasks, index = build_tasks(args, reuse)
    manifest = {
        "format_version": 1, "experiment": "sparse_interface_transfer",
        "data_root": str(Path(args.data_root).resolve()), "base_model": str(Path(args.base_model).resolve()),
        "prototype": str(Path(args.prototype).resolve()), "prototype_sha256": sha256(args.prototype),
        "dcs": str(Path(args.dcs).resolve()), "dcs_sha256": sha256(args.dcs),
        "prompt_bank": str(Path(args.prompt_bank).resolve()), "prompt_bank_sha256": canonical_hash,
        "training_seeds": sorted(set(args.training_seeds)), "generation_seeds": sorted(set(args.generation_seeds)),
        "ipc": args.ipc, "strength": args.strength, "prompts": list(PROMPTS),
        "classifier_repeats": args.classifier_repeats,
        "classifier_seed": args.classifier_seed,
        "guidance_scale": args.guidance_scale,
        "num_inference_steps": args.num_inference_steps,
        "prompt_bank_copies": [
            {"path": str(Path(path).resolve()), "sha256": sha256(path)} for path in args.bank_copy
        ],
        "checkpoint_models": {
            f"{family}_seed{seed}": checkpoint_identity(model_path(args, family, seed))
            for family in FAMILIES for seed in args.training_seeds
        },
        "reuse_indexes": [
            {"path": str(Path(path).resolve()), "sha256": sha256(path)} for path in args.reuse_index
        ],
    }
    manifest_path = root / "run_manifest.json"
    if manifest_path.is_file() and json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
        raise RuntimeError(f"Resume configuration differs from {manifest_path}")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    index_path = root / "evaluation_index.json"
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    run_scheduler(args, tasks)
    subprocess.run([
        sys.executable, str(HERE / "summarize_sparse_interface_transfer.py"),
        "--evaluation-index", str(index_path), "--output-dir", str(root / "summary"),
    ], cwd=REPO_ROOT, check=True)
    (root / "COMPLETE").write_text(time.strftime("%F %T\n"), encoding="utf-8")


if __name__ == "__main__":
    main()
