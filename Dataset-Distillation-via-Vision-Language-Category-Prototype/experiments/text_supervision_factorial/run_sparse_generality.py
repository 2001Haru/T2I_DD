#!/usr/bin/env python3
"""Run fixed-bank sparse-caption supervision across datasets and training seeds."""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
RUNNER = HERE / "run_sparse_prompt_search.py"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nette-data-root", required=True)
    parser.add_argument("--nette-caption-file", required=True)
    parser.add_argument("--woof-data-root", required=True)
    parser.add_argument("--woof-caption-file", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--nette-prototype", required=True)
    parser.add_argument("--nette-dcs", required=True)
    parser.add_argument("--woof-prototype", required=True)
    parser.add_argument("--woof-dcs", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--nette-training-seeds", type=int, nargs="+", default=(1, 2))
    parser.add_argument("--woof-training-seeds", type=int, nargs="+", default=(0, 1))
    parser.add_argument("--generation-seeds", type=int, nargs="+", default=(0, 1))
    parser.add_argument("--ipc", type=int, default=50)
    parser.add_argument("--strength", type=float, default=0.8)
    parser.add_argument("--classifier-repeats", type=int, default=2)
    parser.add_argument("--classifier-seed", type=int, default=0)
    parser.add_argument("--diffusers-src", default="")
    parser.add_argument("--retry-delay-seconds", type=int, default=120)
    parser.add_argument("--old-nette-sparse-index", default="")
    parser.add_argument("--control-index", action="append", default=[])
    return parser.parse_args()


def child_command(args, spec, training_seed, gpu):
    data_root = getattr(args, f"{spec}_data_root")
    caption_file = getattr(args, f"{spec}_caption_file")
    prototype = getattr(args, f"{spec}_prototype")
    dcs = getattr(args, f"{spec}_dcs")
    output = Path(args.run_root).resolve() / spec / f"train_seed_{training_seed}"
    command = [
        sys.executable, str(RUNNER), "--data-root", data_root,
        "--caption-file", caption_file, "--spec", spec,
        "--base-model", args.base_model, "--prototype", prototype, "--dcs", dcs,
        "--run-root", str(output), "--gpus", gpu, "--budgets", "4", "--bank-seeds", "0",
        "--prompts", "label",
        "--training-seed", str(training_seed), "--generation-seeds",
        *[str(value) for value in args.generation_seeds],
        "--ipc", str(args.ipc), "--strength", str(args.strength),
        "--classifier-repeats", str(args.classifier_repeats),
        "--classifier-seed", str(args.classifier_seed),
        "--retry-delay-seconds", str(args.retry_delay_seconds),
    ]
    if args.diffusers_src:
        command.extend(("--diffusers-src", args.diffusers_src))
    return command, output


def child_complete(output):
    return (output / "COMPLETE").is_file()


def write_state(root, pending, running, completed):
    payload = {
        "updated_at": time.strftime("%F %T"),
        "pending": [item[0] for item in pending],
        "running": {gpu: item[0] for gpu, item in running.items()},
        "completed": completed,
    }
    (root / "scheduler_state.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
    gpus = [value.strip() for value in args.gpus.split(",") if value.strip()]
    if not gpus:
        raise ValueError("At least one GPU is required")
    root = Path(args.run_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format_version": 1,
        "experiment": "fixed_m4_sparse_caption_generality",
        "nette_training_seeds": sorted(set(args.nette_training_seeds)),
        "woof_training_seeds": sorted(set(args.woof_training_seeds)),
        "generation_seeds": list(args.generation_seeds),
        "ipc": args.ipc, "strength": args.strength, "bank_seed": 0, "budget": 4,
        "inference_prompts": ["label"],
        "classifier_repeats": args.classifier_repeats,
        "paths": {name: str(Path(getattr(args, name)).resolve()) for name in (
            "nette_data_root", "nette_caption_file", "woof_data_root", "woof_caption_file",
            "base_model", "nette_prototype", "nette_dcs", "woof_prototype", "woof_dcs")},
    }
    manifest_path = root / "run_manifest.json"
    if manifest_path.is_file() and json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
        raise RuntimeError(f"Resume configuration differs from {manifest_path}")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    jobs = []
    for spec, seeds in (("nette", args.nette_training_seeds), ("woof", args.woof_training_seeds)):
        for seed in sorted(set(seeds)):
            command, output = child_command(args, spec, seed, "")
            jobs.append((f"{spec}_train_seed_{seed}", command, output))
    completed = [name for name, _, output in jobs if child_complete(output)]
    pending = [item for item in jobs if item[0] not in completed]
    running = {}
    logs = root / "scheduler_logs"
    logs.mkdir(exist_ok=True)
    print(f"Sparse generality: {len(completed)}/{len(jobs)} child pipelines complete", flush=True)
    while pending or running:
        for gpu, (name, process, handle, output, command) in list(running.items()):
            code = process.poll()
            if code is None:
                continue
            handle.write(f"\n[{time.strftime('%F %T')}] exit code {code}\n")
            handle.close()
            del running[gpu]
            if code == 0 and child_complete(output):
                completed.append(name)
                print(f"DONE GPU {gpu}: {name}", flush=True)
            else:
                print(f"REQUEUE GPU {gpu}: {name} after {args.retry_delay_seconds}s", flush=True)
                time.sleep(args.retry_delay_seconds)
                pending.insert(0, (name, command, output))
        for gpu in [value for value in gpus if value not in running]:
            if not pending:
                break
            name, command, output = pending.pop(0)
            command = list(command)
            command[command.index("--gpus") + 1] = gpu
            log = logs / f"{name}.log"
            handle = log.open("a", encoding="utf-8", buffering=1)
            handle.write(f"\n[{time.strftime('%F %T')}] GPU {gpu}\n{' '.join(command)}\n")
            process = subprocess.Popen(command, cwd=HERE.parents[1], stdout=handle, stderr=subprocess.STDOUT)
            running[gpu] = (name, process, handle, output, command)
            print(f"LAUNCH GPU {gpu}: {name}", flush=True)
        write_state(root, pending, running, sorted(completed))
        time.sleep(5)
    indexes = []
    for _, _, output in jobs:
        index = output / "evaluation_index.json"
        if not index.is_file():
            raise FileNotFoundError(index)
        indexes.extend(json.loads(index.read_text(encoding="utf-8")))
    (root / "evaluation_index.json").write_text(json.dumps(indexes, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.control_index:
        command = [
            sys.executable, str(HERE / "summarize_sparse_generality.py"),
            "--sparse-index", str(root / "evaluation_index.json"),
            "--output-dir", str(root / "summary"),
        ]
        if args.old_nette_sparse_index:
            command.extend(("--old-nette-sparse-index", args.old_nette_sparse_index))
        for index in args.control_index:
            command.extend(("--control-index", index))
        subprocess.run(command, cwd=HERE.parents[1], check=True)
    (root / "COMPLETE").write_text(time.strftime("%F %T\n"), encoding="utf-8")


if __name__ == "__main__":
    main()
