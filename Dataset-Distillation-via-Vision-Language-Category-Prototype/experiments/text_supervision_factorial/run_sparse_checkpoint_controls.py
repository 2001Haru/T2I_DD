#!/usr/bin/env python3
"""Evaluate existing checkpoints as controls for the sparse prompt-bank search."""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from run_sparse_prompt_search import (
    EVAL_DIR,
    REPO_ROOT,
    Task,
    complete_eval,
    eval_command,
    launch,
    write_state,
)


HERE = Path(__file__).resolve().parent
AUDIT_SUPERVISIONS = ("empty_ft", "label_ft", "unpaired_ft", "matched_ft")
EVALUATED_SUPERVISIONS = ("empty_ft", "label_ft", "unpaired_ft")
SUMMARY_SUPERVISION = {
    "empty_ft": "empty",
    "label_ft": "label",
    "unpaired_ft": "unpaired",
    "matched_ft": "matched",
}
CONFIG_FILES = (
    "model_index.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "tokenizer/tokenizer_config.json",
    "unet/config.json",
    "vae/config.json",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--base-run-root", required=True)
    parser.add_argument("--causal-run-root", required=True)
    parser.add_argument("--sparse-run-root", required=True)
    parser.add_argument("--prototype", required=True)
    parser.add_argument("--dcs", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--training-seeds", type=int, nargs="+", default=(0, 1))
    parser.add_argument("--generation-seeds", type=int, nargs="+", default=(0, 1))
    parser.add_argument("--ipc", type=int, default=50)
    parser.add_argument("--strength", type=float, default=0.8)
    parser.add_argument("--classifier-repeats", type=int, default=2)
    parser.add_argument("--classifier-seed", type=int, default=0)
    parser.add_argument("--retry-delay-seconds", type=int, default=120)
    parser.add_argument("--max-retries", type=int, default=0)
    parser.add_argument("--diffusers-src", default="")
    return parser.parse_args()


def checkpoint_path(args, supervision, training_seed):
    if training_seed == 0 and supervision in {"label_ft", "unpaired_ft", "matched_ft"}:
        return Path(args.base_run_root).resolve() / "models" / supervision
    return (
        Path(args.causal_run_root).resolve() / "models" /
        f"train_seed_{training_seed}" / supervision
    )


def scrub_config(value):
    if isinstance(value, dict):
        return {
            key: scrub_config(item)
            for key, item in value.items()
            if key not in {"_name_or_path", "_diffusers_version", "transformers_version"}
        }
    if isinstance(value, list):
        return [scrub_config(item) for item in value]
    return value


def config_digest(path):
    value = scrub_config(json.loads(path.read_text(encoding="utf-8")))
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def audit_checkpoints(args, output):
    records = []
    errors = []
    warnings = []
    reference = {}
    for training_seed in sorted(set(args.training_seeds)):
        for supervision in AUDIT_SUPERVISIONS:
            model = checkpoint_path(args, supervision, training_seed)
            summary_path = model / "training_summary.json"
            if not (model / "model_index.json").is_file() or not summary_path.is_file():
                errors.append(f"Missing complete checkpoint: {model}")
                continue
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            expected_summary = SUMMARY_SUPERVISION[supervision]
            if summary.get("supervision") != expected_summary:
                errors.append(
                    f"{model}: supervision={summary.get('supervision')!r}, expected={expected_summary!r}"
                )
            if int(summary.get("epochs", -1)) != 8:
                errors.append(f"{model}: epochs={summary.get('epochs')!r}, expected=8")
            recorded_seed = summary.get("seed")
            if recorded_seed is None:
                warnings.append(
                    f"{model}: legacy summary does not record seed; "
                    f"using path-selected training seed {training_seed}"
                )
            elif int(recorded_seed) != training_seed:
                errors.append(f"{model}: seed={recorded_seed!r}, expected={training_seed}")
            if summary.get("sparse_bank") not in (None, ""):
                errors.append(f"{model}: unexpectedly records sparse_bank={summary['sparse_bank']!r}")
            digests = {}
            for relative in CONFIG_FILES:
                config = model / relative
                if not config.is_file():
                    errors.append(f"{model}: missing component config {relative}")
                    continue
                digests[relative] = config_digest(config)
                if relative in reference and reference[relative] != digests[relative]:
                    errors.append(f"{model}: component architecture/config differs at {relative}")
                reference.setdefault(relative, digests[relative])
            records.append({
                "training_seed": training_seed,
                "supervision": supervision,
                "model": str(model),
                "training_summary": summary,
                "normalized_config_sha256": digests,
            })
    report = {
        "format_version": 1,
        "status": "pass" if not errors else "fail",
        "checks": records,
        "errors": errors,
        "warnings": warnings,
        "expected_common_protocol": {
            "epochs": 8,
            "resolution": 512,
            "learning_rate": 1e-5,
            "lr_scheduler": "constant",
            "lr_warmup_steps": 0,
            "effective_batch_size": 32,
            "random_flip": True,
            "gradient_checkpointing": True,
            "use_ema": True,
        },
        "boundary": (
            "Legacy training summaries may omit seed and do not store the full optimizer command. A missing "
            "seed is recorded as a warning and the explicitly selected checkpoint path defines the training "
            "seed; a present but mismatched seed remains fatal. Full-protocol equality is reconstructed from "
            "the repository launch scripts, while this audit directly verifies available summary fields and "
            "normalized pipeline component configs."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if errors:
        raise RuntimeError("Checkpoint protocol audit failed:\n- " + "\n- ".join(errors))
    return records


def complete_generation(root, condition, expected):
    def check():
        complete = root / condition / "complete.json"
        if not complete.is_file():
            return False
        return int(json.loads(complete.read_text(encoding="utf-8"))["images"]) == expected
    return check


def generation_command(args, model, output, supervision, generation_seed):
    return [
        sys.executable, str(HERE / "generate_factorial.py"),
        "--prototype", args.prototype, "--dcs", args.dcs,
        "--base-model", args.base_model, "--model", f"{supervision}={model}",
        "--supervisions", supervision, "--prompts", "label",
        "--output-root", str(output), "--generation-seeds", str(generation_seed),
        "--ipc", str(args.ipc), "--strength", str(args.strength),
        "--guidance-scale", "10", "--num-inference-steps", "50",
        "--shuffle-shift", "1", "--size", "256", "--resume",
    ]


def build_tasks(args):
    root = Path(args.run_root).resolve()
    tasks = {}
    index = []
    for training_seed in sorted(set(args.training_seeds)):
        for supervision in EVALUATED_SUPERVISIONS:
            model = checkpoint_path(args, supervision, training_seed)
            for generation_seed in sorted(set(args.generation_seeds)):
                token = f"{supervision}_t{training_seed}_g{generation_seed}"
                output = root / "synthetic" / supervision / f"train_seed_{training_seed}"
                seed_root = output / f"seed_{generation_seed}"
                condition = f"{supervision}_label"
                gen_name = f"gen_{token}"
                tasks[gen_name] = Task(
                    gen_name, 1, "generate",
                    generation_command(args, model, output, supervision, generation_seed),
                    REPO_ROOT, root / "scheduler_logs" / f"{gen_name}.log",
                    complete_generation(seed_root, condition, 10 * args.ipc),
                )
                eval_name = f"eval_{token}"
                log = (
                    root / "evaluation" / supervision / f"train_seed_{training_seed}" /
                    f"seed_{generation_seed}" / f"{condition}.log"
                )
                tasks[eval_name] = Task(
                    eval_name, 1, "eval",
                    eval_command(args, seed_root / condition, eval_name),
                    EVAL_DIR, log, complete_eval(log), dependencies=(gen_name,),
                )
                index.append({
                    "method": "checkpoint_control",
                    "supervision": supervision,
                    "training_seed": training_seed,
                    "generation_seed": generation_seed,
                    "prompt": "label",
                    "ipc": args.ipc,
                    "strength": args.strength,
                    "evaluation_log": str(log),
                })
    return tasks, index


def run_scheduler(args, root, tasks):
    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if not gpus:
        raise ValueError("At least one GPU is required")
    completed = {name for name, task in tasks.items() if task.complete()}
    running = {}
    print(f"Checkpoint controls: {len(completed)}/{len(tasks)} tasks complete", flush=True)
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
                failed = task.log.with_name(
                    f"{task.log.name}.failed_{task.attempts}_{time.strftime('%Y%m%dT%H%M%S')}"
                )
                if task.log.exists():
                    task.log.replace(failed)
                if args.max_retries and task.attempts >= args.max_retries:
                    raise RuntimeError(f"Task exhausted retries: {task.name}; see {failed}")
                task.next_ready = time.time() + args.retry_delay_seconds
                print(f"RETRY in {args.retry_delay_seconds}s: {task.name}; see {failed}", flush=True)
        free = [gpu for gpu in gpus if gpu not in running]
        ready = [
            task for task in tasks.values()
            if task.name not in completed and task.process is None and task.next_ready <= now
            and all(dependency in completed for dependency in task.dependencies)
        ]
        ready.sort(key=lambda task: (task.stage, {"eval": 0, "generate": 1}[task.kind], task.attempts, task.name))
        for gpu in free:
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
    for path in (
        args.data_root, args.base_model, args.base_run_root, args.causal_run_root,
        args.sparse_run_root, args.prototype, args.dcs,
    ):
        if not Path(path).exists():
            raise FileNotFoundError(path)
    root = Path(args.run_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    audit_checkpoints(args, root / "checkpoint_protocol_audit.json")
    tasks, index = build_tasks(args)
    manifest = {
        "format_version": 1,
        "experiment": "sparse_checkpoint_controls",
        "data_root": str(Path(args.data_root).resolve()),
        "base_model": str(Path(args.base_model).resolve()),
        "base_run_root": str(Path(args.base_run_root).resolve()),
        "causal_run_root": str(Path(args.causal_run_root).resolve()),
        "sparse_run_root": str(Path(args.sparse_run_root).resolve()),
        "prototype": str(Path(args.prototype).resolve()),
        "dcs": str(Path(args.dcs).resolve()),
        "audited_supervisions": list(AUDIT_SUPERVISIONS),
        "evaluated_supervisions": list(EVALUATED_SUPERVISIONS),
        "training_seeds": sorted(set(args.training_seeds)),
        "generation_seeds": sorted(set(args.generation_seeds)),
        "ipc": args.ipc,
        "strength": args.strength,
        "classifier_repeats": args.classifier_repeats,
        "gpus": [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()],
        "fine_tuning_tasks": 0,
    }
    manifest_path = root / "run_manifest.json"
    if manifest_path.is_file() and json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
        raise RuntimeError(f"Resume configuration differs from {manifest_path}")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    index_path = root / "evaluation_index.json"
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    run_scheduler(args, root, tasks)
    subprocess.run([
        sys.executable, str(HERE / "summarize_sparse_checkpoint_controls.py"),
        "--control-index", str(index_path),
        "--sparse-index", str(Path(args.sparse_run_root).resolve() / "evaluation_index.json"),
        "--output-dir", str(root / "summary"),
    ], cwd=REPO_ROOT, check=True)
    (root / "COMPLETE").write_text(time.strftime("%F %T\n"), encoding="utf-8")


if __name__ == "__main__":
    main()
