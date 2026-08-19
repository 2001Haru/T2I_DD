#!/usr/bin/env python3
"""Complete Label inference cells for the G=6, C=3 T77 matrix."""

import argparse
import csv
import hashlib
import json
import signal
import subprocess
import sys
import time
from pathlib import Path

from run_low_variance_t77_matrix import (
    EVAL_DIR,
    REPO_ROOT,
    Task,
    checkpoint_complete,
    evaluation_complete,
    evaluation_save_dir,
    linear_paired_bootstrap,
    parse_metrics,
)
from run_sparse_prompt_search import launch, write_state


HERE = Path(__file__).resolve().parent
CELLS = (
    ("matched_ft", "matched_ft"),
    ("unpaired_ft", "unpaired_ft"),
    ("bank_m4", "sparse_ft"),
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--prototype", required=True)
    parser.add_argument("--dcs", required=True)
    parser.add_argument("--matched-model", required=True)
    parser.add_argument("--unpaired-model", required=True)
    parser.add_argument("--bank-m4-model", required=True)
    parser.add_argument("--base-run-root", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--generation-seeds", type=int, nargs="+", default=tuple(range(6)))
    parser.add_argument("--classifier-repeats", type=int, default=3)
    parser.add_argument("--classifier-seed", type=int, default=0)
    parser.add_argument("--tail-k", type=int, default=10)
    parser.add_argument("--ipc", type=int, default=50)
    parser.add_argument("--strength", type=float, default=0.8)
    parser.add_argument("--retry-delay-seconds", type=int, default=120)
    parser.add_argument("--max-retries", type=int, default=0)
    parser.add_argument("--diffusers-src", default="")
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_map(args):
    return {
        "matched_ft": Path(args.matched_model).resolve(),
        "unpaired_ft": Path(args.unpaired_model).resolve(),
        "bank_m4": Path(args.bank_m4_model).resolve(),
    }


def audit(args, root):
    errors = []
    base_root = Path(args.base_run_root).resolve()
    base_manifest_path = base_root / "run_manifest.json"
    base_index_path = base_root / "evaluation_index.json"
    if not base_manifest_path.is_file() or not base_index_path.is_file():
        raise FileNotFoundError(f"Missing base G6/C3 manifest/index under {base_root}")
    base_manifest = json.loads(base_manifest_path.read_text(encoding="utf-8"))
    base_index = json.loads(base_index_path.read_text(encoding="utf-8"))
    if base_manifest.get("generation_seeds") != list(args.generation_seeds):
        errors.append("Generation seeds differ from the base G6/C3 run")
    if base_manifest.get("ipc") != args.ipc or base_manifest.get("strength") != args.strength:
        errors.append("IPC or strength differs from the base G6/C3 run")
    declared_models = {
        row["checkpoint"]: str(Path(row["model"]).resolve())
        for row in base_manifest.get("cells", [])
    }
    rows = []
    for checkpoint, model in model_map(args).items():
        row = {
            "checkpoint": checkpoint, "model": str(model),
            "complete": checkpoint_complete(model),
            "matches_base_manifest": declared_models.get(checkpoint) == str(model),
        }
        if not row["complete"]:
            errors.append(f"Incomplete checkpoint: {model}")
        if not row["matches_base_manifest"]:
            errors.append(
                f"Checkpoint path differs from base matrix for {checkpoint}: "
                f"{declared_models.get(checkpoint)} vs {model}"
            )
        config_path = model / "training_config.json"
        if config_path.is_file():
            config = json.loads(config_path.read_text(encoding="utf-8"))
            row.update({
                "training_seed": config.get("seed"),
                "supervision": config.get("supervision"),
                "train_batch_size": config.get("train_batch_size"),
                "gradient_accumulation_steps": config.get("gradient_accumulation_steps"),
            })
            if int(config.get("seed", -1)) != 0:
                errors.append(f"Training seed is not 0 for {checkpoint}")
        else:
            errors.append(f"Missing training_config.json: {model}")
        rows.append(row)

    reference = next(
        (row for row in base_index if row["checkpoint"] == "matched_ft"), None
    )
    if reference is None:
        errors.append("Base matrix has no matched_ft reference cell")
    else:
        generation_manifest_path = Path(reference["synthetic_dir"]) / "manifest.json"
        if not generation_manifest_path.is_file():
            errors.append(f"Missing base generation manifest: {generation_manifest_path}")
        else:
            generation_manifest = json.loads(
                generation_manifest_path.read_text(encoding="utf-8")
            )
            checks = {
                "prototype_sha256": sha256(args.prototype),
                "dcs_sha256": sha256(args.dcs),
                "ipc": args.ipc,
                "strength": args.strength,
                "guidance_scale": 10.0,
                "num_inference_steps": 50,
            }
            for key, expected in checks.items():
                if generation_manifest.get(key) != expected:
                    errors.append(
                        f"Base generation protocol differs for {key}: "
                        f"{generation_manifest.get(key)} vs {expected}"
                    )
    payload = {
        "format_version": 1, "passed": not errors, "errors": errors,
        "base_run_root": str(base_root), "checkpoints": rows,
        "generation_seeds": list(args.generation_seeds),
        "classifier_repeats": args.classifier_repeats,
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "preflight_audit.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if errors:
        raise RuntimeError("Label-completion audit failed:\n- " + "\n- ".join(errors))
    return base_index


def generation_complete(seed_root, supervision, expected):
    def check():
        condition = f"{supervision}_label"
        complete = seed_root / condition / "complete.json"
        records = seed_root / condition / "prompt_records.json"
        return (
            complete.is_file() and records.is_file()
            and int(json.loads(complete.read_text(encoding="utf-8"))["images"]) == expected
        )
    return check


def generation_command(args, checkpoint, supervision, model, output, generation_seed):
    del checkpoint
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


def evaluation_command(args, synthetic, tag):
    return [
        sys.executable, "train.py", "-d", "imagenet", "--imagenet_dir",
        str(synthetic), str(Path(args.data_root).resolve()), "-n", "resnet_ap",
        "--nclass", "10", "--norm_type", "instance", "--ipc", str(args.ipc),
        "--tag", tag, "--slct_type", "random", "--repeat", str(args.classifier_repeats),
        "--spec", "nette", "--seed", str(args.classifier_seed),
        "--tail_k", str(args.tail_k),
    ]


def build_tasks(args):
    root = Path(args.run_root).resolve()
    models = model_map(args)
    tasks, index = {}, []
    for checkpoint, supervision in CELLS:
        model = models[checkpoint]
        for generation_seed in args.generation_seeds:
            output = root / "synthetic" / checkpoint
            seed_root = output / f"seed_{generation_seed}"
            gen_name = f"gen_{checkpoint}_g{generation_seed}_label"
            tasks[gen_name] = Task(
                gen_name, 1, "generate",
                generation_command(
                    args, checkpoint, supervision, model, output, generation_seed
                ),
                REPO_ROOT, root / "scheduler_logs" / f"{gen_name}.log",
                generation_complete(seed_root, supervision, 10 * args.ipc),
            )
            condition = f"{supervision}_label"
            eval_name = f"eval_{checkpoint}_g{generation_seed}_label"
            log = root / "evaluation" / checkpoint / f"seed_{generation_seed}" / "label.log"
            tag = f"lowvar_t77_label_completion_{checkpoint}_g{generation_seed}"
            tasks[eval_name] = Task(
                eval_name, 2, "eval",
                evaluation_command(args, seed_root / condition, tag),
                EVAL_DIR, log, evaluation_complete(log), dependencies=(gen_name,),
            )
            index.append({
                "checkpoint": checkpoint, "training_seed": 0,
                "generation_seed": int(generation_seed), "prompt": "label",
                "condition": condition, "synthetic_dir": str(seed_root / condition),
                "evaluation_log": str(log),
                "evaluation_save_dir": str(evaluation_save_dir(tag, args.ipc)),
                "ipc": args.ipc, "strength": args.strength,
                "classifier_repeats": args.classifier_repeats, "tail_k": args.tail_k,
            })
    return tasks, index


def summarize(base_index, completion_index, output):
    combined = base_index + completion_index
    output.mkdir(parents=True, exist_ok=True)
    performance = []
    for metric in ("best", "tail"):
        for checkpoint in ("matched_ft", "unpaired_ft", "bank_m4"):
            selected = [
                row for row in completion_index if row["checkpoint"] == checkpoint
            ]
            values = [
                value for row in selected for value in parse_metrics(row["evaluation_log"])[metric]
            ]
            mean, lower, upper = linear_paired_bootstrap(
                combined, ((1.0, (checkpoint, "label")),), metric,
                seed=20260819 + len(performance),
            )
            performance.append({
                "metric": metric, "checkpoint": checkpoint, "prompt": "label",
                "mean_accuracy": mean,
                "bootstrap_ci95_lower": lower, "bootstrap_ci95_upper": upper,
                "generation_cells": len(selected), "classifier_observations": len(values),
            })
    specifications = [
        (
            "matched_correct_minus_label",
            ((1.0, ("matched_ft", "correct_t77")), (-1.0, ("matched_ft", "label"))),
        ),
        (
            "unpaired_correct_minus_label",
            ((1.0, ("unpaired_ft", "correct_t77")), (-1.0, ("unpaired_ft", "label"))),
        ),
        (
            "sparse_m4_bank_minus_label",
            ((1.0, ("bank_m4", "bank_t77")), (-1.0, ("bank_m4", "label"))),
        ),
        (
            "matched_minus_unpaired_label",
            ((1.0, ("matched_ft", "label")), (-1.0, ("unpaired_ft", "label"))),
        ),
        (
            "checkpoint_x_correct_vs_label_interaction",
            (
                (1.0, ("matched_ft", "correct_t77")),
                (-1.0, ("matched_ft", "label")),
                (-1.0, ("unpaired_ft", "correct_t77")),
                (1.0, ("unpaired_ft", "label")),
            ),
        ),
    ]
    contrast_rows = []
    for metric_index, metric in enumerate(("best", "tail")):
        for contrast_index, (name, terms) in enumerate(specifications):
            mean, lower, upper = linear_paired_bootstrap(
                combined, terms, metric,
                seed=20260819 + metric_index * 100 + contrast_index,
            )
            contrast_rows.append({
                "metric": metric, "contrast": name, "mean_difference": mean,
                "bootstrap_ci95_lower": lower, "bootstrap_ci95_upper": upper,
                "generation_seed_levels": 6, "paired_classifier_observations": 18,
                "bootstrap_order": "generation seed -> shared classifier repeat",
            })
    for filename, rows in (("completion_performance.csv", performance), ("core_2x2_contrasts.csv", contrast_rows)):
        with (output / filename).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (output / "summary.json").write_text(
        json.dumps({"performance": performance, "contrasts": contrast_rows}, indent=2) + "\n",
        encoding="utf-8",
    )


def main():
    args = parse_args()
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
    if tuple(args.generation_seeds) != tuple(range(6)):
        raise ValueError("Completion requires generation seeds 0-5")
    if args.classifier_repeats != 3:
        raise ValueError("Completion requires classifier repeats 3")
    gpus = [value.strip() for value in args.gpus.split(",") if value.strip()]
    if len(gpus) != 2:
        raise ValueError("Completion is configured for exactly two GPUs")
    root = Path(args.run_root).resolve()
    base_index = audit(args, root)
    if args.audit_only:
        print(f"Audit passed: {root / 'preflight_audit.json'}")
        return
    tasks, index = build_tasks(args)
    manifest = {
        "format_version": 1, "experiment": "low_variance_t77_label_completion",
        "base_run_root": str(Path(args.base_run_root).resolve()),
        "gpus": gpus, "generation_seeds": list(args.generation_seeds),
        "classifier_repeats": args.classifier_repeats, "tail_k": args.tail_k,
        "cells": [{"checkpoint": checkpoint, "prompt": "label"} for checkpoint, _ in CELLS],
        "fresh_generation": True,
    }
    manifest_path = root / "run_manifest.json"
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        previous.pop("gpus", None)
        current = dict(manifest)
        current.pop("gpus", None)
        if previous != current:
            raise RuntimeError(f"Resume configuration differs from {manifest_path}")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (root / "evaluation_index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
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
                    raise RuntimeError(f"Task exhausted retries: {task.name}")
                task.next_ready = time.time() + args.retry_delay_seconds
        ready = [
            task for task in tasks.values()
            if task.name not in completed and task.process is None and task.next_ready <= now
            and all(dependency in completed for dependency in task.dependencies)
        ]
        ready.sort(key=lambda task: (
            task.stage, {"eval": 0, "generate": 1}[task.kind], task.attempts, task.name
        ))
        for gpu in [value for value in gpus if value not in running]:
            if not ready:
                break
            task = ready.pop(0)
            launch(task, gpu, args)
            running[gpu] = task
        write_state(root, tasks, completed, running)
        time.sleep(5)
    summarize(base_index, index, root / "summary")
    (root / "COMPLETE").write_text(time.strftime("%F %T\n"), encoding="utf-8")
    print(f"Label completion complete: {root}")


if __name__ == "__main__":
    main()
