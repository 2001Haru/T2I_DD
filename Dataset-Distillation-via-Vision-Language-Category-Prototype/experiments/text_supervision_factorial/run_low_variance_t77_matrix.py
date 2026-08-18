#!/usr/bin/env python3
"""Run the fresh G=6, C=3 T77 low-variance evaluation matrix."""

import argparse
import ast
import csv
import hashlib
import json
import random
import re
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

from run_sparse_prompt_search import (
    EVAL_DIR,
    REPO_ROOT,
    Task,
    launch,
    write_state,
)


HERE = Path(__file__).resolve().parent
BEST = re.compile(r"Best, last acc:----(\[[^\]]+\])")
TAIL = re.compile(r"Tail-(\d+) val acc:----(\[[^\]]+\])")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--prototype", required=True)
    parser.add_argument("--dcs", required=True)
    parser.add_argument("--matched-model", required=True)
    parser.add_argument("--unpaired-model", required=True)
    parser.add_argument("--bank-m4-model", required=True)
    parser.add_argument("--bank-m4-json", required=True)
    parser.add_argument("--bank-m64-model", required=True)
    parser.add_argument("--bank-m64-json", required=True)
    parser.add_argument("--label-model", required=True)
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


def checkpoint_complete(path):
    path = Path(path)
    if not (path / "model_index.json").is_file():
        return False
    for component in ("unet", "vae", "text_encoder", "tokenizer", "scheduler"):
        if not (path / component).is_dir():
            return False
    unet_weights = list((path / "unet").glob("*.safetensors")) + list(
        (path / "unet").glob("*.bin")
    )
    return bool(unet_weights)


def checkpoint_specs(args):
    return [
        {
            "key": "matched_ft", "supervision": "matched_ft",
            "training_supervision": "matched", "model": Path(args.matched_model).resolve(),
            "prompts": ("correct_t77", "shuffled_t77"), "prompt_bank": None,
        },
        {
            "key": "unpaired_ft", "supervision": "unpaired_ft",
            "training_supervision": "unpaired", "model": Path(args.unpaired_model).resolve(),
            "prompts": ("correct_t77", "shuffled_t77"), "prompt_bank": None,
        },
        {
            "key": "bank_m4", "supervision": "sparse_ft",
            "training_supervision": "sparse_unpaired",
            "model": Path(args.bank_m4_model).resolve(),
            "prompts": ("bank_t77",), "prompt_bank": Path(args.bank_m4_json).resolve(),
            "budget": 4,
        },
        {
            "key": "bank_m64", "supervision": "sparse_ft",
            "training_supervision": "sparse_unpaired",
            "model": Path(args.bank_m64_model).resolve(),
            "prompts": ("bank_t77",), "prompt_bank": Path(args.bank_m64_json).resolve(),
            "budget": 64,
        },
        {
            "key": "label_ft", "supervision": "label_ft",
            "training_supervision": "label", "model": Path(args.label_model).resolve(),
            "prompts": ("label", "correct_t77"), "prompt_bank": None,
        },
    ]


def audit_models(args, root):
    errors, rows = [], []
    base_model = str(Path(args.base_model).resolve())
    for spec in checkpoint_specs(args):
        model = spec["model"]
        config_path = model / "training_config.json"
        summary_path = model / "training_summary.json"
        row = {
            "checkpoint": spec["key"], "model": str(model),
            "model_complete": checkpoint_complete(model),
            "expected_training_supervision": spec["training_supervision"],
            "expected_training_seed": 0,
            "expected_text_interface": "CLIP T77 truncation and max-length padding",
        }
        if not row["model_complete"]:
            errors.append(f"Incomplete Diffusers checkpoint: {model}")
        if not config_path.is_file() or not summary_path.is_file():
            errors.append(f"Missing training audit files: {model}")
        else:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            row.update({
                "actual_training_supervision": config.get("supervision"),
                "actual_training_seed": config.get("seed"),
                "train_batch_size": config.get("train_batch_size"),
                "gradient_accumulation_steps": config.get("gradient_accumulation_steps"),
                "effective_batch_size": config.get("effective_batch_size"),
                "epochs": config.get("num_train_epochs"),
                "learning_rate": config.get("learning_rate"),
                "pretrained_model": config.get("pretrained_model"),
                "training_complete": summary.get("complete"),
                "training_global_steps": summary.get("global_steps"),
                "model_index_sha256": sha256(model / "model_index.json"),
            })
            if config.get("supervision") != spec["training_supervision"]:
                errors.append(f"Wrong supervision for {spec['key']}: {config.get('supervision')}")
            if int(config.get("seed", -1)) != 0:
                errors.append(f"Wrong training seed for {spec['key']}: {config.get('seed')}")
            if str(Path(config.get("pretrained_model", "")).resolve()) != base_model:
                errors.append(f"Base model differs for {spec['key']}: {config.get('pretrained_model')}")
            if summary.get("complete") is not True:
                errors.append(f"Training is not marked complete: {model}")
        bank = spec.get("prompt_bank")
        if bank is not None:
            if not bank.is_file():
                errors.append(f"Missing prompt bank: {bank}")
            else:
                payload = json.loads(bank.read_text(encoding="utf-8"))
                counts = {key: len(value) for key, value in payload.get("classes", {}).items()}
                row.update({
                    "prompt_bank": str(bank), "prompt_bank_sha256": sha256(bank),
                    "prompt_bank_budget": payload.get("budget_per_class"),
                    "prompt_bank_classes": len(counts),
                    "prompt_bank_minimum_entries": min(counts.values()) if counts else 0,
                    "prompt_bank_maximum_entries": max(counts.values()) if counts else 0,
                })
                if payload.get("budget_per_class") != spec["budget"]:
                    errors.append(
                        f"Wrong bank budget for {spec['key']}: {payload.get('budget_per_class')}"
                    )
                if set(counts.values()) != {spec["budget"]} or len(counts) != 10:
                    errors.append(f"Malformed bank contents for {spec['key']}: {counts}")
                if config_path.is_file():
                    configured_bank = json.loads(config_path.read_text(encoding="utf-8")).get(
                        "sparse_bank"
                    )
                    row["configured_sparse_bank"] = configured_bank
                    if configured_bank and Path(configured_bank).name != bank.name:
                        errors.append(
                            f"Checkpoint/bank filename mismatch for {spec['key']}: "
                            f"{configured_bank} vs {bank}"
                        )
        rows.append(row)

    trainer_source = (HERE / "train_text_to_image_supervision.py").read_text(encoding="utf-8")
    t77_markers = (
        "max_length=self.tokenizer.model_max_length",
        'padding="max_length"',
        "truncation=True",
    )
    if not all(marker in trainer_source for marker in t77_markers):
        errors.append("Controlled trainer no longer provides the audited T77 tokenization path")
    payload = {
        "format_version": 1, "passed": not errors,
        "errors": errors, "checkpoints": rows,
        "shared_generation_seeds": list(args.generation_seeds),
        "classifier_repeats": args.classifier_repeats,
        "fresh_generation_required": True,
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "preflight_audit.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if errors:
        raise RuntimeError("Preflight checkpoint audit failed:\n- " + "\n- ".join(errors))
    return payload


def generation_complete(seed_root, spec, expected):
    def check():
        for prompt in spec["prompts"]:
            condition = f"{spec['supervision']}_{prompt}"
            complete = seed_root / condition / "complete.json"
            records = seed_root / condition / "prompt_records.json"
            if not complete.is_file() or not records.is_file():
                return False
            if int(json.loads(complete.read_text(encoding="utf-8"))["images"]) != expected:
                return False
        return True
    return check


def evaluation_complete(path):
    def check():
        if not Path(path).is_file():
            return False
        text = Path(path).read_text(encoding="utf-8", errors="replace")
        return bool(BEST.search(text) and TAIL.search(text))
    return check


def generation_command(args, spec, output, generation_seed):
    command = [
        sys.executable, str(HERE / "generate_factorial.py"),
        "--prototype", args.prototype, "--dcs", args.dcs,
        "--base-model", args.base_model,
        "--model", f"{spec['supervision']}={spec['model']}",
        "--supervisions", spec["supervision"], "--prompts", *spec["prompts"],
        "--output-root", str(output), "--generation-seeds", str(generation_seed),
        "--ipc", str(args.ipc), "--strength", str(args.strength),
        "--guidance-scale", "10", "--num-inference-steps", "50",
        "--shuffle-shift", "1", "--size", "256", "--resume",
    ]
    if spec.get("prompt_bank") is not None:
        command.extend(("--prompt-bank", str(spec["prompt_bank"])))
    return command


def evaluation_save_dir(tag, ipc):
    return EVAL_DIR / "results" / "imagenet10" / f"resnet10apin_{tag}_rand{ipc}"


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
    tasks, index = {}, []
    for spec in checkpoint_specs(args):
        for generation_seed in args.generation_seeds:
            output = root / "synthetic" / spec["key"]
            seed_root = output / f"seed_{generation_seed}"
            gen_name = f"gen_{spec['key']}_g{generation_seed}"
            tasks[gen_name] = Task(
                gen_name, 1, "generate", generation_command(args, spec, output, generation_seed),
                REPO_ROOT, root / "scheduler_logs" / f"{gen_name}.log",
                generation_complete(seed_root, spec, 10 * args.ipc),
            )
            for prompt in spec["prompts"]:
                condition = f"{spec['supervision']}_{prompt}"
                eval_name = f"eval_{spec['key']}_g{generation_seed}_{prompt}"
                log = root / "evaluation" / spec["key"] / f"seed_{generation_seed}" / f"{prompt}.log"
                tag = f"lowvar_t77_{spec['key']}_g{generation_seed}_{prompt}"
                save_dir = evaluation_save_dir(tag, args.ipc)
                tasks[eval_name] = Task(
                    eval_name, 2, "eval", evaluation_command(args, seed_root / condition, tag),
                    EVAL_DIR, log, evaluation_complete(log), dependencies=(gen_name,),
                )
                index.append({
                    "checkpoint": spec["key"], "training_seed": 0,
                    "generation_seed": int(generation_seed), "prompt": prompt,
                    "condition": condition, "synthetic_dir": str(seed_root / condition),
                    "evaluation_log": str(log), "evaluation_save_dir": str(save_dir),
                    "ipc": args.ipc, "strength": args.strength,
                    "classifier_repeats": args.classifier_repeats, "tail_k": args.tail_k,
                })
    return tasks, index


def parse_metrics(path):
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    best_matches = BEST.findall(text)
    tail_matches = TAIL.findall(text)
    if not best_matches or not tail_matches:
        raise ValueError(f"Missing best or Tail-K metrics: {path}")
    return {
        "best": [float(value) for value in ast.literal_eval(best_matches[-1])],
        "tail_k": int(tail_matches[-1][0]),
        "tail": [float(value) for value in ast.literal_eval(tail_matches[-1][1])],
    }


def percentile(values, probability):
    values = sorted(values)
    return values[min(len(values) - 1, int(probability * len(values)))]


def linear_paired_bootstrap(index, terms, metric, samples=10000, seed=20260818):
    lookup = {
        (row["checkpoint"], row["generation_seed"], row["prompt"]): parse_metrics(
            row["evaluation_log"]
        )[metric]
        for row in index
    }
    generations = sorted({row["generation_seed"] for row in index})
    rng = random.Random(seed)

    def estimate(draw_rng=None):
        selected_generations = generations if draw_rng is None else [
            draw_rng.choice(generations) for _ in generations
        ]
        values = []
        for generation in selected_generations:
            term_values = [
                (coefficient, lookup[(cell[0], generation, cell[1])])
                for coefficient, cell in terms
            ]
            repeat_counts = {len(values) for _, values in term_values}
            if len(repeat_counts) != 1:
                raise ValueError(f"Repeat mismatch for generation {generation}")
            repeat_count = repeat_counts.pop()
            repeats = range(repeat_count) if draw_rng is None else [
                draw_rng.randrange(repeat_count) for _ in range(repeat_count)
            ]
            values.extend(
                sum(coefficient * cell_values[index] for coefficient, cell_values in term_values)
                for index in repeats
            )
        return statistics.fmean(values)

    point = estimate()
    draws = [estimate(rng) for _ in range(samples)]
    return point, percentile(draws, 0.025), percentile(draws, 0.975)


def paired_bootstrap(index, left, right, metric, samples=10000, seed=20260818):
    return linear_paired_bootstrap(
        index, ((1.0, left), (-1.0, right)), metric, samples=samples, seed=seed
    )


def summarize(index, output):
    output.mkdir(parents=True, exist_ok=True)
    performance = []
    for metric in ("best", "tail"):
        for checkpoint in sorted({row["checkpoint"] for row in index}):
            prompts = sorted({
                row["prompt"] for row in index if row["checkpoint"] == checkpoint
            })
            for prompt in prompts:
                selected = [
                    row for row in index
                    if row["checkpoint"] == checkpoint and row["prompt"] == prompt
                ]
                values = [
                    value for row in selected for value in parse_metrics(row["evaluation_log"])[metric]
                ]
                mean, lower, upper = linear_paired_bootstrap(
                    index, ((1.0, (checkpoint, prompt)),), metric,
                    seed=20260818 + len(performance),
                )
                performance.append({
                    "metric": metric, "checkpoint": checkpoint, "prompt": prompt,
                    "mean_accuracy": mean,
                    "bootstrap_ci95_lower": lower, "bootstrap_ci95_upper": upper,
                    "generation_cells": len(selected), "classifier_observations": len(values),
                })
    contrasts = [
        ("matched_correct_minus_shuffled", ("matched_ft", "correct_t77"), ("matched_ft", "shuffled_t77")),
        ("unpaired_correct_minus_shuffled", ("unpaired_ft", "correct_t77"), ("unpaired_ft", "shuffled_t77")),
        ("matched_minus_unpaired_correct", ("matched_ft", "correct_t77"), ("unpaired_ft", "correct_t77")),
        ("matched_minus_unpaired_shuffled", ("matched_ft", "shuffled_t77"), ("unpaired_ft", "shuffled_t77")),
        ("bank_m4_minus_matched_correct", ("bank_m4", "bank_t77"), ("matched_ft", "correct_t77")),
        ("bank_m64_minus_matched_correct", ("bank_m64", "bank_t77"), ("matched_ft", "correct_t77")),
        ("bank_m4_minus_label_anchor", ("bank_m4", "bank_t77"), ("label_ft", "label")),
        ("bank_m64_minus_label_anchor", ("bank_m64", "bank_t77"), ("label_ft", "label")),
        ("label_ft_correct_minus_label", ("label_ft", "correct_t77"), ("label_ft", "label")),
    ]
    contrast_rows = []
    for metric_index, metric in enumerate(("best", "tail")):
        for contrast_index, (name, left, right) in enumerate(contrasts):
            mean, lower, upper = paired_bootstrap(
                index, left, right, metric, seed=20260818 + metric_index * 100 + contrast_index
            )
            contrast_rows.append({
                "metric": metric, "contrast": name, "mean_difference": mean,
                "bootstrap_ci95_lower": lower, "bootstrap_ci95_upper": upper,
                "generation_seed_levels": 6, "paired_classifier_observations": 18,
                "bootstrap_order": "generation seed -> shared classifier repeat",
            })
        interaction_terms = (
            (1.0, ("matched_ft", "correct_t77")),
            (-1.0, ("unpaired_ft", "correct_t77")),
            (-1.0, ("matched_ft", "shuffled_t77")),
            (1.0, ("unpaired_ft", "shuffled_t77")),
        )
        mean, lower, upper = linear_paired_bootstrap(
            index, interaction_terms, metric,
            seed=20260818 + metric_index * 100 + len(contrasts),
        )
        contrast_rows.append({
            "metric": metric, "contrast": "matching_specific_interaction",
            "mean_difference": mean, "bootstrap_ci95_lower": lower,
            "bootstrap_ci95_upper": upper, "generation_seed_levels": 6,
            "paired_classifier_observations": 18,
            "bootstrap_order": "generation seed -> shared classifier repeat",
        })
    for filename, rows in (("performance.csv", performance), ("paired_contrasts.csv", contrast_rows)):
        with (output / filename).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (output / "summary.json").write_text(
        json.dumps({"performance": performance, "paired_contrasts": contrast_rows}, indent=2) + "\n",
        encoding="utf-8",
    )


def semantic_manifest(payload):
    return {key: value for key, value in payload.items() if key != "gpus"}


def main():
    args = parse_args()
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
    if tuple(args.generation_seeds) != tuple(range(6)):
        raise ValueError("The preregistered matrix requires fresh generation seeds 0-5")
    if args.classifier_repeats != 3 or args.tail_k not in range(5, 11):
        raise ValueError("The matrix requires C=3 and Tail-K between 5 and 10")
    gpus = [value.strip() for value in args.gpus.split(",") if value.strip()]
    if len(gpus) != 2:
        raise ValueError("This schedule is preregistered for exactly two GPUs")
    for path in (args.data_root, args.base_model, args.prototype, args.dcs):
        if not Path(path).exists():
            raise FileNotFoundError(path)
    root = Path(args.run_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    audit_models(args, root)
    if args.audit_only:
        print(f"Preflight audit passed: {root / 'preflight_audit.json'}")
        return
    tasks, index = build_tasks(args)
    manifest = {
        "format_version": 1, "experiment": "fresh_low_variance_t77_matrix",
        "gpus": gpus, "generation_seeds": list(args.generation_seeds),
        "classifier_repeats": args.classifier_repeats, "classifier_seed": args.classifier_seed,
        "tail_k": args.tail_k, "ipc": args.ipc, "strength": args.strength,
        "guidance_scale": 10.0, "num_inference_steps": 50,
        "fresh_generation_no_historical_reuse": True,
        "cells": [
            {"checkpoint": spec["key"], "prompts": list(spec["prompts"]), "model": str(spec["model"])}
            for spec in checkpoint_specs(args)
        ],
    }
    manifest_path = root / "run_manifest.json"
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if semantic_manifest(previous) != semantic_manifest(manifest):
            raise RuntimeError(f"Resume configuration differs from {manifest_path}")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (root / "evaluation_index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    completed = {name for name, task in tasks.items() if task.complete()}
    running = {}
    print(f"Low-variance T77 matrix: {len(completed)}/{len(tasks)} tasks complete", flush=True)
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
        ready = [
            task for task in tasks.values()
            if task.name not in completed and task.process is None and task.next_ready <= now
            and all(dependency in completed for dependency in task.dependencies)
        ]
        ready.sort(key=lambda task: (
            task.stage, {"eval": 0, "generate": 1, "train": 2}[task.kind],
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
    summarize(index, root / "summary")
    (root / "COMPLETE").write_text(time.strftime("%F %T\n"), encoding="utf-8")
    print(f"Low-variance T77 matrix complete: {root}")


if __name__ == "__main__":
    main()
