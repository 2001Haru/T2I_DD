#!/usr/bin/env python3
"""Run targeted IPC50 checkpoint-by-visual-interface follow-up matrices."""

import argparse
import json
import signal
import subprocess
import sys
import time
from pathlib import Path

from run_conditioning_interface_matrix import (
    add_prompt_grid,
    cell_key,
    choose_ready,
    load_reuse_catalog,
    nette_artifacts,
    nette_model,
)
from run_generality import complete_model, launch, write_scheduler_state
from subset_specs import validate_subset


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
NEW_STRENGTHS = (0.70, 0.90)
REFERENCE_STRENGTHS = (0.70, 0.80, 0.90, 1.00)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nette-data-root", required=True)
    parser.add_argument("--nette-caption-file", required=True)
    parser.add_argument("--woof-data-root", required=True)
    parser.add_argument("--woof-caption-file", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--base-run-root", required=True)
    parser.add_argument("--causal-run-root", required=True)
    parser.add_argument("--generality-run-root", required=True)
    parser.add_argument("--interface-run-root", required=True)
    parser.add_argument(
        "--woof-model-root", default="",
        help="Run root containing models/woof/train_seed_*/; defaults to interface-run-root with a generality-run sibling fallback",
    )
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--reuse-index", action="append", default=[])
    parser.add_argument("--specs", nargs="+", choices=("nette", "woof"), default=("nette", "woof"))
    parser.add_argument("--training-seeds", type=int, nargs="+", default=(0, 1))
    parser.add_argument("--generation-seeds", type=int, nargs="+", default=(0, 1))
    parser.add_argument(
        "--supervisions", nargs="+", choices=("label_ft", "unpaired_ft", "matched_ft"),
        default=("label_ft", "matched_ft"),
    )
    parser.add_argument(
        "--matrices", nargs="+", choices=("E", "F", "R"), default=("E", "F"),
        help="E=schedule-matched noise, F=pure noise, R=prototype initialization",
    )
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--classifier-repeats", type=int, default=2)
    parser.add_argument("--classifier-seed", type=int, default=0)
    parser.add_argument("--guidance-scale", type=float, default=10.0)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--max-parallel-evals", type=int, default=4)
    parser.add_argument("--retry-delay-seconds", type=int, default=120)
    parser.add_argument("--max-retries", type=int, default=0)
    parser.add_argument("--max-walltime-hours", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--diffusers-src", default="")
    return parser.parse_args()


def woof_artifacts(args):
    root = Path(args.interface_run_root).resolve() / "artifacts" / "woof" / "ipc50"
    return root / "woof-ipc50-0.7-30-kmexpand1.json", root / "dcs.json"


def model_for(args, spec, supervision, training_seed):
    if spec == "nette":
        return nette_model(args, supervision, training_seed)
    interface_root = Path(args.interface_run_root).resolve()
    roots = (
        [Path(args.woof_model_root).resolve()]
        if args.woof_model_root
        else [
            interface_root,
            interface_root.parent / "conditioning_interface_generality_v0",
        ]
    )
    candidates = [
        root / "models" / "woof" / f"train_seed_{training_seed}" / supervision
        for root in roots
    ]
    for candidate in candidates:
        if complete_model(candidate)():
            return candidate
    return candidates[0]


def add_reference_rows(index, reuse, spec, training_seed, generation_seed):
    for strength in REFERENCE_STRENGTHS:
        for prompt in ("label", "correct", "shuffled"):
            metadata = {
                "matrix": "R", "spec": spec, "ipc": 50, "visual_mode": "prototype",
                "strength": strength, "supervision": "matched_ft",
                "training_seed": training_seed, "generation_seed": generation_seed,
                "prompt": prompt, "shuffle_shift": 1 if prompt == "shuffled" else None,
                "phase": "reused_prototype_reference",
            }
            reused = reuse.get(cell_key(metadata))
            if reused:
                index.append({**metadata, **{
                    key: value for key, value in reused.items()
                    if key in {"evaluation_log", "synthetic_dir"}
                }, "source": "reused"})


def build_tasks(args, reuse):
    tasks, index = {}, []
    matrices = tuple(getattr(args, "matrices", ("E", "F")))
    roots = {
        "nette": Path(args.nette_data_root).resolve(),
        "woof": Path(args.woof_data_root).resolve(),
    }
    artifacts = {"nette": nette_artifacts(args, 50), "woof": woof_artifacts(args)}
    for spec in args.specs:
        prototype, dcs = artifacts[spec]
        for path in (prototype, dcs):
            if not path.is_file():
                raise FileNotFoundError(path)
        for supervision in args.supervisions:
            for training_seed in args.training_seeds:
                model = model_for(args, spec, supervision, training_seed)
                if not complete_model(model)():
                    raise RuntimeError(f"Missing {supervision} checkpoint: {model}")
                for generation_seed in args.generation_seeds:
                    if "E" in matrices:
                        for strength in NEW_STRENGTHS:
                            add_prompt_grid(
                                tasks, index, args, reuse, "E", spec, roots[spec], 50,
                                prototype, dcs, model, supervision, training_seed,
                                generation_seed, strength, "schedule_matched_noise",
                                stage=1, phase="schedule_matched_content_control",
                            )
                    if "F" in matrices:
                        add_prompt_grid(
                            tasks, index, args, reuse, "F", spec, roots[spec], 50,
                            prototype, dcs, model, supervision, training_seed,
                            generation_seed, None, "pure_noise", extra_shifts=(2, 4, 7),
                            stage=2, phase="pure_noise_ipc50_endpoint",
                        )
                    if "R" in matrices:
                        for strength in NEW_STRENGTHS:
                            add_prompt_grid(
                                tasks, index, args, reuse, "R", spec, roots[spec], 50,
                                prototype, dcs, model, supervision, training_seed,
                                generation_seed, strength, "prototype",
                                stage=3, phase="prototype_checkpoint_control",
                            )
    return tasks, index


def write_manifest(args, index):
    root = Path(args.run_root).resolve()
    matrices = list(getattr(args, "matrices", ("E", "F")))
    question = (
        "Separate rich-caption marginal supervision from training-time matching covariance "
        "under prototype initialization"
        if matrices == ["R"]
        else "Separate prototype content from the shortened img2img schedule"
    )
    payload = {
        "format_version": 1,
        "question": question,
        "matrices": matrices,
        "ipc": 50,
        "specs": list(args.specs),
        "schedule_matched_strengths": list(NEW_STRENGTHS),
        "reference_strengths": list(REFERENCE_STRENGTHS),
        "pure_noise_shuffle_shifts": [1, 2, 4, 7],
        "training_seeds": list(args.training_seeds),
        "generation_seeds": list(args.generation_seeds),
        "supervisions": list(args.supervisions),
        "preregistered_checkpoint_contrasts": [
            "unpaired_ft_minus_label_ft: rich-caption marginal supervision",
            "matched_ft_minus_unpaired_ft: training-time image-caption covariance",
        ],
        "preregistered_inference_contrast": (
            "correct_minus_shuffled: inference-time cluster correspondence"
        ),
        "classifier_repeats": args.classifier_repeats,
        "gpus": args.gpus,
        "base_model": str(Path(args.base_model).resolve()),
        "interface_run_root": str(Path(args.interface_run_root).resolve()),
        "woof_model_root": (
            str(Path(args.woof_model_root).resolve()) if args.woof_model_root else "auto"
        ),
        "reuse_indexes": [str(Path(path).resolve()) for path in args.reuse_index],
    }
    path = root / "run_manifest.json"
    if path.is_file() and json.loads(path.read_text(encoding="utf-8")) != payload:
        raise RuntimeError(f"Resume configuration differs from {path}")
    if not path.is_file():
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (root / "evaluation_index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def summarize(root, allow_incomplete=False):
    command = [
        sys.executable, str(HERE / "summarize_conditioning_interface_matrix.py"),
        "--evaluation-index", str(root / "evaluation_index.json"),
        "--output-dir", str(root / ("summary_partial" if allow_incomplete else "summary")),
    ]
    if allow_incomplete:
        command.append("--allow-incomplete")
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def main():
    args = parse_args()
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
    for spec, root, captions in (
        ("nette", args.nette_data_root, args.nette_caption_file),
        ("woof", args.woof_data_root, args.woof_caption_file),
    ):
        if spec in args.specs:
            validate_subset(root, spec, captions)
    gpus = [value.strip() for value in args.gpus.split(",") if value.strip()]
    if not gpus:
        raise ValueError("At least one GPU is required")
    root = Path(args.run_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    reuse = load_reuse_catalog(args.reuse_index)
    tasks, index = build_tasks(args, reuse)
    write_manifest(args, index)
    completed = {name for name, task in tasks.items() if task.complete()}
    running = {}
    started = time.time()
    deadline = started + args.max_walltime_hours * 3600 if args.max_walltime_hours else None
    print(f"Schedule-matched follow-up: {len(index)} cells, {len(completed)}/{len(tasks)} tasks complete", flush=True)
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
                task.next_ready = now + args.retry_delay_seconds
        if deadline is None or now < deadline:
            for gpu in [value for value in gpus if value not in running]:
                active_evals = sum(task.kind == "eval" for task in running.values())
                selected = choose_ready(
                    tasks, completed, running, now, active_evals,
                    min(args.max_parallel_evals, len(gpus)),
                )
                if selected is None:
                    break
                launch(selected, gpu, args)
                running[gpu] = selected
        elif not running:
            write_scheduler_state(root, tasks, completed, running)
            summarize(root, allow_incomplete=True)
            print("Walltime reached; active tasks completed and the run is resume-safe.", flush=True)
            return
        write_scheduler_state(root, tasks, completed, running)
        time.sleep(5)
    summarize(root)
    (root / "COMPLETE").write_text(time.strftime("%F %T\n"), encoding="utf-8")


if __name__ == "__main__":
    main()
