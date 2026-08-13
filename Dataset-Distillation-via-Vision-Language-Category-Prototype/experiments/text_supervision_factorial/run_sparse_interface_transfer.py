#!/usr/bin/env python3
"""Run the paired Sparse-m4/Matched-FT inference-interface transfer test."""

import argparse
import hashlib
import json
import os
import shlex
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

from run_generality import EVAL_DIR, REPO_ROOT, Task, complete_eval, eval_command
from run_sparse_prompt_search import write_state


HERE = Path(__file__).resolve().parent
PROMPTS = ("label", "correct", "shuffled", "bank")
FAMILIES = ("sparse_m4_ft", "matched_ft")


def append_scheduler_event(root, event, **fields):
    """Persist scheduler lifecycle events even when a child log is archived."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "time": time.strftime("%F %T"),
        "event": event,
        "scheduler_pid": os.getpid(),
        **fields,
    }
    with (root / "scheduler_events.jsonl").open("a", encoding="utf-8", buffering=1) as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def acquire_run_lock(root):
    """Prevent two scheduler processes from mutating the same run directory."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / "scheduler.lock"
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        import fcntl
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        handle.seek(0)
        owner = handle.read().strip() or "unknown owner"
        handle.close()
        raise RuntimeError(
            f"Another sparse-interface scheduler already holds {lock_path}: {owner}"
        ) from error
    except ImportError:
        # This experiment runs on Linux; retain importability for local Windows tests.
        pass
    handle.seek(0)
    handle.truncate()
    handle.write(json.dumps({"pid": os.getpid(), "started_at": time.strftime("%F %T")}) + "\n")
    handle.flush()
    os.fsync(handle.fileno())
    return handle


def restore_attempt_counts(root, tasks):
    """Keep attempt numbering monotonic across an intentional scheduler restart."""
    path = Path(root) / "scheduler_events.jsonl"
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("event") != "launch" or row.get("task") not in tasks:
            continue
        task = tasks[row["task"]]
        task.attempts = max(task.attempts, int(row.get("attempt", 0)))


def decoded_returncode(code):
    if code is None:
        return None
    if code < 0:
        try:
            return signal.Signals(-code).name
        except ValueError:
            return f"signal_{-code}"
    if code >= 128:
        try:
            return signal.Signals(code - 128).name
        except ValueError:
            pass
    return "normal_exit"


def launch_task(task, gpu, args, root):
    task.log.parent.mkdir(parents=True, exist_ok=True)
    task.handle = task.log.open("a", encoding="utf-8", buffering=1)
    attempt = task.attempts + 1
    command = list(task.command)
    task.handle.write(
        f"\n[{time.strftime('%F %T')}] attempt {attempt} GPU {gpu} "
        f"scheduler_pid={os.getpid()}\n{shlex.join(command)}\n"
    )
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": gpu,
        "PYTHONUNBUFFERED": "1",
        "TOKENIZERS_PARALLELISM": "false",
    })
    if args.diffusers_src:
        env["PYTHONPATH"] = args.diffusers_src + (
            os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
        )
    task.process = subprocess.Popen(
        command, cwd=task.cwd, env=env, stdout=task.handle, stderr=subprocess.STDOUT
    )
    task.attempts = attempt
    append_scheduler_event(
        root, "launch", task=task.name, kind=task.kind, gpu=str(gpu), attempt=attempt,
        child_pid=task.process.pid, log=str(task.log), command=command,
    )
    print(
        f"LAUNCH GPU {gpu}: {task.name} (attempt {attempt}, pid {task.process.pid})",
        flush=True,
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--prototype", required=True)
    parser.add_argument("--dcs", required=True)
    parser.add_argument("--prompt-bank", default="", help="Legacy fallback bank for every training seed")
    parser.add_argument("--sparse-seed0-bank", default="")
    parser.add_argument("--sparse-seed1-bank", default="")
    parser.add_argument("--sparse-seed0-model", required=True)
    parser.add_argument("--sparse-seed1-model", required=True)
    parser.add_argument("--matched-seed0-model", required=True)
    parser.add_argument("--matched-seed1-model", required=True)
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


def bank_semantic_payload(path):
    """Return only prompt-bank content that can affect conditioning."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    classes = payload.get("classes")
    if not isinstance(classes, dict) or not classes:
        raise ValueError(f"Invalid sparse prompt bank: {path}")
    normalized = {}
    for class_key, rows in sorted(classes.items()):
        normalized[class_key] = [
            {
                "relative": str(row.get("relative", "")),
                "caption": str(row.get("caption", "")),
                "nested_rank": int(row.get("nested_rank", rank)),
            }
            for rank, row in enumerate(rows)
        ]
    return normalized


def bank_semantic_sha256(path):
    encoded = json.dumps(
        bank_semantic_payload(path), ensure_ascii=False, sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def model_path(args, family, seed):
    return Path(getattr(args, f"{family.split('_')[0]}_seed{seed}_model")).resolve()


def prompt_bank_path(args, seed):
    explicit = getattr(args, f"sparse_seed{seed}_bank", "")
    selected = explicit or args.prompt_bank
    if not selected:
        raise ValueError(f"No sparse m4 prompt bank configured for training seed {seed}")
    return Path(selected).resolve()


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


def generation_command(args, family, model, prompt_bank, output, generation_seed, prompts):
    supervision = "sparse_ft" if family == "sparse_m4_ft" else "matched_ft"
    return [
        sys.executable, str(HERE / "generate_factorial.py"),
        "--prototype", args.prototype, "--dcs", args.dcs,
        "--base-model", args.base_model, "--model", f"{supervision}={model}",
        "--supervisions", supervision, "--prompts", *prompts,
        "--prompt-bank", str(prompt_bank), "--output-root", str(output),
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
            prompt_bank = prompt_bank_path(args, training_seed)
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
                    generation_command(
                        args, family, model, prompt_bank, output, generation_seed, missing
                    ),
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
    if not gpus:
        raise ValueError("At least one GPU is required")
    restore_attempt_counts(root, tasks)
    completed = {name for name, task in tasks.items() if task.complete()}
    running = {}
    append_scheduler_event(
        root, "scheduler_start", gpus=gpus, tasks=len(tasks), completed=len(completed)
    )
    print(f"Sparse-interface transfer: {len(completed)}/{len(tasks)} tasks complete", flush=True)
    while len(completed) < len(tasks):
        now = time.time()
        for gpu, task in list(running.items()):
            code = task.process.poll()
            if code is None:
                continue
            exit_kind = decoded_returncode(code)
            task.handle.write(
                f"[{time.strftime('%F %T')}] exit code {code} ({exit_kind}) "
                f"scheduler_pid={os.getpid()} child_pid={task.process.pid}\n"
            )
            task.handle.close()
            child_pid = task.process.pid
            task.process = task.handle = None
            del running[gpu]
            if code == 0 and task.complete():
                completed.add(task.name)
                append_scheduler_event(
                    root, "complete", task=task.name, gpu=str(gpu), attempt=task.attempts,
                    child_pid=child_pid, returncode=code, exit_kind=exit_kind,
                )
                print(f"DONE GPU {gpu}: {task.name} ({len(completed)}/{len(tasks)})", flush=True)
            else:
                failed = task.log.with_name(f"{task.log.name}.failed_{task.attempts}_{time.strftime('%Y%m%dT%H%M%S')}")
                if task.log.exists():
                    task.log.replace(failed)
                task.log.write_text(
                    f"Previous attempt {task.attempts} exited with {code} ({exit_kind}).\n"
                    f"Archived log: {failed}\n"
                    f"Scheduler event log: {root / 'scheduler_events.jsonl'}\n",
                    encoding="utf-8",
                )
                append_scheduler_event(
                    root, "failure", task=task.name, gpu=str(gpu), attempt=task.attempts,
                    child_pid=child_pid, returncode=code, exit_kind=exit_kind,
                    archived_log=str(failed), retry_delay_seconds=args.retry_delay_seconds,
                )
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
            launch_task(task, gpu, args, root)
            running[gpu] = task
        write_state(root, tasks, completed, running)
        time.sleep(5)
    append_scheduler_event(root, "scheduler_complete", tasks=len(tasks), completed=len(completed))


def main():
    args = parse_args()
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
    required = [args.data_root, args.base_model, args.prototype, args.dcs]
    required += [str(model_path(args, family, seed)) for family in FAMILIES for seed in args.training_seeds]
    required += [str(prompt_bank_path(args, seed)) for seed in args.training_seeds]
    for path in required:
        if not Path(path).exists():
            raise FileNotFoundError(path)
    bank_semantic_hashes = {
        seed: bank_semantic_sha256(prompt_bank_path(args, seed))
        for seed in sorted(set(args.training_seeds))
    }
    if len(set(bank_semantic_hashes.values())) != 1:
        raise RuntimeError(
            "Sparse m4 banks differ in selected class-caption content: "
            + json.dumps(bank_semantic_hashes, sort_keys=True)
        )
    root = Path(args.run_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    run_lock = acquire_run_lock(root)
    append_scheduler_event(root, "lock_acquired", lock=str(root / "scheduler.lock"))
    reuse = load_reuse(args.reuse_index)
    tasks, index = build_tasks(args, reuse)
    manifest = {
        "format_version": 1, "experiment": "sparse_interface_transfer",
        "data_root": str(Path(args.data_root).resolve()), "base_model": str(Path(args.base_model).resolve()),
        "prototype": str(Path(args.prototype).resolve()), "prototype_sha256": sha256(args.prototype),
        "dcs": str(Path(args.dcs).resolve()), "dcs_sha256": sha256(args.dcs),
        "prompt_banks": {
            str(seed): {
                "path": str(prompt_bank_path(args, seed)),
                "file_sha256": sha256(prompt_bank_path(args, seed)),
                "semantic_sha256": bank_semantic_hashes[seed],
            }
            for seed in sorted(set(args.training_seeds))
        },
        "training_seeds": sorted(set(args.training_seeds)), "generation_seeds": sorted(set(args.generation_seeds)),
        "ipc": args.ipc, "strength": args.strength, "prompts": list(PROMPTS),
        "classifier_repeats": args.classifier_repeats,
        "classifier_seed": args.classifier_seed,
        "guidance_scale": args.guidance_scale,
        "num_inference_steps": args.num_inference_steps,
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
    append_scheduler_event(root, "experiment_complete")
    run_lock.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        run_root = None
        try:
            argv = sys.argv
            if "--run-root" in argv:
                run_root = Path(argv[argv.index("--run-root") + 1]).resolve()
                append_scheduler_event(
                    run_root, "scheduler_fatal", error_type=type(error).__name__,
                    error=str(error), traceback=traceback.format_exc(),
                )
                (run_root / "scheduler_fatal.log").write_text(
                    traceback.format_exc(), encoding="utf-8"
                )
        except Exception:
            pass
        raise
