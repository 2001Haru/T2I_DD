#!/usr/bin/env python
"""Compare the executed img2img timestep schedules across Diffusers installs.

This audit intentionally loads only the scheduler config. It does not load model
weights, allocate a GPU, encode an image, or run the UNet.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--official-diffusers", required=True)
    parser.add_argument("--strengths", type=float, nargs="+", default=(0.7, 0.8, 0.9, 1.0))
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--output", default="timestep_protocol_audit.json")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--pipeline", choices=("img2img", "latents2img"), help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", help=argparse.SUPPRESS)
    return parser.parse_args()


def package_source(path: Path) -> Path:
    path = path.resolve()
    if (path / "src" / "diffusers" / "__init__.py").is_file():
        return path / "src"
    if (path / "diffusers" / "__init__.py").is_file():
        return path
    raise FileNotFoundError(f"No importable diffusers package under {path}")


def digest_json(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def run_worker(args):
    import diffusers

    if args.pipeline == "img2img":
        from diffusers import StableDiffusionImg2ImgPipeline as pipeline_class
    else:
        from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_latents2img import (
            StableDiffusionLatents2ImgPipeline as pipeline_class,
        )

    scheduler_config_path = Path(args.base_model) / "scheduler" / "scheduler_config.json"
    scheduler_config = json.loads(scheduler_config_path.read_text(encoding="utf-8"))
    scheduler_class = getattr(diffusers, scheduler_config["_class_name"])
    scheduler = scheduler_class.from_pretrained(args.base_model, subfolder="scheduler")
    schedules = {}
    for strength in args.strengths:
        scheduler.set_timesteps(args.num_inference_steps, device="cpu")
        full = [int(value) for value in scheduler.timesteps.cpu().tolist()]
        holder = SimpleNamespace(scheduler=scheduler)
        executed, effective_steps = pipeline_class.get_timesteps(
            holder, args.num_inference_steps, strength, "cpu"
        )
        schedules[strength_token(strength)] = {
            "strength": strength,
            "effective_inference_steps": int(effective_steps),
            "full_scheduler_timesteps": full,
            "executed_timesteps": [int(value) for value in executed.cpu().tolist()],
        }
    result = {
        "diffusers_version": diffusers.__version__,
        "diffusers_module": str(Path(diffusers.__file__).resolve()),
        "pipeline": pipeline_class.__name__,
        "scheduler": scheduler_class.__name__,
        "scheduler_config_sha256": digest_json(scheduler_config),
        "scheduler_config": scheduler_config,
        "num_inference_steps": args.num_inference_steps,
        "schedules": schedules,
    }
    Path(args.worker_output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def strength_token(value: float) -> str:
    return format(value, ".12g")


def invoke_worker(args, pipeline: str, output: Path, python_path: Path | None):
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--pipeline",
        pipeline,
        "--worker-output",
        str(output),
        "--base-model",
        str(Path(args.base_model).resolve()),
        "--num-inference-steps",
        str(args.num_inference_steps),
        "--strengths",
        *[str(value) for value in args.strengths],
        "--official-diffusers",
        str(Path(args.official_diffusers).resolve()),
    ]
    environment = os.environ.copy()
    if python_path is not None:
        old = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = str(python_path) + (os.pathsep + old if old else "")
    subprocess.run(command, check=True, env=environment)


def compare(current, official):
    rows = []
    all_equal = True
    for key in current["schedules"]:
        left = current["schedules"][key]["executed_timesteps"]
        right = official["schedules"][key]["executed_timesteps"]
        equal = left == right
        all_equal &= equal
        first_difference = next(
            (index for index, pair in enumerate(zip(left, right)) if pair[0] != pair[1]),
            None,
        )
        if first_difference is None and len(left) != len(right):
            first_difference = min(len(left), len(right))
        rows.append({
            "strength": current["schedules"][key]["strength"],
            "exact_match": equal,
            "current_count": len(left),
            "official_count": len(right),
            "current_first": left[0] if left else None,
            "official_first": right[0] if right else None,
            "current_last": left[-1] if left else None,
            "official_last": right[-1] if right else None,
            "first_difference_index": first_difference,
        })
    return all_equal, rows


def main():
    args = parse_args()
    if args.worker:
        run_worker(args)
        return
    official_source = package_source(Path(args.official_diffusers))
    with tempfile.TemporaryDirectory(prefix="vlcp-timestep-audit-") as temporary:
        temporary = Path(temporary)
        current_path = temporary / "current.json"
        official_path = temporary / "official.json"
        invoke_worker(args, "img2img", current_path, None)
        invoke_worker(args, "latents2img", official_path, official_source)
        current = json.loads(current_path.read_text(encoding="utf-8"))
        official = json.loads(official_path.read_text(encoding="utf-8"))
    exact, comparisons = compare(current, official)
    result = {
        "format_version": 1,
        "decision": "exact_match" if exact else "mismatch",
        "interpretation": (
            "The two pipelines execute identical integer timestep sequences at every audited strength."
            if exact else
            "At least one strength executes a different integer timestep sequence; strength results are not directly comparable."
        ),
        "current": current,
        "official": official,
        "comparisons": comparisons,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"decision": result["decision"], "comparisons": comparisons}, indent=2))
    print(f"Wrote full audit: {output.resolve()}")


if __name__ == "__main__":
    main()
