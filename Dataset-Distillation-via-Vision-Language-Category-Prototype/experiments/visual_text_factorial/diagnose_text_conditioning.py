import argparse
import csv
import json
import math
import statistics
from pathlib import Path

import torch
import torch.nn.functional as functional

from diagnostic_common import load_consistent_manifest, parse_shift_runs


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure correct versus shuffled DCS in SD 1.5 conditioning space"
    )
    parser.add_argument("--base-run-root", required=True)
    parser.add_argument(
        "--shuffle-run",
        action="append",
        default=[],
        metavar="SHIFT=RUN_ROOT",
    )
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def pad_hidden(hidden, length):
    if hidden.shape[0] == length:
        return hidden
    return functional.pad(hidden, (0, 0, 0, length - hidden.shape[0]))


def pair_metrics(correct, shuffled, epsilon=1e-12):
    length = max(correct["hidden"].shape[0], shuffled["hidden"].shape[0])
    correct_hidden = pad_hidden(correct["hidden"], length)
    shuffled_hidden = pad_hidden(shuffled["hidden"], length)
    correct_flat = correct_hidden.flatten()
    shuffled_flat = shuffled_hidden.flatten()
    denominator = 0.5 * (
        torch.linalg.vector_norm(correct_flat)
        + torch.linalg.vector_norm(shuffled_flat)
    )
    correct_tokens = set(correct["content_token_ids"])
    shuffled_tokens = set(shuffled["content_token_ids"])
    union = correct_tokens | shuffled_tokens
    return {
        "mean_hidden_cosine": functional.cosine_similarity(
            correct["mean_hidden"], shuffled["mean_hidden"], dim=0
        ).item(),
        "flat_hidden_cosine": functional.cosine_similarity(
            correct_flat, shuffled_flat, dim=0
        ).item(),
        "symmetric_relative_l2": (
            torch.linalg.vector_norm(correct_flat - shuffled_flat)
            / denominator.clamp_min(epsilon)
        ).item(),
        "token_jaccard": len(correct_tokens & shuffled_tokens) / max(len(union), 1),
        "correct_token_count": correct["token_count"],
        "shuffled_token_count": shuffled["token_count"],
        "correct_conditioning_length": correct["hidden"].shape[0],
        "shuffled_conditioning_length": shuffled["hidden"].shape[0],
        "correct_chunk_count": correct["chunk_count"],
        "shuffled_chunk_count": shuffled["chunk_count"],
    }


class ConditioningEncoder:
    def __init__(self, model_root, device):
        from transformers import CLIPTextModel, CLIPTokenizer

        self.device = torch.device(device)
        self.tokenizer = CLIPTokenizer.from_pretrained(
            model_root, subfolder="tokenizer", local_files_only=True
        )
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.encoder = CLIPTextModel.from_pretrained(
            model_root,
            subfolder="text_encoder",
            local_files_only=True,
            torch_dtype=dtype,
        ).to(self.device)
        self.encoder.eval()
        self.cache = {}

    @torch.inference_mode()
    def encode(self, prompt, negative_prompt):
        key = (prompt, negative_prompt)
        if key in self.cache:
            return self.cache[key]

        prompt_unpadded = self.tokenizer(
            prompt, return_tensors="pt", truncation=False
        )
        negative_unpadded = self.tokenizer(
            negative_prompt, return_tensors="pt", truncation=False
        )
        sequence_length = max(
            prompt_unpadded.input_ids.shape[-1],
            negative_unpadded.input_ids.shape[-1],
        )
        encoded = self.tokenizer(
            prompt,
            truncation=False,
            padding="max_length",
            max_length=sequence_length,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids = encoded.input_ids.to(self.device)
        attention_mask = encoded.attention_mask[0].to(self.device).bool()
        max_length = self.tokenizer.model_max_length
        chunks = []
        for start in range(0, sequence_length, max_length):
            chunk_ids = input_ids[:, start : start + max_length]
            chunks.append(self.encoder(chunk_ids)[0][0].float().cpu())
        hidden = torch.cat(chunks, dim=0)
        mask = attention_mask.cpu()
        mean_hidden = hidden[mask].mean(dim=0)
        active_ids = input_ids[0, attention_mask].detach().cpu().tolist()
        special_ids = set(self.tokenizer.all_special_ids)
        result = {
            "hidden": hidden,
            "mean_hidden": mean_hidden,
            "token_count": int(mask.sum().item()),
            "chunk_count": math.ceil(sequence_length / max_length),
            "content_token_ids": [
                int(token_id) for token_id in active_ids if token_id not in special_ids
            ],
        }
        self.cache[key] = result
        return result


def keyed_records(manifest):
    return {
        (record["synset"], int(record["image_index"])): record
        for record in manifest["prompt_records"]
    }


def mean_or_zero(values):
    return statistics.fmean(values) if values else 0.0


def write_csv(path, rows, fieldnames):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(rows, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as pyplot

    shifts = [row["shuffle_shift"] for row in rows]
    figure, axes = pyplot.subplots(1, 3, figsize=(15, 4.5))
    panels = (
        ("symmetric_relative_l2_mean", "Conditioning relative L2", False),
        ("mean_hidden_cosine_mean", "Mean-hidden cosine", True),
        ("token_jaccard_mean", "Content-token Jaccard", True),
    )
    for axis, (field, title, similarity) in zip(axes, panels):
        values = [row[field] for row in rows]
        errors = [row[field.replace("_mean", "_std")] for row in rows]
        axis.errorbar(shifts, values, yerr=errors, marker="o", capsize=4)
        axis.set_title(title)
        axis.set_xlabel("Within-class cyclic shift")
        if similarity:
            axis.set_ylim(0, 1.02)
        axis.grid(alpha=0.25)
    figure.suptitle("Correct DCS versus shuffled DCS in SD 1.5 conditioning space")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    pyplot.close(figure)


def main():
    args = parse_args()
    base_run_root = Path(args.base_run_root).resolve()
    shuffle_runs = {1: base_run_root, **parse_shift_runs(args.shuffle_run)}
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    correct_manifest, generation_seeds = load_consistent_manifest(
        base_run_root, "prototype_dcs"
    )
    correct_records = keyed_records(correct_manifest)
    negative_prompt = correct_manifest["negative_prompt"]
    encoder = ConditioningEncoder(args.base_model, args.device)

    pair_rows = []
    for shift, run_root in sorted(shuffle_runs.items()):
        shuffled_manifest, shuffled_seeds = load_consistent_manifest(
            run_root, "prototype_dcs_shuffled"
        )
        if shuffled_seeds != generation_seeds:
            raise RuntimeError(
                f"Generation seed mismatch for shift {shift}: "
                f"{shuffled_seeds} != {generation_seeds}"
            )
        manifest_shift = int(shuffled_manifest["shuffle_strategy"]["shift"])
        if manifest_shift != shift:
            raise RuntimeError(
                f"Shift {shift} points to a manifest with shift {manifest_shift}"
            )
        if shuffled_manifest["negative_prompt"] != negative_prompt:
            raise RuntimeError(f"Negative prompt mismatch for shift {shift}")
        shuffled_records = keyed_records(shuffled_manifest)
        if set(shuffled_records) != set(correct_records):
            raise RuntimeError(f"Prompt record keys differ for shift {shift}")

        for key, correct_record in correct_records.items():
            shuffled_record = shuffled_records[key]
            correct_encoding = encoder.encode(correct_record["prompt"], negative_prompt)
            shuffled_encoding = encoder.encode(shuffled_record["prompt"], negative_prompt)
            row = {
                "shuffle_shift": shift,
                "synset": key[0],
                "image_index": key[1],
                "prototype_index": int(correct_record["prototype_index"]),
                "correct_prompt_source_index": correct_record["prompt_source_index"],
                "shuffled_prompt_source_index": shuffled_record["prompt_source_index"],
                "correct_prompt": correct_record["prompt"],
                "shuffled_prompt": shuffled_record["prompt"],
            }
            row.update(pair_metrics(correct_encoding, shuffled_encoding))
            pair_rows.append(row)

    metric_names = (
        "mean_hidden_cosine",
        "flat_hidden_cosine",
        "symmetric_relative_l2",
        "token_jaccard",
    )
    summary_rows = []
    for shift in sorted(shuffle_runs):
        selected = [row for row in pair_rows if row["shuffle_shift"] == shift]
        summary = {"shuffle_shift": shift, "pairs": len(selected)}
        for metric in metric_names:
            values = [float(row[metric]) for row in selected]
            summary[f"{metric}_mean"] = mean_or_zero(values)
            summary[f"{metric}_std"] = statistics.pstdev(values) if len(values) > 1 else 0.0
        summary["correct_over_77_fraction"] = mean_or_zero(
            [row["correct_token_count"] > 77 for row in selected]
        )
        summary["shuffled_over_77_fraction"] = mean_or_zero(
            [row["shuffled_token_count"] > 77 for row in selected]
        )
        summary_rows.append(summary)

    class_summary_rows = []
    for shift in sorted(shuffle_runs):
        synsets = sorted(
            {row["synset"] for row in pair_rows if row["shuffle_shift"] == shift}
        )
        for synset in synsets:
            selected = [
                row
                for row in pair_rows
                if row["shuffle_shift"] == shift and row["synset"] == synset
            ]
            summary = {
                "shuffle_shift": shift,
                "synset": synset,
                "pairs": len(selected),
            }
            for metric in metric_names:
                values = [float(row[metric]) for row in selected]
                summary[f"{metric}_mean"] = mean_or_zero(values)
                summary[f"{metric}_std"] = (
                    statistics.pstdev(values) if len(values) > 1 else 0.0
                )
            class_summary_rows.append(summary)

    pair_fields = list(pair_rows[0])
    summary_fields = list(summary_rows[0])
    write_csv(output_dir / "conditioning_pairs.csv", pair_rows, pair_fields)
    write_csv(output_dir / "conditioning_shift_summary.csv", summary_rows, summary_fields)
    write_csv(
        output_dir / "conditioning_class_summary.csv",
        class_summary_rows,
        list(class_summary_rows[0]),
    )
    (output_dir / "conditioning_summary.json").write_text(
        json.dumps(
            {
                "base_run_root": str(base_run_root),
                "shuffle_runs": {
                    str(shift): str(root) for shift, root in sorted(shuffle_runs.items())
                },
                "base_model": str(Path(args.base_model).resolve()),
                "generation_seeds": generation_seeds,
                "rows": summary_rows,
                "class_rows": class_summary_rows,
                "notes": {
                    "encoding": "Mirrors generation-time untruncated 77-token chunking.",
                    "relative_l2": "Pairwise zero-padded hidden states; symmetric norm denominator.",
                    "mean_hidden": "Attention-mask mean over active token hidden states.",
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    plot_summary(summary_rows, output_dir / "conditioning_shift_summary.png")
    print(json.dumps(summary_rows, indent=2))


if __name__ == "__main__":
    main()
