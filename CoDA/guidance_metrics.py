"""Raw recording and visualization for text/image guidance interactions."""

import csv
import json
import os


RAW_FIELDS = [
    "method",
    "gpu_id",
    "class_id",
    "class_name",
    "sample_index",
    "image_seed",
    "step_index",
    "timestep",
    "sigma",
    "text_norm_l2",
    "image_norm_l2",
    "q_text_over_image",
    "cosine_similarity",
]


def write_worker_metrics(output_dir, gpu_id, records):
    metrics_dir = os.path.join(output_dir, "guidance_metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    path = os.path.join(metrics_dir, f"raw_gpu{gpu_id}.csv")
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=RAW_FIELDS)
        writer.writeheader()
        writer.writerows(records)
    return path


def _load_worker_records(metrics_dir, num_gpus):
    records = []
    for gpu_id in range(num_gpus):
        path = os.path.join(metrics_dir, f"raw_gpu{gpu_id}.csv")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing guidance metrics from GPU {gpu_id}: {path}")
        with open(path, "r", encoding="utf-8", newline="") as file:
            records.extend(csv.DictReader(file))
    return records


def _float(record, key):
    return float(record[key])


def finalize_guidance_metrics(output_dir, num_gpus, metadata):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    metrics_dir = os.path.join(output_dir, "guidance_metrics")
    records = _load_worker_records(metrics_dir, num_gpus)
    if not records:
        raise ValueError("No active CoDA guidance measurements were recorded.")

    raw_path = os.path.join(metrics_dir, "guidance_metrics_raw.csv")
    with open(raw_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=RAW_FIELDS)
        writer.writeheader()
        writer.writerows(records)

    cosine = np.asarray([_float(record, "cosine_similarity") for record in records])
    q_values = np.asarray([_float(record, "q_text_over_image") for record in records])
    text_norms = np.asarray([_float(record, "text_norm_l2") for record in records])
    image_norms = np.asarray([_float(record, "image_norm_l2") for record in records])
    step_indices = sorted({int(record["step_index"]) for record in records})

    per_step = []
    for step_index in step_indices:
        step_records = [record for record in records if int(record["step_index"]) == step_index]
        step_cosine = np.asarray([_float(record, "cosine_similarity") for record in step_records])
        step_q = np.asarray([_float(record, "q_text_over_image") for record in step_records])
        per_step.append({
            "step_index": step_index,
            "timestep": int(step_records[0]["timestep"]),
            "count": len(step_records),
            "cosine_mean": float(np.mean(step_cosine)),
            "cosine_std": float(np.std(step_cosine)),
            "cosine_negative_fraction": float(np.mean(step_cosine < 0.0)),
            "q_mean": float(np.mean(step_q)),
            "q_median": float(np.median(step_q)),
            "q_p25": float(np.percentile(step_q, 25)),
            "q_p75": float(np.percentile(step_q, 75)),
        })

    summary = {
        "format_version": 1,
        "definition": {
            "g_text": "epsilon_conditional - epsilon_unconditional",
            "g_img": "CoDA delta_epsilon_text",
            "q_t": "L2(g_text) / L2(g_img)",
            "space": "SDXL noise-prediction space before CFG scaling",
        },
        "metadata": metadata,
        "num_records": len(records),
        "num_samples": len({(record["class_id"], record["sample_index"]) for record in records}),
        "overall": {
            "cosine_mean": float(np.mean(cosine)),
            "cosine_median": float(np.median(cosine)),
            "cosine_std": float(np.std(cosine)),
            "cosine_negative_fraction": float(np.mean(cosine < 0.0)),
            "q_mean": float(np.mean(q_values)),
            "q_median": float(np.median(q_values)),
            "q_p25": float(np.percentile(q_values, 25)),
            "q_p75": float(np.percentile(q_values, 75)),
            "text_norm_mean": float(np.mean(text_norms)),
            "image_norm_mean": float(np.mean(image_norms)),
        },
        "per_step": per_step,
    }
    with open(os.path.join(metrics_dir, "guidance_metrics_summary.json"), "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
        file.write("\n")

    x = np.asarray([item["step_index"] for item in per_step])
    cosine_mean = np.asarray([item["cosine_mean"] for item in per_step])
    cosine_std = np.asarray([item["cosine_std"] for item in per_step])
    q_median = np.asarray([item["q_median"] for item in per_step])
    q_p25 = np.asarray([item["q_p25"] for item in per_step])
    q_p75 = np.asarray([item["q_p75"] for item in per_step])

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    axes[0].plot(x, cosine_mean, marker="o", label="mean cosine")
    axes[0].fill_between(x, cosine_mean - cosine_std, cosine_mean + cosine_std, alpha=0.2)
    axes[0].axhline(0.0, color="black", linewidth=1, linestyle="--")
    axes[0].set_ylabel("cos(g_text, g_img)")
    axes[0].set_title(metadata.get("method", "Guidance interaction"))
    axes[0].grid(alpha=0.25)

    axes[1].plot(x, q_median, marker="o", color="tab:orange", label="median q")
    axes[1].fill_between(x, q_p25, q_p75, color="tab:orange", alpha=0.2, label="IQR")
    axes[1].axhline(1.0, color="black", linewidth=1, linestyle="--")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Denoising step index (early to late)")
    axes[1].set_ylabel("q = ||g_text|| / ||g_img||")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(metrics_dir, "guidance_over_steps.png"), dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].hist(cosine, bins=40, color="tab:blue", alpha=0.85)
    axes[0].axvline(0.0, color="black", linewidth=1, linestyle="--")
    axes[0].set_xlabel("cos(g_text, g_img)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Direction conflict distribution")
    positive_q = q_values[q_values > 0.0]
    if positive_q.size == 0:
        raise ValueError("All recorded q values are non-positive; log-scale visualization is undefined.")
    q_min = float(positive_q.min())
    q_max = float(positive_q.max())
    if q_min == q_max:
        q_min *= 0.9
        q_max *= 1.1
    axes[1].hist(positive_q, bins=np.logspace(np.log10(q_min), np.log10(q_max), 40),
                 color="tab:orange", alpha=0.85)
    axes[1].set_xscale("log")
    axes[1].axvline(1.0, color="black", linewidth=1, linestyle="--")
    axes[1].set_xlabel("q = ||g_text|| / ||g_img||")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Relative guidance strength")
    fig.tight_layout()
    fig.savefig(os.path.join(metrics_dir, "guidance_distributions.png"), dpi=180)
    plt.close(fig)

    print(f"Saved raw guidance metrics and plots to: {metrics_dir}")
    return summary
