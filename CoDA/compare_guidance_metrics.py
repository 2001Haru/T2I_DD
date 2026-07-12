"""Create a combined comparison plot from multiple guidance metric summaries."""

import argparse
import csv
import json
import os
import statistics


def _backfill_conflict_projection(summary, summary_path):
    if all("conflict_projection_ratio_median" in item for item in summary["per_step"]):
        return

    raw_path = os.path.join(os.path.dirname(summary_path), "guidance_metrics_raw.csv")
    by_step = {}
    all_values = []
    with open(raw_path, "r", encoding="utf-8", newline="") as file:
        for record in csv.DictReader(file):
            cosine = float(record["cosine_similarity"])
            q_value = max(float(record["q_text_over_image"]), 1e-12)
            value = max(0.0, -cosine) / q_value
            by_step.setdefault(int(record["step_index"]), []).append(value)
            all_values.append(value)

    for item in summary["per_step"]:
        values = by_step[item["step_index"]]
        item["conflict_projection_ratio_median"] = statistics.median(values)
    summary["overall"]["conflict_projection_ratio_mean"] = statistics.fmean(all_values)
    summary["overall"]["conflict_projection_ratio_median"] = statistics.median(all_values)


def main():
    parser = argparse.ArgumentParser(description="Compare CoDA guidance diagnostics.")
    parser.add_argument("--input", action="append", required=True, metavar="LABEL=SUMMARY_JSON")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summaries = []
    for item in args.input:
        label, path = item.split("=", 1)
        with open(path, "r", encoding="utf-8") as file:
            summary = json.load(file)
        _backfill_conflict_projection(summary, path)
        summaries.append((label, summary))

    os.makedirs(args.output_dir, exist_ok=True)
    comparison = {}
    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
    for label, summary in summaries:
        per_step = summary["per_step"]
        x = [item["step_index"] for item in per_step]
        cosine = [item["cosine_mean"] for item in per_step]
        q_values = [item["q_median"] for item in per_step]
        conflict = [item["conflict_projection_ratio_median"] for item in per_step]
        axes[0].plot(x, cosine, marker="o", label=label)
        axes[1].plot(x, q_values, marker="o", label=label)
        axes[2].plot(x, conflict, marker="o", label=label)
        comparison[label] = summary["overall"]

    axes[0].axhline(0.0, color="black", linewidth=1, linestyle="--")
    axes[0].set_ylabel("Mean cos(g_text, g_img)")
    axes[0].set_title("Text-image guidance interaction")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].axhline(1.0, color="black", linewidth=1, linestyle="--")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Median q")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    axes[2].set_xlabel("Denoising step index (early to late)")
    axes[2].set_ylabel("Median kappa")
    axes[2].grid(alpha=0.25)
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "guidance_comparison.png"), dpi=180)
    plt.close(fig)

    with open(os.path.join(args.output_dir, "guidance_comparison_summary.json"), "w", encoding="utf-8") as file:
        json.dump(comparison, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(f"Saved guidance comparison to: {args.output_dir}")


if __name__ == "__main__":
    main()
