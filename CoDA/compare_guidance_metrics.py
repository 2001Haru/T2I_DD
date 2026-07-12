"""Create a combined comparison plot from multiple guidance metric summaries."""

import argparse
import json
import os


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
            summaries.append((label, json.load(file)))

    os.makedirs(args.output_dir, exist_ok=True)
    comparison = {}
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for label, summary in summaries:
        per_step = summary["per_step"]
        x = [item["step_index"] for item in per_step]
        cosine = [item["cosine_mean"] for item in per_step]
        q_values = [item["q_median"] for item in per_step]
        axes[0].plot(x, cosine, marker="o", label=label)
        axes[1].plot(x, q_values, marker="o", label=label)
        comparison[label] = summary["overall"]

    axes[0].axhline(0.0, color="black", linewidth=1, linestyle="--")
    axes[0].set_ylabel("Mean cos(g_text, g_img)")
    axes[0].set_title("Text-image guidance interaction")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].axhline(1.0, color="black", linewidth=1, linestyle="--")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Denoising step index (early to late)")
    axes[1].set_ylabel("Median q")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "guidance_comparison.png"), dpi=180)
    plt.close(fig)

    with open(os.path.join(args.output_dir, "guidance_comparison_summary.json"), "w", encoding="utf-8") as file:
        json.dump(comparison, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(f"Saved guidance comparison to: {args.output_dir}")


if __name__ == "__main__":
    main()
