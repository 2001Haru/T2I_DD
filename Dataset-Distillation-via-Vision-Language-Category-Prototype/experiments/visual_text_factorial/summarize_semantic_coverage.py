import argparse
import csv
import json
import statistics
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Join SD conditioning displacement with DINO coverage diagnostics"
    )
    parser.add_argument("--text-dir", required=True)
    parser.add_argument("--dino-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def read_csv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def numeric(row, field):
    return float(row[field])


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_overview(text_rows, joined_rows, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as pyplot

    shifts = [int(row["shuffle_shift"]) for row in text_rows]
    relative_l2 = [numeric(row, "symmetric_relative_l2_mean") for row in text_rows]
    figure, axes = pyplot.subplots(2, 2, figsize=(13, 9))
    axes[0, 0].plot(shifts, relative_l2, marker="o")
    axes[0, 0].set_title("SD 1.5 conditioning displacement")
    axes[0, 0].set_xlabel("Shuffle shift")
    axes[0, 0].set_ylabel("Symmetric relative L2")

    prototype = [row for row in joined_rows if row["visual_mode"] == "prototype"]
    no_visual = [row for row in joined_rows if row["visual_mode"] == "no_visual"]
    for rows, label, marker in (
        (prototype, "prototype", "o"),
        (no_visual, "no visual", "x"),
    ):
        axes[0, 1].scatter(
            [int(row["shuffle_shift"]) for row in rows],
            [numeric(row, "coverage_distance_improvement") for row in rows],
            label=label,
            marker=marker,
        )
        axes[1, 0].scatter(
            [int(row["shuffle_shift"]) for row in rows],
            [numeric(row, "diversity_change") for row in rows],
            label=label,
            marker=marker,
        )
    axes[0, 1].axhline(0, color="black", linewidth=0.8)
    axes[0, 1].set_title("Shuffled minus correct: coverage")
    axes[0, 1].set_xlabel("Shuffle shift")
    axes[0, 1].set_ylabel("NN-distance improvement (positive is better)")
    axes[0, 1].legend()
    axes[1, 0].axhline(0, color="black", linewidth=0.8)
    axes[1, 0].set_title("Shuffled minus correct: diversity")
    axes[1, 0].set_xlabel("Shuffle shift")
    axes[1, 0].set_ylabel("Pairwise-distance change")
    axes[1, 0].legend()

    for rows, label, marker in (
        (prototype, "prototype", "o"),
        (no_visual, "no visual", "x"),
    ):
        axes[1, 1].scatter(
            [numeric(row, "conditioning_relative_l2") for row in rows],
            [numeric(row, "coverage_distance_improvement") for row in rows],
            label=label,
            marker=marker,
        )
    axes[1, 1].axhline(0, color="black", linewidth=0.8)
    axes[1, 1].set_title("Conditioning displacement versus coverage")
    axes[1, 1].set_xlabel("Conditioning relative L2")
    axes[1, 1].set_ylabel("NN-distance improvement")
    axes[1, 1].legend()
    for axis in axes.flat:
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    pyplot.close(figure)


def plot_class_relation(rows, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as pyplot

    figure, axes = pyplot.subplots(1, 2, figsize=(13, 5))
    for visual_mode, marker in (("prototype", "o"), ("no_visual", "x")):
        selected = [row for row in rows if row["visual_mode"] == visual_mode]
        axes[0].scatter(
            [numeric(row, "conditioning_relative_l2") for row in selected],
            [numeric(row, "coverage_distance_improvement") for row in selected],
            label=visual_mode,
            marker=marker,
            alpha=0.7,
        )
        axes[1].scatter(
            [numeric(row, "diversity_change") for row in selected],
            [numeric(row, "coverage_distance_improvement") for row in selected],
            label=visual_mode,
            marker=marker,
            alpha=0.7,
        )
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_xlabel("Class-level conditioning relative L2")
    axes[0].set_ylabel("Class-level coverage improvement")
    axes[0].set_title("Prompt displacement versus coverage")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].set_xlabel("Class-level diversity change")
    axes[1].set_ylabel("Class-level coverage improvement")
    axes[1].set_title("Diversity versus real-manifold coverage")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    pyplot.close(figure)


def aggregate(rows):
    fields = (
        "coverage_distance_improvement",
        "fidelity_distance_improvement",
        "coverage_fraction_change",
        "precision_fraction_change",
        "diversity_change",
        "centroid_distance_improvement",
    )
    result = {}
    for visual_mode in ("no_visual", "prototype"):
        selected = [row for row in rows if row["visual_mode"] == visual_mode]
        result[visual_mode] = {}
        for field in fields:
            values = [numeric(row, field) for row in selected]
            result[visual_mode][field] = {
                "mean_over_shifts_and_generation_seeds": statistics.fmean(values),
                "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
                "values": values,
            }
    return result


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    text_rows = read_csv(Path(args.text_dir) / "conditioning_shift_summary.csv")
    text_class_rows = read_csv(
        Path(args.text_dir) / "conditioning_class_summary.csv"
    )
    dino_rows = read_csv(Path(args.dino_dir) / "dino_shuffled_minus_correct.csv")
    dino_class_rows = read_csv(
        Path(args.dino_dir) / "dino_shuffled_minus_correct_per_class.csv"
    )
    text_by_shift = {int(row["shuffle_shift"]): row for row in text_rows}
    text_by_shift_class = {
        (int(row["shuffle_shift"]), row["synset"]): row for row in text_class_rows
    }

    joined = []
    for row in dino_rows:
        shift = int(float(row["shuffle_shift"]))
        text = text_by_shift[shift]
        joined.append(
            {
                **row,
                "conditioning_relative_l2": text["symmetric_relative_l2_mean"],
                "conditioning_mean_hidden_cosine": text["mean_hidden_cosine_mean"],
                "conditioning_token_jaccard": text["token_jaccard_mean"],
            }
        )
    write_csv(output_dir / "conditioning_and_coverage.csv", joined)
    joined_class = []
    for row in dino_class_rows:
        shift = int(float(row["shuffle_shift"]))
        text = text_by_shift_class[(shift, row["synset"])]
        joined_class.append(
            {
                **row,
                "conditioning_relative_l2": text["symmetric_relative_l2_mean"],
                "conditioning_mean_hidden_cosine": text[
                    "mean_hidden_cosine_mean"
                ],
                "conditioning_token_jaccard": text["token_jaccard_mean"],
            }
        )
    write_csv(
        output_dir / "conditioning_and_coverage_per_class.csv",
        joined_class,
    )
    summary = aggregate(joined)
    (output_dir / "semantic_coverage_summary.json").write_text(
        json.dumps(
            {
                "aggregate": summary,
                "interpretation_rule": (
                    "The coverage-expansion hypothesis requires positive prototype "
                    "coverage/diversity changes without a systematic fidelity or "
                    "precision decrease. No-visual results serve as an interaction control."
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    plot_overview(
        text_rows,
        joined,
        output_dir / "semantic_coverage_diagnostic.png",
    )
    plot_class_relation(
        joined_class,
        output_dir / "semantic_coverage_per_class.png",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
