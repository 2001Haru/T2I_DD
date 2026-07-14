"""Relate class-level guidance conflict to downstream ImageB accuracy."""

import argparse
from collections import defaultdict
import csv
import json
import os

import numpy as np
from scipy.stats import pearsonr, spearmanr


METHODS = ("coda_baseline", "v1_focused_alpha_0", "v1_focused_alpha_0p5")
METHOD_LABELS = {
    "coda_baseline": "baseline",
    "v1_focused_alpha_0": "focused_alpha_0",
    "v1_focused_alpha_0p5": "focused_alpha_0p5",
}
PHASES = ("overall", "early", "middle", "late")


def _float(record, preferred, fallback):
    value = record.get(preferred)
    if value not in (None, ""):
        return float(value)
    return float(record[fallback])


def _load_guidance(run_dir, generation_seeds):
    records = []
    for generation_seed in generation_seeds:
        for method in METHODS:
            path = os.path.join(
                run_dir,
                f"seed_{generation_seed}",
                f"generated_images_{method}",
                "guidance_metrics",
                "guidance_metrics_raw.csv",
            )
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Missing raw guidance metrics: {path}")
            with open(path, "r", encoding="utf-8", newline="") as file:
                for record in csv.DictReader(file):
                    records.append({
                        "generation_seed": generation_seed,
                        "method": method,
                        "class_id": record["class_id"],
                        "class_name": record["class_name"],
                        "step_index": int(record["step_index"]),
                        "cosine": _float(
                            record, "pre_projection_cosine_similarity", "cosine_similarity"
                        ),
                        "q": _float(
                            record, "pre_projection_q_text_over_image", "q_text_over_image"
                        ),
                        "kappa": _float(
                            record, "pre_projection_conflict_ratio", "conflict_projection_ratio"
                        ),
                    })
    if not records:
        raise ValueError("No guidance records were loaded.")
    return records


def _load_accuracy(trained_root, generation_seeds):
    values = defaultdict(list)
    class_info = {}
    for generation_seed in generation_seeds:
        for method in METHODS:
            path = os.path.join(
                trained_root,
                f"seed_{generation_seed}",
                f"{method}-resnet_ap",
                "per_class_accuracy_all_seeds.json",
            )
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Missing per-class classifier result: {path}")
            with open(path, "r", encoding="utf-8") as file:
                payload = json.load(file)
            for run in payload["runs"]:
                for row in run["classes"]:
                    key = (method, row["class_id"])
                    values[key].append(float(row["accuracy"]))
                    class_info[row["class_id"]] = {
                        "local_label": int(row["local_label"]),
                        "class_id": row["class_id"],
                        "class_name": row["class_name"],
                    }
    means = {key: float(np.mean(items)) for key, items in values.items()}
    return means, class_info


def _phase(step_index, step_count):
    phase_index = min(2, (3 * step_index) // step_count)
    return ("early", "middle", "late")[phase_index]


def _metric_summary(records):
    max_step = max(record["step_index"] for record in records)
    step_count = max_step + 1
    buckets = defaultdict(list)
    curves = defaultdict(lambda: defaultdict(list))
    for record in records:
        class_method = (record["class_id"], record["method"])
        buckets[(class_method, "overall")].append(record)
        buckets[(class_method, _phase(record["step_index"], step_count))].append(record)
        curves[class_method][record["step_index"]].append(record)

    summary = {}
    for (class_method, phase), items in buckets.items():
        cosine = np.asarray([item["cosine"] for item in items])
        q_values = np.asarray([item["q"] for item in items])
        kappa = np.asarray([item["kappa"] for item in items])
        summary[(class_method[0], class_method[1], phase)] = {
            "cosine_mean": float(np.mean(cosine)),
            "cosine_negative_fraction": float(np.mean(cosine < 0.0)),
            "q_median": float(np.median(q_values)),
            "kappa_mean": float(np.mean(kappa)),
            "kappa_median": float(np.median(kappa)),
        }

    curve_summary = {}
    for class_method, by_step in curves.items():
        curve_summary[class_method] = {
            step: {
                "cosine_mean": float(np.mean([item["cosine"] for item in items])),
                "kappa_mean": float(np.mean([item["kappa"] for item in items])),
            }
            for step, items in sorted(by_step.items())
        }
    return summary, curve_summary, step_count


def _correlation(name, x_name, y_name, x, y):
    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)
    result = {"name": name, "x": x_name, "y": y_name, "n": len(x)}
    if len(x) < 3 or np.std(x_array) == 0.0 or np.std(y_array) == 0.0:
        result.update({"pearson_r": None, "pearson_p": None, "spearman_r": None, "spearman_p": None})
        return result
    pearson = pearsonr(x_array, y_array)
    spearman = spearmanr(x_array, y_array)
    result.update({
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "spearman_r": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
    })
    return result


def _plot_scatter(ax, rows, x_key, y_key, title, x_label, y_label):
    x = np.asarray([row[x_key] for row in rows])
    y = np.asarray([row[y_key] for row in rows])
    ax.scatter(x, y, s=45)
    if np.std(x) > 0.0:
        design = np.column_stack([x, np.ones_like(x)])
        slope, intercept = np.linalg.lstsq(design, y, rcond=None)[0]
        line_x = np.linspace(float(x.min()), float(x.max()), 100)
        ax.plot(line_x, slope * line_x + intercept, linestyle="--", alpha=0.65)
    for row, x_value, y_value in zip(rows, x, y):
        ax.annotate(row["class_name"].split(",")[0], (x_value, y_value), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5) if "gain" in y_key else None
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.25)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--trained_root", required=True)
    parser.add_argument("--generation_seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = args.output_dir or os.path.join(args.run_dir, "class_guidance_diagnostics")
    os.makedirs(output_dir, exist_ok=True)
    guidance_records = _load_guidance(args.run_dir, args.generation_seeds)
    accuracy, class_info = _load_accuracy(args.trained_root, args.generation_seeds)
    metrics, curves, step_count = _metric_summary(guidance_records)
    class_ids = [
        item["class_id"]
        for item in sorted(class_info.values(), key=lambda item: item["local_label"])
    ]

    rows = []
    for class_id in class_ids:
        info = class_info[class_id]
        row = dict(info)
        for method in METHODS:
            label = METHOD_LABELS[method]
            row[f"accuracy_{label}"] = accuracy[(method, class_id)]
            for phase in PHASES:
                for metric, value in metrics[(class_id, method, phase)].items():
                    row[f"{label}_{phase}_{metric}"] = value
        row["caption_gain"] = row["accuracy_focused_alpha_0"] - row["accuracy_baseline"]
        row["projection_gain"] = row["accuracy_focused_alpha_0p5"] - row["accuracy_focused_alpha_0"]
        row["projection_vs_baseline"] = row["accuracy_focused_alpha_0p5"] - row["accuracy_baseline"]
        for phase in PHASES:
            row[f"caption_delta_{phase}_kappa_mean"] = (
                row[f"focused_alpha_0_{phase}_kappa_mean"]
                - row[f"baseline_{phase}_kappa_mean"]
            )
            row[f"caption_delta_{phase}_cosine_mean"] = (
                row[f"focused_alpha_0_{phase}_cosine_mean"]
                - row[f"baseline_{phase}_cosine_mean"]
            )
        rows.append(row)

    csv_path = os.path.join(output_dir, "per_class_guidance_accuracy.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    correlation_specs = []
    for phase in PHASES:
        correlation_specs.extend([
            (
                f"baseline_accuracy_vs_{phase}_kappa",
                f"baseline_{phase}_kappa_mean", "accuracy_baseline",
            ),
            (
                f"focused_accuracy_vs_{phase}_kappa",
                f"focused_alpha_0_{phase}_kappa_mean", "accuracy_focused_alpha_0",
            ),
            (
                f"projection_gain_vs_{phase}_kappa",
                f"focused_alpha_0_{phase}_kappa_mean", "projection_gain",
            ),
            (
                f"caption_gain_vs_{phase}_kappa_delta",
                f"caption_delta_{phase}_kappa_mean", "caption_gain",
            ),
        ])
    correlations = [
        _correlation(
            name, x_key, y_key,
            [row[x_key] for row in rows], [row[y_key] for row in rows],
        )
        for name, x_key, y_key in correlation_specs
    ]
    with open(os.path.join(output_dir, "correlations.json"), "w", encoding="utf-8") as file:
        json.dump({"step_count": step_count, "correlations": correlations}, file, ensure_ascii=False, indent=2)
        file.write("\n")

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    _plot_scatter(
        axes[0, 0], rows, "baseline_early_kappa_mean", "accuracy_baseline",
        "Baseline accuracy vs early conflict", "Baseline early mean kappa", "Baseline accuracy",
    )
    _plot_scatter(
        axes[0, 1], rows, "focused_alpha_0_early_kappa_mean", "accuracy_focused_alpha_0",
        "Focused accuracy vs early conflict", "Focused early mean kappa", "Focused accuracy",
    )
    _plot_scatter(
        axes[1, 0], rows, "focused_alpha_0_early_kappa_mean", "projection_gain",
        "Projection gain vs original conflict", "Focused early mean kappa", "Alpha 0.5 - alpha 0 accuracy",
    )
    _plot_scatter(
        axes[1, 1], rows, "caption_delta_early_kappa_mean", "caption_gain",
        "Caption gain vs caption-induced conflict", "Focused - baseline early kappa", "Focused - baseline accuracy",
    )
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "guidance_accuracy_scatter.png"), dpi=180)
    plt.close(fig)

    for metric, ylabel, filename in (
        ("cosine_mean", "Mean pre-projection cosine", "per_class_cosine_curves.png"),
        ("kappa_mean", "Mean pre-projection kappa", "per_class_kappa_curves.png"),
    ):
        fig, axes = plt.subplots(2, 5, figsize=(18, 7), sharex=True, sharey=True)
        for axis, class_id in zip(axes.flat, class_ids):
            for method in ("coda_baseline", "v1_focused_alpha_0"):
                curve = curves[(class_id, method)]
                steps = sorted(curve)
                axis.plot(
                    steps, [curve[step][metric] for step in steps],
                    marker="o", markersize=2.5, label=METHOD_LABELS[method],
                )
            axis.axhline(0.0, color="black", linewidth=0.7, linestyle="--")
            axis.set_title(class_info[class_id]["class_name"].split(",")[0], fontsize=9)
            axis.grid(alpha=0.2)
        axes[0, 0].legend(fontsize=8)
        fig.supxlabel("Denoising step index (early to late)")
        fig.supylabel(ylabel)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, filename), dpi=180)
        plt.close(fig)

    print(f"Saved class-level guidance diagnostics to: {output_dir}")


if __name__ == "__main__":
    main()
