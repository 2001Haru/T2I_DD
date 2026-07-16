"""Relate linear and logarithmic PCS measurements to downstream gains."""

import argparse
import csv
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr


DELTA_FIELDS = (
    "delta_class_vs_vae",
    "delta_class_vs_real",
    "delta_best_diffusion_vs_preserved",
    "delta_class_vs_empty",
    "delta_montage_vs_class",
)
LOG_EPSILON = 1e-12


def _read_csv(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Required diagnostic input was not found: {path}")
    with open(path, "r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _write_csv(path, rows):
    if not rows:
        raise ValueError(f"Cannot write an empty CSV: {path}")
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _correlation(rows, x_field, y_field):
    x = np.asarray([row[x_field] for row in rows], dtype=np.float64)
    y = np.asarray([row[y_field] for row in rows], dtype=np.float64)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return {
            "n": len(x), "pearson_r": None, "pearson_p": None,
            "spearman_rho": None, "spearman_p": None,
        }
    pearson = pearsonr(x, y)
    spearman = spearmanr(x, y)
    return {
        "n": len(x),
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "spearman_rho": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
    }


def _zero_threshold_test(rows, score_field, delta_field, score_name):
    score_positive = np.asarray([row[score_field] > 0 for row in rows])
    gain_positive = np.asarray([row[delta_field] > 0 for row in rows])
    true_positive = int(np.sum(score_positive & gain_positive))
    true_negative = int(np.sum(~score_positive & ~gain_positive))
    false_positive = int(np.sum(score_positive & ~gain_positive))
    false_negative = int(np.sum(~score_positive & gain_positive))
    positive_recall = true_positive / max(true_positive + false_negative, 1)
    negative_recall = true_negative / max(true_negative + false_positive, 1)
    return {
        "threshold": 0.0,
        "prediction": (
            f"use class-prompt diffusion when {score_name} > 0, "
            "otherwise VAE reconstruction"
        ),
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "accuracy": (true_positive + true_negative) / len(rows),
        "balanced_accuracy": (positive_recall + negative_recall) / 2,
    }


def _summarize(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "pcs_log_mean": float(np.mean(values)),
        "pcs_log_median": float(np.median(values)),
        "pcs_log_std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "pcs_log_positive_fraction": float(np.mean(values > 0.0)),
        "record_count": len(values),
    }


def _derive_log_pcs(spec, class_csv):
    raw_path = os.path.join(os.path.dirname(class_csv), "pcs_raw.csv")
    raw_rows = _read_csv(raw_path)
    class_groups = {}
    class_timestep_groups = {}
    for row in raw_rows:
        unconditional = max(float(row["unconditional_mse"]), LOG_EPSILON)
        conditional = max(float(row["conditional_mse"]), LOG_EPSILON)
        value = float(np.log(unconditional) - np.log(conditional))
        class_key = row["class_id"]
        timestep_key = (
            row["class_id"], int(row["timestep_index"]), int(row["timestep"])
        )
        class_groups.setdefault(class_key, []).append(value)
        class_timestep_groups.setdefault(timestep_key, []).append(value)

    class_rows = []
    for class_id, values in sorted(class_groups.items()):
        class_rows.append({"spec": spec, "class_id": class_id, **_summarize(values)})

    class_timestep_rows = []
    for (class_id, timestep_index, timestep), values in sorted(
        class_timestep_groups.items()
    ):
        class_timestep_rows.append({
            "spec": spec,
            "class_id": class_id,
            "timestep_index": timestep_index,
            "timestep": timestep,
            **_summarize(values),
        })
    return class_rows, class_timestep_rows


def _short_name(name):
    return name.split(",")[0].strip()


def _format_stat(value):
    return "n/a" if value is None else f"{value:.4f}"


def _scatter(axis, rows, x_field, y_field, title, xlabel):
    x = np.asarray([row[x_field] for row in rows], dtype=np.float64)
    y = np.asarray([row[y_field] for row in rows], dtype=np.float64)
    axis.axhline(0, color="black", linewidth=0.8, alpha=0.6)
    axis.axvline(0, color="black", linewidth=0.8, alpha=0.6)
    axis.scatter(x, y)
    for row, x_value, y_value in zip(rows, x, y):
        axis.annotate(
            _short_name(row["class_name"]), (x_value, y_value), fontsize=7,
            xytext=(3, 3), textcoords="offset points",
        )
    if len(rows) >= 2 and np.std(x) > 0:
        slope, intercept = np.polyfit(x, y, 1)
        line_x = np.linspace(np.min(x), np.max(x), 100)
        axis.plot(line_x, slope * line_x + intercept, linestyle="--", alpha=0.7)
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(y_field.replace("delta_", "").replace("_", " "))
    axis.grid(alpha=0.25)


def _timestep_diagnostics(pcs_inputs, accuracy_by_key, log_timestep_rows):
    log_by_key = {
        (row["spec"], row["class_id"], row["timestep_index"], row["timestep"]): row
        for row in log_timestep_rows
    }
    output = []
    for spec, class_csv in pcs_inputs:
        timestep_csv = os.path.join(
            os.path.dirname(class_csv), "pcs_per_class_timestep.csv"
        )
        grouped = {}
        for pcs_row in _read_csv(timestep_csv):
            accuracy_key = (spec, pcs_row["class_id"])
            if accuracy_key not in accuracy_by_key:
                raise KeyError(f"No downstream accuracy row for {spec}/{pcs_row['class_id']}")
            timestep_index = int(pcs_row["timestep_index"])
            timestep = int(pcs_row["timestep"])
            log_key = (spec, pcs_row["class_id"], timestep_index, timestep)
            accuracy = accuracy_by_key[accuracy_key]
            item = {
                "pcs_mean": float(pcs_row["pcs_mean"]),
                "pcs_log_mean": log_by_key[log_key]["pcs_log_mean"],
                "delta_class_vs_vae": (
                    float(accuracy["class_prompt_mean"])
                    - float(accuracy["vae_reconstruction_mean"])
                ),
            }
            grouped.setdefault((timestep_index, timestep), []).append(item)

        for (timestep_index, timestep), items in sorted(grouped.items()):
            linear = _correlation(items, "pcs_mean", "delta_class_vs_vae")
            logarithmic = _correlation(items, "pcs_log_mean", "delta_class_vs_vae")
            output.append({
                "spec": spec,
                "timestep_index": timestep_index,
                "timestep": timestep,
                **{f"pcs_{key}": value for key, value in linear.items()},
                **{f"pcs_log_{key}": value for key, value in logarithmic.items()},
            })
    return output


def _plot_relationship(groups, merged, score_field, z_field, score_name, output_path):
    specs = sorted(spec for spec in groups if spec != "combined")
    figure, axes = plt.subplots(2, 2, figsize=(16, 12))
    for axis, spec in zip(axes[0], specs[:2]):
        _scatter(
            axis, groups[spec], score_field, "delta_class_vs_vae",
            f"{spec}: {score_name} vs class-prompt gain over VAE", f"Mean {score_name}",
        )
    _scatter(
        axes[1, 0], merged, z_field, "delta_class_vs_vae",
        f"Combined: normalized {score_name} vs class-prompt gain",
        f"Within-subset {score_name} z-score",
    )
    _scatter(
        axes[1, 1], merged, z_field, "delta_best_diffusion_vs_preserved",
        f"Combined: {score_name} vs best refinement-family gain",
        f"Within-subset {score_name} z-score",
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accuracy_csv", required=True)
    parser.add_argument(
        "--pcs", nargs=2, action="append", metavar=("SPEC", "PCS_PER_CLASS_CSV"),
        required=True,
    )
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    accuracy_rows = _read_csv(args.accuracy_csv)
    accuracy_by_key = {(row["spec"], row["class_id"]): row for row in accuracy_rows}

    log_class_rows = []
    log_timestep_rows = []
    for spec, path in args.pcs:
        class_rows, timestep_rows = _derive_log_pcs(spec, path)
        log_class_rows.extend(class_rows)
        log_timestep_rows.extend(timestep_rows)
    log_by_key = {(row["spec"], row["class_id"]): row for row in log_class_rows}
    timestep_diagnostics = _timestep_diagnostics(
        args.pcs, accuracy_by_key, log_timestep_rows
    )

    merged = []
    seen_specs = []
    for spec, path in args.pcs:
        seen_specs.append(spec)
        for pcs_row in _read_csv(path):
            key = (spec, pcs_row["class_id"])
            if key not in accuracy_by_key:
                raise KeyError(f"No downstream accuracy row for {spec}/{pcs_row['class_id']}")
            accuracy = accuracy_by_key[key]
            log_row = log_by_key[key]
            values = {
                field: float(accuracy[field])
                for field in (
                    "real_representative_mean", "vae_reconstruction_mean",
                    "empty_prompt_mean", "generic_prompt_mean", "class_prompt_mean",
                    "montage_caption_mean",
                )
            }
            row = {
                "spec": spec,
                "local_label": int(accuracy["local_label"]),
                "class_id": pcs_row["class_id"],
                "class_name": accuracy["class_name"],
                "pcs_mean": float(pcs_row["pcs_mean"]),
                "pcs_median": float(pcs_row["pcs_median"]),
                "pcs_std": float(pcs_row["pcs_std"]),
                "pcs_positive_fraction": float(pcs_row["pcs_positive_fraction"]),
                "pcs_log_mean": log_row["pcs_log_mean"],
                "pcs_log_median": log_row["pcs_log_median"],
                "pcs_log_std": log_row["pcs_log_std"],
                "pcs_log_positive_fraction": log_row["pcs_log_positive_fraction"],
                **values,
            }
            row.update({
                "delta_class_vs_vae": values["class_prompt_mean"] - values["vae_reconstruction_mean"],
                "delta_class_vs_real": values["class_prompt_mean"] - values["real_representative_mean"],
                "delta_best_diffusion_vs_preserved": (
                    max(values["class_prompt_mean"], values["montage_caption_mean"])
                    - max(values["real_representative_mean"], values["vae_reconstruction_mean"])
                ),
                "delta_class_vs_empty": values["class_prompt_mean"] - values["empty_prompt_mean"],
                "delta_montage_vs_class": values["montage_caption_mean"] - values["class_prompt_mean"],
            })
            merged.append(row)

    for spec in set(seen_specs):
        spec_rows = [row for row in merged if row["spec"] == spec]
        for score_field, z_field in (
            ("pcs_mean", "pcs_z_within_spec"),
            ("pcs_log_mean", "pcs_log_z_within_spec"),
        ):
            values = np.asarray([row[score_field] for row in spec_rows])
            mean, std = float(np.mean(values)), float(np.std(values))
            for row in spec_rows:
                row[z_field] = (row[score_field] - mean) / max(std, LOG_EPSILON)

    merged.sort(key=lambda row: (row["spec"], row["local_label"]))
    groups = {
        spec: [row for row in merged if row["spec"] == spec]
        for spec in sorted(set(seen_specs))
    }
    groups["combined"] = merged
    diagnostics = {
        "primary_delta": "delta_class_vs_vae",
        "log_definition": "mean(log(unconditional_mse) - log(conditional_mse))",
        "log_epsilon": LOG_EPSILON,
        "metrics": {"pcs": {}, "pcs_log": {}},
        "exploratory_correlations_by_timestep": timestep_diagnostics,
    }
    for metric, score_field, z_field in (
        ("pcs", "pcs_mean", "pcs_z_within_spec"),
        ("pcs_log", "pcs_log_mean", "pcs_log_z_within_spec"),
    ):
        correlations = {}
        threshold_tests = {}
        for label, rows in groups.items():
            x_field = z_field if label == "combined" else score_field
            correlations[label] = {
                delta: _correlation(rows, x_field, delta) for delta in DELTA_FIELDS
            }
            threshold_tests[label] = _zero_threshold_test(
                rows, score_field, "delta_class_vs_vae", metric
            )
        diagnostics["metrics"][metric] = {
            "correlations": correlations,
            "zero_threshold": threshold_tests,
        }

    os.makedirs(args.output_dir, exist_ok=True)
    _write_csv(os.path.join(args.output_dir, "pcs_accuracy_merged.csv"), merged)
    _write_csv(os.path.join(args.output_dir, "pcs_log_per_class.csv"), log_class_rows)
    _write_csv(
        os.path.join(args.output_dir, "pcs_log_per_class_timestep.csv"), log_timestep_rows
    )
    _write_csv(
        os.path.join(args.output_dir, "pcs_timestep_accuracy_correlations.csv"),
        timestep_diagnostics,
    )
    with open(
        os.path.join(args.output_dir, "pcs_accuracy_diagnostics.json"),
        "w", encoding="utf-8",
    ) as file:
        json.dump(diagnostics, file, ensure_ascii=False, indent=2)
        file.write("\n")

    _plot_relationship(
        groups, merged, "pcs_mean", "pcs_z_within_spec", "PCS",
        os.path.join(args.output_dir, "pcs_accuracy_relationship.png"),
    )
    _plot_relationship(
        groups, merged, "pcs_log_mean", "pcs_log_z_within_spec", "PCS_log",
        os.path.join(args.output_dir, "pcs_log_accuracy_relationship.png"),
    )

    specs = sorted(spec for spec in groups if spec != "combined")
    figure, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    for axis, metric_field, title in (
        (axes[0], "pcs_spearman_rho", "Linear PCS"),
        (axes[1], "pcs_log_spearman_rho", "Logarithmic PCS"),
    ):
        for spec in specs:
            spec_rows = [row for row in timestep_diagnostics if row["spec"] == spec]
            axis.plot(
                [row["timestep"] for row in spec_rows],
                [row[metric_field] for row in spec_rows],
                marker="o", label=spec,
            )
        axis.axhline(0, color="black", linewidth=0.8, alpha=0.6)
        axis.set_xlabel("SDXL training timestep")
        axis.set_title(title)
        axis.grid(alpha=0.25)
        axis.legend()
    axes[0].set_ylabel("Spearman rho vs class-prompt gain over VAE")
    figure.suptitle("Exploratory PCS correlation by noise level")
    figure.tight_layout()
    figure.savefig(os.path.join(args.output_dir, "pcs_timestep_correlations.png"), dpi=180)
    plt.close(figure)

    print("Primary correlation: score vs class-prompt gain over VAE")
    for label in groups:
        linear = diagnostics["metrics"]["pcs"]["correlations"][label][
            "delta_class_vs_vae"
        ]["spearman_rho"]
        logarithmic = diagnostics["metrics"]["pcs_log"]["correlations"][label][
            "delta_class_vs_vae"
        ]["spearman_rho"]
        log_threshold = diagnostics["metrics"]["pcs_log"]["zero_threshold"][label]
        print(
            f"  {label}: PCS rho={_format_stat(linear)}, "
            f"PCS_log rho={_format_stat(logarithmic)}, "
            f"PCS_log>0 accuracy={log_threshold['accuracy']:.4f}, "
            f"balanced_accuracy={log_threshold['balanced_accuracy']:.4f}"
        )
    print(f"Saved linear/log PCS diagnostics to: {args.output_dir}")


if __name__ == "__main__":
    main()
