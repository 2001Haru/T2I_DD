"""Relate PCS measurements to downstream refinement gains."""

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


def _read_csv(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Required diagnostic input was not found: {path}")
    with open(path, "r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _write_csv(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _correlation(rows, x_field, y_field):
    x = np.asarray([row[x_field] for row in rows], dtype=np.float64)
    y = np.asarray([row[y_field] for row in rows], dtype=np.float64)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return {"n": len(x), "pearson_r": None, "pearson_p": None,
                "spearman_rho": None, "spearman_p": None}
    pearson = pearsonr(x, y)
    spearman = spearmanr(x, y)
    return {
        "n": len(x),
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "spearman_rho": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
    }


def _zero_threshold_test(rows, delta_field):
    pcs_positive = np.asarray([row["pcs_mean"] > 0 for row in rows])
    gain_positive = np.asarray([row[delta_field] > 0 for row in rows])
    true_positive = int(np.sum(pcs_positive & gain_positive))
    true_negative = int(np.sum(~pcs_positive & ~gain_positive))
    false_positive = int(np.sum(pcs_positive & ~gain_positive))
    false_negative = int(np.sum(~pcs_positive & gain_positive))
    positive_recall = true_positive / max(true_positive + false_negative, 1)
    negative_recall = true_negative / max(true_negative + false_positive, 1)
    return {
        "threshold": 0.0,
        "prediction": "use class-prompt diffusion when PCS > 0, otherwise VAE reconstruction",
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "accuracy": (true_positive + true_negative) / len(rows),
        "balanced_accuracy": (positive_recall + negative_recall) / 2,
    }


def _short_name(name):
    return name.split(",")[0].strip()


def _scatter(axis, rows, x_field, y_field, title, xlabel):
    x = np.asarray([row[x_field] for row in rows], dtype=np.float64)
    y = np.asarray([row[y_field] for row in rows], dtype=np.float64)
    axis.axhline(0, color="black", linewidth=0.8, alpha=0.6)
    axis.axvline(0, color="black", linewidth=0.8, alpha=0.6)
    axis.scatter(x, y)
    for row, x_value, y_value in zip(rows, x, y):
        axis.annotate(_short_name(row["class_name"]), (x_value, y_value), fontsize=7,
                      xytext=(3, 3), textcoords="offset points")
    if len(rows) >= 2 and np.std(x) > 0:
        slope, intercept = np.polyfit(x, y, 1)
        line_x = np.linspace(np.min(x), np.max(x), 100)
        axis.plot(line_x, slope * line_x + intercept, linestyle="--", alpha=0.7)
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(y_field.replace("delta_", "").replace("_", " "))
    axis.grid(alpha=0.25)


def _timestep_diagnostics(pcs_inputs, accuracy_by_key):
    rows = []
    for spec, class_csv in pcs_inputs:
        timestep_csv = os.path.join(
            os.path.dirname(class_csv), "pcs_per_class_timestep.csv"
        )
        timestep_rows = _read_csv(timestep_csv)
        grouped = {}
        for pcs_row in timestep_rows:
            key = (spec, pcs_row["class_id"])
            if key not in accuracy_by_key:
                raise KeyError(f"No downstream accuracy row for {spec}/{pcs_row['class_id']}")
            accuracy = accuracy_by_key[key]
            item = {
                "spec": spec,
                "timestep_index": int(pcs_row["timestep_index"]),
                "timestep": int(pcs_row["timestep"]),
                "pcs_mean": float(pcs_row["pcs_mean"]),
                "delta_class_vs_vae": (
                    float(accuracy["class_prompt_mean"])
                    - float(accuracy["vae_reconstruction_mean"])
                ),
            }
            grouped.setdefault((item["timestep_index"], item["timestep"]), []).append(item)

        for (timestep_index, timestep), items in sorted(grouped.items()):
            correlation = _correlation(items, "pcs_mean", "delta_class_vs_vae")
            rows.append({
                "spec": spec,
                "timestep_index": timestep_index,
                "timestep": timestep,
                **correlation,
            })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accuracy_csv", required=True)
    parser.add_argument(
        "--pcs", nargs=2, action="append", metavar=("SPEC", "PCS_PER_CLASS_CSV"), required=True
    )
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    accuracy_rows = _read_csv(args.accuracy_csv)
    accuracy_by_key = {(row["spec"], row["class_id"]): row for row in accuracy_rows}
    timestep_diagnostics = _timestep_diagnostics(args.pcs, accuracy_by_key)
    merged = []
    seen_specs = []
    for spec, path in args.pcs:
        seen_specs.append(spec)
        for pcs_row in _read_csv(path):
            key = (spec, pcs_row["class_id"])
            if key not in accuracy_by_key:
                raise KeyError(f"No downstream accuracy row for {spec}/{pcs_row['class_id']}")
            accuracy = accuracy_by_key[key]
            values = {
                field: float(accuracy[field])
                for field in (
                    "real_representative_mean", "vae_reconstruction_mean", "empty_prompt_mean",
                    "generic_prompt_mean", "class_prompt_mean", "montage_caption_mean",
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
        values = np.asarray([row["pcs_mean"] for row in spec_rows])
        mean, std = float(np.mean(values)), float(np.std(values))
        for row in spec_rows:
            row["pcs_z_within_spec"] = (row["pcs_mean"] - mean) / max(std, 1e-12)

    merged.sort(key=lambda row: (row["spec"], row["local_label"]))
    diagnostics = {"primary_delta": "delta_class_vs_vae", "correlations": {}, "zero_threshold": {}}
    groups = {spec: [row for row in merged if row["spec"] == spec] for spec in sorted(set(seen_specs))}
    groups["combined"] = merged
    for label, rows in groups.items():
        x_field = "pcs_z_within_spec" if label == "combined" else "pcs_mean"
        diagnostics["correlations"][label] = {
            delta: _correlation(rows, x_field, delta) for delta in DELTA_FIELDS
        }
        diagnostics["zero_threshold"][label] = _zero_threshold_test(rows, "delta_class_vs_vae")
    diagnostics["exploratory_correlations_by_timestep"] = timestep_diagnostics

    os.makedirs(args.output_dir, exist_ok=True)
    _write_csv(os.path.join(args.output_dir, "pcs_accuracy_merged.csv"), merged)
    _write_csv(
        os.path.join(args.output_dir, "pcs_timestep_accuracy_correlations.csv"),
        timestep_diagnostics,
    )
    with open(os.path.join(args.output_dir, "pcs_accuracy_diagnostics.json"), "w", encoding="utf-8") as file:
        json.dump(diagnostics, file, ensure_ascii=False, indent=2)
        file.write("\n")

    specs = sorted(spec for spec in groups if spec != "combined")
    figure, axes = plt.subplots(2, 2, figsize=(16, 12))
    for axis, spec in zip(axes[0], specs[:2]):
        _scatter(axis, groups[spec], "pcs_mean", "delta_class_vs_vae",
                 f"{spec}: PCS vs class-prompt gain over VAE", "Mean PCS")
    _scatter(axes[1, 0], merged, "pcs_z_within_spec", "delta_class_vs_vae",
             "Combined: normalized PCS vs class-prompt gain", "Within-subset PCS z-score")
    _scatter(axes[1, 1], merged, "pcs_z_within_spec", "delta_best_diffusion_vs_preserved",
             "Combined: PCS vs best refinement-family gain", "Within-subset PCS z-score")
    figure.tight_layout()
    figure.savefig(os.path.join(args.output_dir, "pcs_accuracy_relationship.png"), dpi=180)
    plt.close(figure)

    timestep_figure, timestep_axis = plt.subplots(figsize=(10, 6))
    for spec in specs:
        spec_rows = [row for row in timestep_diagnostics if row["spec"] == spec]
        timestep_axis.plot(
            [row["timestep"] for row in spec_rows],
            [row["spearman_rho"] for row in spec_rows],
            marker="o",
            label=spec,
        )
    timestep_axis.axhline(0, color="black", linewidth=0.8, alpha=0.6)
    timestep_axis.set_xlabel("SDXL training timestep")
    timestep_axis.set_ylabel("Spearman rho: PCS vs class-prompt gain over VAE")
    timestep_axis.set_title("Exploratory PCS correlation by noise level")
    timestep_axis.grid(alpha=0.25)
    timestep_axis.legend()
    timestep_figure.tight_layout()
    timestep_figure.savefig(
        os.path.join(args.output_dir, "pcs_timestep_correlations.png"), dpi=180
    )
    plt.close(timestep_figure)
    print(f"Saved PCS/accuracy diagnostics to: {args.output_dir}")


if __name__ == "__main__":
    main()
