"""Summarize the held-out Montage and conflict-control experiment."""

import argparse
import csv
import json
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHODS = (
    "coda_baseline",
    "montage_common_mode",
    "montage_soft_alpha_0p5",
    "montage_kappa_cap_0p3",
)
PAIRED_COMPARISONS = (
    ("montage_minus_baseline", "montage_common_mode", "coda_baseline"),
    ("soft_minus_montage", "montage_soft_alpha_0p5", "montage_common_mode"),
    ("kappa_minus_montage", "montage_kappa_cap_0p3", "montage_common_mode"),
    ("soft_minus_baseline", "montage_soft_alpha_0p5", "coda_baseline"),
    ("kappa_minus_baseline", "montage_kappa_cap_0p3", "coda_baseline"),
)


def _load(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing classifier result: {path}")
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _result_path(trained_root, spec, generation_seed, method):
    return os.path.join(
        trained_root,
        spec,
        f"seed_{generation_seed}",
        f"{method}-resnet_ap",
        "per_class_accuracy_all_seeds.json",
    )


def _runs_by_seed(payload, context):
    runs = {}
    for run in payload.get("runs", []):
        seed = int(run["training_seed"])
        if seed in runs:
            raise ValueError(f"Duplicate classifier seed {seed} in {context}")
        runs[seed] = run
    if not runs:
        raise ValueError(f"No classifier runs found in {context}")
    return runs


def _mean_std(values):
    values = [float(value) for value in values]
    return {
        "values": values,
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "count": len(values),
    }


def _write_json(path, payload):
    temporary = f"{path}.tmp"
    with open(temporary, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")
    os.replace(temporary, path)


def _write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(trained_root, specs, generation_seeds, output_dir):
    if os.path.exists(output_dir):
        raise FileExistsError(f"Refusing to overwrite summary: {output_dir}")
    temporary_output = f"{output_dir}.tmp"
    if os.path.exists(temporary_output):
        raise FileExistsError(
            f"Incomplete temporary summary exists; inspect and remove it before retrying: "
            f"{temporary_output}"
        )

    loaded = {}
    run_maps = {}
    for spec in specs:
        loaded[spec] = {}
        run_maps[spec] = {}
        for generation_seed in generation_seeds:
            loaded[spec][generation_seed] = {}
            run_maps[spec][generation_seed] = {}
            for method in METHODS:
                path = _result_path(trained_root, spec, generation_seed, method)
                payload = _load(path)
                loaded[spec][generation_seed][method] = (path, payload)
                run_maps[spec][generation_seed][method] = _runs_by_seed(
                    payload, f"{spec}/seed_{generation_seed}/{method}"
                )

            seed_sets = [
                set(run_maps[spec][generation_seed][method]) for method in METHODS
            ]
            if not all(seed_set == seed_sets[0] for seed_set in seed_sets[1:]):
                raise ValueError(
                    f"Methods for {spec}/seed_{generation_seed} do not contain "
                    "identical classifier seeds."
                )

    condition_rows = []
    paired_rows = []
    class_observations = defaultdict(lambda: defaultdict(list))
    class_metadata = {}

    for spec in specs:
        for generation_seed in generation_seeds:
            classifier_seeds = sorted(
                run_maps[spec][generation_seed][METHODS[0]]
            )
            for classifier_seed in classifier_seeds:
                scores = {}
                for method in METHODS:
                    path, payload = loaded[spec][generation_seed][method]
                    run = run_maps[spec][generation_seed][method][classifier_seed]
                    scores[method] = float(run["overall_top1"])
                    condition_rows.append({
                        "spec": spec,
                        "generation_seed": generation_seed,
                        "classifier_seed": classifier_seed,
                        "method": method,
                        "overall_top1": scores[method],
                        "result_path": os.path.abspath(path),
                    })
                    for class_row in run["classes"]:
                        class_id = class_row["class_id"]
                        class_observations[(spec, class_id)][method].append(
                            float(class_row["accuracy"])
                        )
                    for info in payload.get("class_summary", []):
                        class_metadata[(spec, info["class_id"])] = {
                            "local_label": int(info["local_label"]),
                            "class_name": info.get("class_name", info["class_id"]),
                        }

                row = {
                    "spec": spec,
                    "generation_seed": generation_seed,
                    "classifier_seed": classifier_seed,
                }
                for label, left, right in PAIRED_COMPARISONS:
                    row[label] = scores[left] - scores[right]
                paired_rows.append(row)

    summary = {
        "protocol": {
            "development_subsets": [spec for spec in specs if spec != "imageC"],
            "held_out_confirmation_subset": "imageC" if "imageC" in specs else None,
            "methods": list(METHODS),
            "generation_seeds": list(generation_seeds),
            "primary_comparisons": [
                "montage_minus_baseline",
                "soft_minus_montage",
                "kappa_minus_montage",
            ],
        },
        "subsets": {},
        "cross_subset": {},
    }

    for spec in specs:
        spec_rows = [row for row in condition_rows if row["spec"] == spec]
        spec_paired = [row for row in paired_rows if row["spec"] == spec]
        method_summary = {}
        for method in METHODS:
            method_rows = [row for row in spec_rows if row["method"] == method]
            by_generation_seed = {}
            for generation_seed in generation_seeds:
                values = [
                    row["overall_top1"] for row in method_rows
                    if row["generation_seed"] == generation_seed
                ]
                by_generation_seed[str(generation_seed)] = _mean_std(values)
            method_summary[method] = {
                "by_generation_seed": by_generation_seed,
                "all_classifier_runs": _mean_std(
                    [row["overall_top1"] for row in method_rows]
                ),
            }

        paired_summary = {}
        for label, _, _ in PAIRED_COMPARISONS:
            by_generation_seed = {}
            for generation_seed in generation_seeds:
                values = [
                    row[label] for row in spec_paired
                    if row["generation_seed"] == generation_seed
                ]
                by_generation_seed[str(generation_seed)] = _mean_std(values)
            paired_summary[label] = {
                "by_generation_seed": by_generation_seed,
                "all_paired_classifier_runs": _mean_std(
                    [row[label] for row in spec_paired]
                ),
            }
        summary["subsets"][spec] = {
            "methods": method_summary,
            "paired_comparisons": paired_summary,
        }

    for label, _, _ in PAIRED_COMPARISONS:
        values = [row[label] for row in paired_rows]
        summary["cross_subset"][label] = {
            "all_paired_classifier_runs": _mean_std(values),
            "subset_means": {
                spec: float(np.mean([
                    row[label] for row in paired_rows if row["spec"] == spec
                ]))
                for spec in specs
            },
        }

    class_rows = []
    for key, methods in sorted(
        class_observations.items(),
        key=lambda item: (
            specs.index(item[0][0]),
            class_metadata.get(item[0], {}).get("local_label", 10**9),
        ),
    ):
        spec, class_id = key
        metadata = class_metadata.get(key, {})
        row = {
            "spec": spec,
            "local_label": metadata.get("local_label", ""),
            "class_id": class_id,
            "class_name": metadata.get("class_name", class_id),
        }
        for method in METHODS:
            row[f"{method}_mean"] = float(np.mean(methods[method]))
        for label, left, right in PAIRED_COMPARISONS:
            row[label] = row[f"{left}_mean"] - row[f"{right}_mean"]
        class_rows.append(row)

    os.makedirs(temporary_output)
    _write_json(os.path.join(temporary_output, "experiment_summary.json"), summary)
    _write_csv(os.path.join(temporary_output, "per_classifier_run.csv"), condition_rows)
    _write_csv(os.path.join(temporary_output, "paired_comparisons.csv"), paired_rows)
    _write_csv(os.path.join(temporary_output, "per_class_comparison.csv"), class_rows)

    figure, axes = plt.subplots(1, len(specs), figsize=(5.5 * len(specs), 4), squeeze=False)
    for axis, spec in zip(axes[0], specs):
        means = [
            summary["subsets"][spec]["methods"][method]["all_classifier_runs"]["mean"]
            for method in METHODS
        ]
        errors = [
            summary["subsets"][spec]["methods"][method]["all_classifier_runs"]["std"]
            for method in METHODS
        ]
        labels = ("baseline", "montage", "montage+soft", "montage+cap")
        axis.bar(np.arange(len(METHODS)), means, yerr=errors, capsize=3)
        axis.set_xticks(np.arange(len(METHODS)), labels, rotation=20, ha="right")
        axis.set_ylabel("Validation top-1 accuracy")
        axis.set_title(spec + (" (held out)" if spec == "imageC" else ""))
    figure.tight_layout()
    figure.savefig(os.path.join(temporary_output, "method_accuracy_by_subset.png"), dpi=180)
    plt.close(figure)

    primary = ("montage_minus_baseline", "soft_minus_montage", "kappa_minus_montage")
    figure, axes = plt.subplots(1, len(specs), figsize=(5.5 * len(specs), 4), squeeze=False)
    for axis, spec in zip(axes[0], specs):
        positions = np.arange(len(generation_seeds))
        width = 0.24
        for index, label in enumerate(primary):
            values = [
                summary["subsets"][spec]["paired_comparisons"][label]
                ["by_generation_seed"][str(seed)]["mean"]
                for seed in generation_seeds
            ]
            axis.bar(positions + (index - 1) * width, values, width, label=label)
        axis.axhline(0.0, color="black", linewidth=1)
        axis.set_xticks(positions, [str(seed) for seed in generation_seeds])
        axis.set_xlabel("Generation seed")
        axis.set_ylabel("Paired accuracy difference")
        axis.set_title(spec + (" (held out)" if spec == "imageC" else ""))
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(os.path.join(temporary_output, "paired_gains_by_generation_seed.png"), dpi=180)
    plt.close(figure)
    os.replace(temporary_output, output_dir)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trained_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--specs", nargs="+", default=["imageA", "imageB", "imageC"])
    parser.add_argument("--generation_seeds", nargs="+", type=int, default=[0, 1])
    args = parser.parse_args()
    summary = summarize(
        args.trained_root, args.specs, args.generation_seeds, args.output_dir
    )
    for spec in args.specs:
        comparisons = summary["subsets"][spec]["paired_comparisons"]
        print(
            f"{spec}: montage-baseline "
            f"{comparisons['montage_minus_baseline']['all_paired_classifier_runs']['mean']:.2f}, "
            f"soft-montage "
            f"{comparisons['soft_minus_montage']['all_paired_classifier_runs']['mean']:.2f}, "
            f"cap-montage "
            f"{comparisons['kappa_minus_montage']['all_paired_classifier_runs']['mean']:.2f}"
        )
    print(f"Saved final Montage conflict summary to: {args.output_dir}")


if __name__ == "__main__":
    main()
