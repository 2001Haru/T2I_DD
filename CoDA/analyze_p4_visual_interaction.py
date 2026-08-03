"""Analyze how prototype initialization changes P4 text effects.

This is intentionally a post-hoc analysis of paired_effects_raw.csv. It does
not load a vision encoder or refit a probe, and it never modifies the primary
P4 outputs.
"""

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


EFFECTS = (
    "delta_target",
    "delta_pull",
    "caption_rank_improvement",
    "visual_rank_drop",
)
PAIR_KEYS = (
    "encoder",
    "probe",
    "spec",
    "class_key",
    "class_id",
    "class_name",
    "visual_cluster_id",
    "caption_source_cluster_id",
    "generation_seed",
    "image_seed",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Supplemental paired P4 interaction analysis")
    parser.add_argument("--paired-effects", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--random-seed", type=int, default=20260803)
    return parser.parse_args()


def read_csv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def atomic_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def build_paired_interactions(rows):
    paired = defaultdict(dict)
    for row in rows:
        key = tuple(row[column] for column in PAIR_KEYS)
        mode = row["visual_mode"]
        if mode in paired[key]:
            raise ValueError(f"Duplicate visual mode {mode} for paired unit {key}")
        paired[key][mode] = row

    output = []
    for key, modes in sorted(paired.items()):
        if set(modes) != {"i0g0", "i1g0"}:
            raise ValueError(f"Incomplete visual-mode pair for {key}: {sorted(modes)}")
        i0, i1 = modes["i0g0"], modes["i1g0"]
        if i0["image_seed"] != i1["image_seed"]:
            raise ValueError(f"Visual modes do not share image seed for {key}")
        record = {column: value for column, value in zip(PAIR_KEYS, key)}
        for effect in EFFECTS:
            i0_value = float(i0[effect])
            i1_value = float(i1[effect])
            record[f"i0g0_{effect}"] = i0_value
            record[f"i1g0_{effect}"] = i1_value
            record[f"interaction_{effect}"] = i1_value - i0_value
        output.append(record)
    return output


def bootstrap(values, samples, seed):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        raise ValueError("Cannot bootstrap an empty sample")
    rng = np.random.default_rng(seed)
    draws = np.mean(rng.choice(values, size=(samples, len(values)), replace=True), axis=1)
    return float(np.mean(values)), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def group_mean_rows(rows, effect):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["class_key"], row["visual_cluster_id"])].append(
            float(row[f"interaction_{effect}"])
        )
    return [float(np.mean(values)) for values in grouped.values()]


def summarize_interactions(rows, samples, random_seed):
    output = []
    scopes = ["combined"] + sorted({row["spec"] for row in rows})
    counter = 0
    for scope in scopes:
        scoped = rows if scope == "combined" else [row for row in rows if row["spec"] == scope]
        for encoder in sorted({row["encoder"] for row in scoped}):
            for probe in sorted({row["probe"] for row in scoped}):
                selected = [
                    row for row in scoped
                    if row["encoder"] == encoder and row["probe"] == probe
                ]
                for effect in EFFECTS:
                    values = group_mean_rows(selected, effect)
                    mean, lower, upper = bootstrap(values, samples, random_seed + counter)
                    counter += 1
                    output.append(
                        {
                            "scope": scope,
                            "encoder": encoder,
                            "probe": probe,
                            "effect": effect,
                            "interaction": "i1g0_minus_i0g0",
                            "mean": mean,
                            "bootstrap_ci_lower": lower,
                            "bootstrap_ci_upper": upper,
                            "class_cluster_groups": len(values),
                            "raw_paired_observations": len(selected),
                            "positive_group_fraction": float(np.mean(np.asarray(values) > 0)),
                        }
                    )
    return output


def summarize_by_seed(rows, samples, random_seed):
    output = []
    counter = 0
    for scope in ["combined"] + sorted({row["spec"] for row in rows}):
        scoped = rows if scope == "combined" else [row for row in rows if row["spec"] == scope]
        for encoder in sorted({row["encoder"] for row in scoped}):
            for probe in sorted({row["probe"] for row in scoped}):
                for generation_seed in sorted({int(row["generation_seed"]) for row in scoped}):
                    selected = [
                        row for row in scoped
                        if row["encoder"] == encoder and row["probe"] == probe
                        and int(row["generation_seed"]) == generation_seed
                    ]
                    for effect in EFFECTS:
                        values = [float(row[f"interaction_{effect}"]) for row in selected]
                        mean, lower, upper = bootstrap(values, samples, random_seed + counter)
                        counter += 1
                        output.append(
                            {
                                "scope": scope,
                                "encoder": encoder,
                                "probe": probe,
                                "generation_seed": generation_seed,
                                "effect": effect,
                                "interaction": "i1g0_minus_i0g0",
                                "mean": mean,
                                "bootstrap_ci_lower": lower,
                                "bootstrap_ci_upper": upper,
                                "class_cluster_groups": len(values),
                                "positive_group_fraction": float(np.mean(np.asarray(values) > 0)),
                            }
                        )
    return output


def plot_summary(output_dir, summary, by_seed):
    selected = [
        row for row in summary
        if row["scope"] == "combined" and row["encoder"] == "dino"
        and row["effect"] in {"delta_target", "delta_pull"}
    ]
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    labels = [f"{row['probe']}\n{row['effect'].replace('delta_', '')}" for row in selected]
    means = [row["mean"] for row in selected]
    errors = np.asarray([
        [row["mean"] - row["bootstrap_ci_lower"] for row in selected],
        [row["bootstrap_ci_upper"] - row["mean"] for row in selected],
    ])
    axes[0].bar(np.arange(len(selected)), means, yerr=errors, capsize=4)
    axes[0].set_xticks(np.arange(len(selected)), labels)
    axes[0].set_title("Paired visual-mode interaction")
    axes[0].set_ylabel("I1G0 effect - I0G0 effect")

    seed_rows = [
        row for row in by_seed
        if row["scope"] == "combined" and row["encoder"] == "dino"
        and row["probe"] == "nearest_centroid"
        and row["effect"] in {"delta_target", "delta_pull"}
    ]
    for effect in ("delta_target", "delta_pull"):
        effect_rows = sorted(
            (row for row in seed_rows if row["effect"] == effect),
            key=lambda row: row["generation_seed"],
        )
        axes[1].plot(
            [row["generation_seed"] for row in effect_rows],
            [row["mean"] for row in effect_rows],
            marker="o", label=effect.replace("delta_", ""),
        )
    axes[1].set_title("Interaction by generation seed")
    axes[1].set_xlabel("Generation seed")
    axes[1].set_ylabel("I1G0 effect - I0G0 effect")
    axes[1].legend()
    for axis in axes:
        axis.axhline(0, color="black", linestyle="--", linewidth=1)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("P4 supplemental: does prototype initialization attenuate text effects?")
    figure.tight_layout()
    figure.savefig(Path(output_dir) / "p4_visual_interaction.png", dpi=180)
    plt.close(figure)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_csv(args.paired_effects)
    interactions = build_paired_interactions(rows)
    summary = summarize_interactions(interactions, args.bootstrap_samples, args.random_seed)
    by_seed = summarize_by_seed(interactions, args.bootstrap_samples, args.random_seed + 10000)
    write_csv(output_dir / "visual_interactions_raw.csv", interactions)
    write_csv(output_dir / "visual_interactions_summary.csv", summary)
    write_csv(output_dir / "visual_interactions_by_seed.csv", by_seed)
    plot_summary(output_dir, summary, by_seed)

    dino_primary = [
        row for row in summary
        if row["scope"] == "combined" and row["encoder"] == "dino"
        and row["effect"] in {"delta_target", "delta_pull"}
    ]
    atomic_json(
        output_dir / "summary.json",
        {
            "format_version": 1,
            "interaction_definition": (
                "paired I1G0 text effect minus I0G0 text effect for the same encoder, "
                "probe, spec, class, visual cluster, caption source, generation seed, and image seed"
            ),
            "bootstrap_unit": (
                "class_key x visual_cluster_id after averaging generation seeds; "
                "generation-seed tables bootstrap class-cluster observations within each seed"
            ),
            "sign_interpretation": (
                "negative means prototype initialization attenuates the measured text effect; "
                "positive means it amplifies it"
            ),
            "primary": {
                f"{row['probe']}_{row['effect']}": row for row in dino_primary
            },
            "interpretation_boundary": (
                "I0G0 uses the full denoising schedule, whereas I1G0 starts from a noised "
                "prototype at the configured strength and may use fewer denoising steps. "
                "This interaction therefore measures the complete initialization regime, "
                "not prototype content alone."
            ),
        },
    )
    print(f"P4 supplemental interaction analysis complete: {output_dir}")


if __name__ == "__main__":
    main()
