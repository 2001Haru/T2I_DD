"""Hierarchical P6 inference and class-level mechanism diagnostics."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from summarize_p6_downstream_value import (
    PROMPTS,
    REGIMES,
    class_scores,
    contrast_definitions,
    load_results,
    run_scores,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trained-root", required=True)
    parser.add_argument("--p2p3-run-dir", required=True)
    parser.add_argument("--p4-run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--specs", nargs="+", required=True)
    parser.add_argument("--generation-seeds", nargs="+", type=int, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--permutation-samples", type=int, default=10000)
    parser.add_argument("--random-seed", type=int, default=20260805)
    return parser.parse_args()


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def rankdata(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def correlation(x, y, method):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if method == "spearman":
        x, y = rankdata(x), rankdata(y)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def percentile_ci(values):
    values = np.asarray([value for value in values if np.isfinite(value)])
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def contrast_for_payloads(payloads, terms, class_id=None):
    maps = []
    for coefficient, regime, prompt in terms:
        scores = run_scores(payloads[(regime, prompt)]) if class_id is None else class_scores(
            payloads[(regime, prompt)]
        )
        maps.append((coefficient, scores))
    if class_id is None:
        seeds = sorted(set.intersection(*(set(scores) for _, scores in maps)))
        return float(np.mean([
            sum(coefficient * scores[seed] for coefficient, scores in maps)
            for seed in seeds
        ]))
    seeds = sorted(set.intersection(*(
        {seed for seed, key_class in scores if key_class == class_id}
        for _, scores in maps
    )))
    return float(np.mean([
        sum(coefficient * scores[(seed, class_id)] for coefficient, scores in maps)
        for seed in seeds
    ]))


def build_generation_records(results, specs, generation_seeds):
    overall = {}
    classes = {}
    for spec in specs:
        sample = results[(spec, generation_seeds[0], REGIMES[0], PROMPTS[0])]
        metadata = {
            row["class_id"]: {
                "local_label": int(row["local_label"]),
                "class_name": row["class_name"],
            }
            for row in sample["runs"][0]["classes"]
        }
        for generation_seed in generation_seeds:
            payloads = {
                (regime, prompt): results[(spec, generation_seed, regime, prompt)]
                for regime in REGIMES for prompt in PROMPTS
            }
            overall[(spec, generation_seed)] = payloads
            for class_id, item in metadata.items():
                row = {
                    "spec": spec,
                    "class_id": class_id,
                    "class_key": f"{spec}:{class_id}",
                    "class_name": item["class_name"],
                    "local_label": item["local_label"],
                    "generation_seed": generation_seed,
                }
                for regime in REGIMES:
                    for prompt in PROMPTS:
                        scores = class_scores(payloads[(regime, prompt)])
                        row[f"{regime}_{prompt}"] = float(np.mean([
                            value for (training_seed, key_class), value in scores.items()
                            if key_class == class_id
                        ]))
                classes[(spec, class_id, generation_seed)] = row
    return overall, classes


def hierarchical_contrast_bootstrap(overall, definitions, specs, generation_seeds, samples, rng):
    unit_values = {}
    for (spec, generation_seed), payloads in overall.items():
        for name, terms in definitions.items():
            unit_values[(spec, generation_seed, name)] = contrast_for_payloads(payloads, terms)
    rows = []
    for name in definitions:
        observed = np.asarray([
            unit_values[(spec, seed, name)] for spec in specs for seed in generation_seeds
        ])
        boot = []
        for _ in range(samples):
            values = []
            for spec_index in rng.integers(0, len(specs), len(specs)):
                spec = specs[spec_index]
                for seed_index in rng.integers(0, len(generation_seeds), len(generation_seeds)):
                    values.append(unit_values[(spec, generation_seeds[seed_index], name)])
            boot.append(float(np.mean(values)))
        lower, upper = percentile_ci(boot)
        rows.append({
            "contrast": name,
            "mean": float(observed.mean()),
            "bootstrap_ci_lower": lower,
            "bootstrap_ci_upper": upper,
            "spec_generation_units": len(observed),
            "bootstrap_samples": samples,
            "unit_values": json.dumps(observed.tolist()),
        })
    return rows


def load_p3(path):
    rows = read_csv(path)
    selected = {}
    for row in rows:
        if row["metric"] == "bidirectional_mrr":
            selected[row["class_key"]] = {
                "p3_correspondence_delta": float(row["delta_over_null"]),
                "p3_correspondence_true": float(row["true_value"]),
            }
    return selected


def load_p4(path):
    grouped = defaultdict(list)
    for row in read_csv(path):
        if row["encoder"] != "dino" or row["probe"] not in {"linear_probe", "nearest_centroid"}:
            continue
        if row["visual_mode"] not in {"i0g0", "i1g0"}:
            continue
        key = (
            row["class_key"], int(row["generation_seed"]), row["visual_mode"], row["probe"]
        )
        grouped[key].append(float(row["delta_pull"]))
    return {key: float(np.mean(values)) for key, values in grouped.items()}


def add_external_metrics(class_records, p3, p4):
    rows = []
    for key in sorted(class_records):
        row = dict(class_records[key])
        if row["class_key"] not in p3:
            raise KeyError(f"Missing P3 correspondence for {row['class_key']}")
        row.update(p3[row["class_key"]])
        for regime in REGIMES:
            row[f"{regime}_content_gain"] = row[f"{regime}_correct"] - row[
                f"{regime}_matched_label"
            ]
            row[f"{regime}_raw_dcs_gain"] = row[f"{regime}_correct"] - row[
                f"{regime}_label"
            ]
            row[f"{regime}_correct_minus_shuffled"] = row[f"{regime}_correct"] - row[
                f"{regime}_shuffled"
            ]
        for regime in ("i0g0", "i1g0"):
            for probe in ("linear_probe", "nearest_centroid"):
                p4_key = (row["class_key"], row["generation_seed"], regime, probe)
                if p4_key not in p4:
                    raise KeyError(f"Missing P4 source-pull value for {p4_key}")
                row[f"p4_{regime}_{probe}_delta_pull"] = p4[p4_key]
        rows.append(row)
    return rows


def aggregate_relation_points(records, x_key, y_key):
    grouped = defaultdict(list)
    for row in records:
        grouped[(row["spec"], row["class_id"])].append(row)
    points = []
    for (spec, class_id), members in sorted(grouped.items()):
        points.append({
            "spec": spec,
            "class_id": class_id,
            "class_key": members[0]["class_key"],
            "class_name": members[0]["class_name"],
            "x": float(np.mean([row[x_key] for row in members])),
            "y": float(np.mean([row[y_key] for row in members])),
        })
    return points


def hierarchical_correlation(records, x_key, y_key, method, samples, permutations, rng, expected):
    by_spec_class = defaultdict(list)
    for row in records:
        by_spec_class[(row["spec"], row["class_id"])].append(row)
    specs = sorted({row["spec"] for row in records})
    classes_by_spec = {
        spec: sorted(key for key in by_spec_class if key[0] == spec)
        for spec in specs
    }
    points = aggregate_relation_points(records, x_key, y_key)
    observed = correlation([row["x"] for row in points], [row["y"] for row in points], method)
    boot = []
    for _ in range(samples):
        bx, by = [], []
        for spec_index in rng.integers(0, len(specs), len(specs)):
            spec = specs[spec_index]
            class_keys = classes_by_spec[spec]
            for class_index in rng.integers(0, len(class_keys), len(class_keys)):
                members = by_spec_class[class_keys[class_index]]
                selected = [members[index] for index in rng.integers(0, len(members), len(members))]
                bx.append(float(np.mean([row[x_key] for row in selected])))
                by.append(float(np.mean([row[y_key] for row in selected])))
        boot.append(correlation(bx, by, method))
    lower, upper = percentile_ci(boot)

    null = []
    for _ in range(permutations):
        permuted_y = []
        actual_x = []
        for spec in specs:
            selected = [row for row in points if row["spec"] == spec]
            actual_x.extend(row["x"] for row in selected)
            values = np.asarray([row["y"] for row in selected])
            permuted_y.extend(values[rng.permutation(len(values))])
        null.append(correlation(actual_x, permuted_y, method))
    null = np.asarray(null)
    if expected == "negative":
        one_sided = (1 + int(np.sum(null <= observed))) / (permutations + 1)
    else:
        one_sided = (1 + int(np.sum(null >= observed))) / (permutations + 1)
    two_sided = (1 + int(np.sum(np.abs(null) >= abs(observed)))) / (permutations + 1)
    return {
        "value": observed,
        "bootstrap_ci_lower": lower,
        "bootstrap_ci_upper": upper,
        "permutation_p_expected_direction": one_sided,
        "permutation_p_two_sided": two_sided,
        "classes": len(points),
        "class_generation_records": len(records),
        "bootstrap_samples": samples,
        "permutation_samples": permutations,
    }, points


def relation_definitions():
    rows = [
        {
            "analysis": "dcs_rescue_primary",
            "regime": "i0g0",
            "probe": "",
            "x_key": "i0g0_matched_label",
            "y_key": "i0g0_content_gain",
            "expected": "negative",
        },
        {
            "analysis": "dcs_rescue_raw_label_sensitivity",
            "regime": "i0g0",
            "probe": "",
            "x_key": "i0g0_label",
            "y_key": "i0g0_raw_dcs_gain",
            "expected": "negative",
        },
    ]
    for regime in REGIMES:
        rows.append({
            "analysis": "p3_correspondence_vs_downstream_value",
            "regime": regime,
            "probe": "",
            "x_key": "p3_correspondence_delta",
            "y_key": f"{regime}_correct_minus_shuffled",
            "expected": "positive",
        })
    for regime in ("i0g0", "i1g0"):
        for probe in ("linear_probe", "nearest_centroid"):
            rows.append({
                "analysis": "p4_source_pull_vs_downstream_value",
                "regime": regime,
                "probe": probe,
                "x_key": f"p4_{regime}_{probe}_delta_pull",
                "y_key": f"{regime}_correct_minus_shuffled",
                "expected": "positive",
            })
    return rows


def plot_relations(points_by_name, output_path):
    selected_names = [
        "dcs_rescue_primary|i0g0|",
        "dcs_rescue_raw_label_sensitivity|i0g0|",
        "p3_correspondence_vs_downstream_value|i0g0|",
        "p3_correspondence_vs_downstream_value|i1g0|",
        "p4_source_pull_vs_downstream_value|i0g0|linear_probe",
        "p4_source_pull_vs_downstream_value|i1g0|linear_probe",
    ]
    figure, axes = plt.subplots(2, 3, figsize=(18, 10))
    for axis, name in zip(axes.flat, selected_names):
        payload = points_by_name[name]
        points, result, definition = payload["points"], payload["result"], payload["definition"]
        for spec in sorted({row["spec"] for row in points}):
            subset = [row for row in points if row["spec"] == spec]
            axis.scatter([row["x"] for row in subset], [row["y"] for row in subset], label=spec)
        axis.axhline(0, color="black", linestyle="--", linewidth=1)
        axis.set_title(
            f"{definition['analysis']} {definition['regime']}\n"
            f"Spearman rho={result['value']:.3f}"
        )
        axis.set_xlabel(definition["x_key"])
        axis.set_ylabel(definition["y_key"])
        axis.legend()
    figure.suptitle("P6 class-level mechanism diagnostics")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.random_seed)
    results = load_results(Path(args.trained_root), args.specs, args.generation_seeds)
    overall, class_records = build_generation_records(results, args.specs, args.generation_seeds)

    hierarchical_rows = hierarchical_contrast_bootstrap(
        overall, contrast_definitions(), args.specs, args.generation_seeds,
        args.bootstrap_samples, rng,
    )
    write_csv(output_dir / "hierarchical_paired_contrasts.csv", hierarchical_rows)

    p3 = load_p3(Path(args.p2p3_run_dir) / "dcs_correspondence_per_class.csv")
    p4 = load_p4(Path(args.p4_run_dir) / "analysis" / "paired_effects_raw.csv")
    records = add_external_metrics(class_records, p3, p4)
    write_csv(output_dir / "class_generation_metrics.csv", records)

    relationship_rows = []
    point_rows = []
    points_by_name = {}
    for definition in relation_definitions():
        for method in ("pearson", "spearman"):
            result, points = hierarchical_correlation(
                records, definition["x_key"], definition["y_key"], method,
                args.bootstrap_samples, args.permutation_samples, rng,
                definition["expected"],
            )
            relationship_rows.append({**definition, "correlation": method, **result})
            if method == "spearman":
                name = "|".join([
                    definition["analysis"], definition["regime"], definition.get("probe", "")
                ])
                points_by_name[name] = {
                    "points": points, "result": result, "definition": definition,
                }
                for row in points:
                    point_rows.append({**definition, **row})
    write_csv(output_dir / "class_relationship_correlations.csv", relationship_rows)
    write_csv(output_dir / "class_relationship_points.csv", point_rows)
    plot_relations(points_by_name, output_dir / "p6_class_relationships.png")

    primary = {
        "dcs_rescue": next(
            row for row in relationship_rows
            if row["analysis"] == "dcs_rescue_primary" and row["correlation"] == "spearman"
        ),
        "p3_correspondence_i0g0": next(
            row for row in relationship_rows
            if row["analysis"] == "p3_correspondence_vs_downstream_value"
            and row["regime"] == "i0g0" and row["correlation"] == "spearman"
        ),
        "p4_source_pull_i0g0": next(
            row for row in relationship_rows
            if row["analysis"] == "p4_source_pull_vs_downstream_value"
            and row["regime"] == "i0g0" and row["probe"] == "linear_probe"
            and row["correlation"] == "spearman"
        ),
        "p4_source_pull_i1g0": next(
            row for row in relationship_rows
            if row["analysis"] == "p4_source_pull_vs_downstream_value"
            and row["regime"] == "i1g0" and row["probe"] == "linear_probe"
            and row["correlation"] == "spearman"
        ),
    }
    summary = {
        "format_version": 1,
        "bootstrap_unit_overall": "spec -> generation seed after averaging classifier repeats",
        "bootstrap_unit_correlations": (
            "spec -> class -> generation seed after averaging classifier repeats; "
            "permutation is within spec"
        ),
        "interpretation_boundary": (
            "P3 correspondence is class-level and P4 source-pull is averaged over eligible "
            "visual clusters. Correlation does not establish that either mechanism causes "
            "downstream class accuracy."
        ),
        "primary": primary,
    }
    (output_dir / "class_relationship_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Saved P6 hierarchical and class-level analysis to {output_dir}")


if __name__ == "__main__":
    main()
