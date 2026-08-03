"""Evaluate P4 text execution with probes frozen exclusively on real images."""

import argparse
import csv
import gc
import json
import os
import pickle
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score

from diagnose_cluster_recoverability import IndependentEncoder


PROBES = ("nearest_centroid", "linear_probe")


def parse_args():
    parser = argparse.ArgumentParser(description="P4 frozen-probe execution diagnostic")
    parser.add_argument("--prepared-dir", required=True)
    parser.add_argument("--generation-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dino-model", required=True)
    parser.add_argument("--clip-model", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--random-seed", type=int, default=20260803)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def atomic_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


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


def normalize(features):
    return features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)


def load_prompt_records(dataset_dir):
    records = []
    for path in sorted(Path(dataset_dir).glob("prompt_records_gpu*.json")):
        records.extend(json.loads(path.read_text(encoding="utf-8")))
    if not records:
        raise FileNotFoundError(f"No prompt records in {dataset_dir}")
    return {(row["class_id"], int(row["visual_cluster_id"])): row for row in records}


def inventory(prepared_dir, manifest_path):
    pairs = read_csv(Path(prepared_dir) / "pair_manifest.csv")
    manifests = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    expected_conditions = {
        (visual_mode, prompt_condition)
        for visual_mode in ("i0g0", "i1g0")
        for prompt_condition in ("label", "correct", "shuffled")
    }
    by_spec_seed = defaultdict(set)
    dataset_entries = {}
    for row in manifests:
        key = (
            row["spec"], int(row["generation_seed"]),
            row["visual_mode"], row["prompt_condition"],
        )
        if key in dataset_entries:
            raise ValueError(f"Duplicate generation manifest entry: {key}")
        dataset_entries[key] = Path(row["dataset_dir"])
        by_spec_seed[(key[0], key[1])].add((key[2], key[3]))
    for key, conditions in by_spec_seed.items():
        if conditions != expected_conditions:
            raise ValueError(f"Incomplete P4 condition matrix for {key}: {conditions}")

    samples = []
    pair_lookup = defaultdict(list)
    for row in pairs:
        row = dict(row)
        row["visual_cluster_id"] = int(row["visual_cluster_id"])
        row["correct_caption_cluster_id"] = int(row["correct_caption_cluster_id"])
        row["shuffled_caption_cluster_id"] = int(row["shuffled_caption_cluster_id"])
        pair_lookup[row["spec"]].append(row)
    for key in sorted(dataset_entries):
        spec, seed, visual_mode, condition = key
        dataset_dir = dataset_entries[key]
        prompt_records = load_prompt_records(dataset_dir)
        for pair in pair_lookup[spec]:
            cluster_id = pair["visual_cluster_id"]
            image_path = dataset_dir / pair["class_id"] / f"{cluster_id}.png"
            if not image_path.is_file():
                raise FileNotFoundError(f"Missing generated P4 image: {image_path}")
            prompt = prompt_records.get((pair["class_id"], cluster_id))
            if prompt is None:
                raise ValueError(f"Missing prompt record for {image_path}")
            expected_caption = None
            if condition == "correct":
                expected_caption = pair["correct_caption_cluster_id"]
            elif condition == "shuffled":
                expected_caption = pair["shuffled_caption_cluster_id"]
            if prompt.get("caption_cluster_id") != expected_caption:
                raise ValueError(
                    f"Caption source mismatch for {image_path}: "
                    f"{prompt.get('caption_cluster_id')} vs {expected_caption}"
                )
            samples.append(
                {
                    **pair,
                    "generation_seed": seed,
                    "visual_mode": visual_mode,
                    "prompt_condition": condition,
                    "image_seed": int(prompt["image_seed"]),
                    "path": str(image_path.resolve()),
                }
            )

    paired = defaultdict(dict)
    for sample in samples:
        key = (
            sample["spec"], sample["class_key"], sample["visual_cluster_id"],
            sample["generation_seed"], sample["visual_mode"],
        )
        paired[key][sample["prompt_condition"]] = sample
    for key, rows in paired.items():
        if set(rows) != {"label", "correct", "shuffled"}:
            raise ValueError(f"Incomplete paired prompts for {key}")
        if len({row["image_seed"] for row in rows.values()}) != 1:
            raise ValueError(f"Prompt conditions do not share image seed for {key}")
    cross_visual = defaultdict(set)
    for sample in samples:
        key = (
            sample["spec"], sample["class_key"], sample["visual_cluster_id"],
            sample["generation_seed"],
        )
        cross_visual[key].add(sample["image_seed"])
    for key, seeds in cross_visual.items():
        if len(seeds) != 1:
            raise ValueError(f"Visual modes do not share image seed for {key}: {seeds}")
    return samples, paired


def feature_signature(samples, model_root, encoder):
    payload = [encoder, str(Path(model_root).resolve())]
    for sample in samples:
        path = Path(sample["path"])
        stat = path.stat()
        payload.extend((str(path), str(stat.st_size), str(stat.st_mtime_ns)))
    import hashlib
    return hashlib.sha256("\n".join(payload).encode("utf-8")).hexdigest()


def load_or_extract(args, samples, encoder, model_root):
    cache_path = Path(args.output_dir) / "feature_cache" / f"{encoder}.npz"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    signature = feature_signature(samples, model_root, encoder)
    paths = np.asarray([sample["path"] for sample in samples])
    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as cached:
            if str(cached["signature"].item()) != signature:
                raise RuntimeError(f"Generated feature cache changed: {cache_path}")
            if not np.array_equal(cached["paths"].astype(str), paths):
                raise RuntimeError(f"Generated feature order changed: {cache_path}")
            return cached["features"].astype(np.float32, copy=False)
    extractor = IndependentEncoder(encoder, model_root, args.device)
    features = extractor.extract(paths.tolist(), args.batch_size)
    del extractor
    gc.collect()
    temporary = cache_path.with_name(cache_path.name + ".tmp.npz")
    np.savez_compressed(temporary, features=features, paths=paths, signature=np.asarray(signature))
    os.replace(temporary, cache_path)
    return features


def probe_scores(feature, payload, probe):
    feature = normalize(feature.reshape(1, -1))[0]
    if probe == "nearest_centroid":
        return payload["centroid_cluster_ids"], feature @ payload["centroids"].T
    standardized = (feature - payload["scaler_mean"]) / np.maximum(payload["scaler_scale"], 1e-12)
    scores = payload["ridge_coef"] @ standardized + payload["ridge_intercept"]
    return payload["class_ids"], np.asarray(scores).reshape(-1)


def score_for(cluster_ids, scores, target):
    positions = np.flatnonzero(cluster_ids == target)
    if len(positions) != 1:
        raise ValueError(f"Probe does not contain target cluster {target}")
    return float(scores[int(positions[0])])


def rank_for(cluster_ids, scores, target):
    target_score = score_for(cluster_ids, scores, target)
    return int(1 + np.sum(scores > target_score))


def margin_for(cluster_ids, scores, target):
    target_score = score_for(cluster_ids, scores, target)
    other = scores[cluster_ids != target]
    return float(target_score - np.max(other))


def paired_metrics(samples, paired, features, probe_bundle, encoder):
    sample_index = {sample["path"]: index for index, sample in enumerate(samples)}
    score_cache = {}
    for sample in samples:
        for probe in PROBES:
            payload = probe_bundle["encoders"][encoder]["classes"][sample["class_key"]]
            score_cache[(sample["path"], probe)] = probe_scores(
                features[sample_index[sample["path"]]], payload, probe
            )

    effect_rows = []
    recovery_rows = []
    for key, rows in sorted(paired.items()):
        label, correct, shuffled = rows["label"], rows["correct"], rows["shuffled"]
        k = correct["visual_cluster_id"]
        j = shuffled["shuffled_caption_cluster_id"]
        for probe in PROBES:
            label_ids, label_scores = score_cache[(label["path"], probe)]
            correct_ids, correct_scores = score_cache[(correct["path"], probe)]
            shuffled_ids, shuffled_scores = score_cache[(shuffled["path"], probe)]
            target_delta = margin_for(correct_ids, correct_scores, k) - margin_for(
                label_ids, label_scores, k
            )
            correct_preference = score_for(correct_ids, correct_scores, j) - score_for(
                correct_ids, correct_scores, k
            )
            shuffled_preference = score_for(shuffled_ids, shuffled_scores, j) - score_for(
                shuffled_ids, shuffled_scores, k
            )
            effect_rows.append(
                {
                    "encoder": encoder,
                    "probe": probe,
                    "spec": correct["spec"],
                    "class_key": correct["class_key"],
                    "class_id": correct["class_id"],
                    "class_name": correct["class_name"],
                    "visual_cluster_id": k,
                    "caption_source_cluster_id": j,
                    "generation_seed": correct["generation_seed"],
                    "visual_mode": correct["visual_mode"],
                    "image_seed": correct["image_seed"],
                    "label_target_margin": margin_for(label_ids, label_scores, k),
                    "correct_target_margin": margin_for(correct_ids, correct_scores, k),
                    "delta_target": target_delta,
                    "correct_j_minus_k": correct_preference,
                    "shuffled_j_minus_k": shuffled_preference,
                    "delta_pull": shuffled_preference - correct_preference,
                    "correct_target_rank": rank_for(correct_ids, correct_scores, k),
                    "correct_caption_source_rank": rank_for(correct_ids, correct_scores, j),
                    "shuffled_caption_rank": rank_for(shuffled_ids, shuffled_scores, j),
                    "shuffled_visual_rank": rank_for(shuffled_ids, shuffled_scores, k),
                    "caption_rank_improvement": (
                        rank_for(correct_ids, correct_scores, j)
                        - rank_for(shuffled_ids, shuffled_scores, j)
                    ),
                    "visual_rank_drop": (
                        rank_for(shuffled_ids, shuffled_scores, k)
                        - rank_for(correct_ids, correct_scores, k)
                    ),
                }
            )
            for condition, sample, ids, scores in (
                ("label", label, label_ids, label_scores),
                ("correct", correct, correct_ids, correct_scores),
                ("shuffled", shuffled, shuffled_ids, shuffled_scores),
            ):
                targets = [("visual_source", k)]
                if condition == "correct":
                    targets.append(("caption_source", k))
                elif condition == "shuffled":
                    targets.append(("caption_source", j))
                prediction = int(ids[int(np.argmax(scores))])
                for target_kind, target in targets:
                    rank = rank_for(ids, scores, target)
                    recovery_rows.append(
                        {
                            "encoder": encoder,
                            "probe": probe,
                            "spec": sample["spec"],
                            "class_key": sample["class_key"],
                            "class_id": sample["class_id"],
                            "class_name": sample["class_name"],
                            "visual_cluster_id": k,
                            "caption_source_cluster_id": (
                                j if condition == "shuffled" else (k if condition == "correct" else "")
                            ),
                            "generation_seed": sample["generation_seed"],
                            "visual_mode": sample["visual_mode"],
                            "prompt_condition": condition,
                            "target_kind": target_kind,
                            "target_cluster_id": target,
                            "predicted_cluster_id": prediction,
                            "correct": int(prediction == target),
                            "rank": rank,
                            "reciprocal_rank": 1.0 / rank,
                        }
                    )
    return effect_rows, recovery_rows


def bootstrap_ci(rows, value_key, samples, seed):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["class_key"], row["visual_cluster_id"])].append(float(row[value_key]))
    values = np.asarray([np.mean(group) for group in grouped.values()], dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = np.mean(rng.choice(values, size=(samples, len(values)), replace=True), axis=1)
    return float(np.mean(values)), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975)), len(values)


def summarize_effects(rows, bootstrap_samples, random_seed):
    output = []
    scopes = ["combined"] + sorted({row["spec"] for row in rows})
    for scope in scopes:
        selected_scope = rows if scope == "combined" else [row for row in rows if row["spec"] == scope]
        for encoder in sorted({row["encoder"] for row in selected_scope}):
            for probe in PROBES:
                for visual_mode in ("i0g0", "i1g0"):
                    selected = [
                        row for row in selected_scope
                        if row["encoder"] == encoder and row["probe"] == probe
                        and row["visual_mode"] == visual_mode
                    ]
                    for effect in (
                        "delta_target", "delta_pull",
                        "caption_rank_improvement", "visual_rank_drop",
                    ):
                        mean, lower, upper, groups = bootstrap_ci(
                            selected, effect, bootstrap_samples,
                            random_seed + len(output),
                        )
                        group_values = defaultdict(list)
                        for row in selected:
                            group_values[
                                (row["class_key"], row["visual_cluster_id"])
                            ].append(float(row[effect]))
                        output.append(
                            {
                                "scope": scope,
                                "encoder": encoder,
                                "probe": probe,
                                "visual_mode": visual_mode,
                                "effect": effect,
                                "mean": mean,
                                "bootstrap_ci_lower": lower,
                                "bootstrap_ci_upper": upper,
                                "class_cluster_groups": groups,
                                "raw_observations": len(selected),
                                "positive_group_fraction": float(np.mean([
                                    np.mean(group) > 0 for group in group_values.values()
                                ])),
                            }
                        )
    return output


def summarize_recovery(rows):
    per_class = []
    groups = defaultdict(list)
    for row in rows:
        key = (
            row["encoder"], row["probe"], row["spec"], row["class_key"],
            row["visual_mode"], row["prompt_condition"], row["target_kind"],
            row["generation_seed"],
        )
        groups[key].append(row)
    for key, selected in groups.items():
        encoder, probe, spec, class_key, visual_mode, condition, target_kind, seed = key
        targets = [int(row["target_cluster_id"]) for row in selected]
        predictions = [int(row["predicted_cluster_id"]) for row in selected]
        labels = sorted(set(targets))
        per_class.append(
            {
                "encoder": encoder, "probe": probe, "spec": spec, "class_key": class_key,
                "visual_mode": visual_mode, "prompt_condition": condition,
                "target_kind": target_kind, "generation_seed": seed,
                "top1": float(np.mean([row["correct"] for row in selected])),
                "macro_f1": float(f1_score(targets, predictions, labels=labels, average="macro", zero_division=0)),
                "mrr": float(np.mean([row["reciprocal_rank"] for row in selected])),
                "mean_rank": float(np.mean([row["rank"] for row in selected])),
                "clusters": len(selected),
            }
        )
    aggregate = []
    aggregate_groups = defaultdict(list)
    for row in per_class:
        key = (
            row["encoder"], row["probe"], row["visual_mode"],
            row["prompt_condition"], row["target_kind"],
        )
        aggregate_groups[key].append(row)
    for key, selected in aggregate_groups.items():
        aggregate.append(
            {
                "scope": "combined", "encoder": key[0], "probe": key[1],
                "visual_mode": key[2], "prompt_condition": key[3], "target_kind": key[4],
                **{metric: float(np.mean([row[metric] for row in selected]))
                   for metric in ("top1", "macro_f1", "mrr", "mean_rank")},
                "class_seed_observations": len(selected),
            }
        )
    return per_class, aggregate


def plot_results(output_dir, effects, recovery):
    dino = [row for row in effects if row["scope"] == "combined" and row["encoder"] == "dino"]
    figure, axes = plt.subplots(2, 2, figsize=(14, 9))
    for column, effect in enumerate(("delta_target", "delta_pull")):
        selected = [row for row in dino if row["effect"] == effect]
        labels = [f"{row['visual_mode']}\n{row['probe']}" for row in selected]
        means = [row["mean"] for row in selected]
        errors = np.asarray([
            [row["mean"] - row["bootstrap_ci_lower"] for row in selected],
            [row["bootstrap_ci_upper"] - row["mean"] for row in selected],
        ])
        axes[0, column].bar(np.arange(len(selected)), means, yerr=errors, capsize=4)
        axes[0, column].axhline(0, color="black", linewidth=1)
        axes[0, column].set_xticks(np.arange(len(selected)), labels)
        axes[0, column].set_title(effect.replace("_", " "))
    selected = [
        row for row in recovery if row["encoder"] == "dino"
        and row["probe"] == "linear_probe" and row["target_kind"] == "caption_source"
    ]
    for column, metric in enumerate(("macro_f1", "mrr")):
        labels = [f"{row['visual_mode']}\n{row['prompt_condition']}" for row in selected]
        axes[1, column].bar(np.arange(len(selected)), [row[metric] for row in selected])
        axes[1, column].set_xticks(np.arange(len(selected)), labels)
        axes[1, column].set_title(f"Caption-target {metric}")
    for axis in axes.flat:
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("P4: Does frozen SDXL execute cluster-specific DCS text?")
    figure.tight_layout()
    figure.savefig(Path(output_dir) / "p4_text_execution.png", dpi=180)
    plt.close(figure)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared_dir = Path(args.prepared_dir)
    samples, paired = inventory(prepared_dir, args.generation_manifest)
    with (prepared_dir / "frozen_real_image_probes.pkl").open("rb") as handle:
        probes = pickle.load(handle)
    if probes.get("training_data") != "real_images_only":
        raise RuntimeError("P4 probes are not marked real_images_only")

    all_effects = []
    all_recovery = []
    for encoder, model_root in (("dino", args.dino_model), ("clip", args.clip_model)):
        expected_model_root = Path(
            probes["encoders"][encoder]["model_root"]
        ).resolve()
        if Path(model_root).resolve() != expected_model_root:
            raise RuntimeError(
                f"{encoder} model differs from the model frozen with the real probes: "
                f"{Path(model_root).resolve()} vs {expected_model_root}"
            )
        features = load_or_extract(args, samples, encoder, model_root)
        effects, recovery = paired_metrics(samples, paired, features, probes, encoder)
        all_effects.extend(effects)
        all_recovery.extend(recovery)
    effect_summary = summarize_effects(
        all_effects, args.bootstrap_samples, args.random_seed
    )
    recovery_per_class, recovery_summary = summarize_recovery(all_recovery)
    write_csv(output_dir / "paired_effects_raw.csv", all_effects)
    write_csv(output_dir / "paired_effects_summary.csv", effect_summary)
    write_csv(output_dir / "cluster_recoverability_raw.csv", all_recovery)
    write_csv(output_dir / "cluster_recoverability_per_class_seed.csv", recovery_per_class)
    write_csv(output_dir / "cluster_recoverability_summary.csv", recovery_summary)
    plot_results(output_dir, effect_summary, recovery_summary)

    primary = {}
    for visual_mode in ("i0g0", "i1g0"):
        selected = [
            row for row in effect_summary
            if row["scope"] == "combined" and row["encoder"] == "dino"
            and row["visual_mode"] == visual_mode
        ]
        primary[visual_mode] = {
            f"{row['probe']}_{row['effect']}": row for row in selected
        }
        primary[visual_mode]["strict_pass"] = all(
            row["mean"] > 0 and row["bootstrap_ci_lower"] > 0
            for row in selected
            if row["effect"] in {"delta_target", "delta_pull"}
        )
    atomic_json(
        output_dir / "summary.json",
        {
            "format_version": 1,
            "primary": primary,
            "probe_training_data": "real_images_only; no generated-image refitting",
            "pairing": "same image seed within visual mode and cluster",
            "bootstrap_unit": "class_key x visual_cluster_id after averaging generation seeds",
            "interpretation_boundary": (
                "P4 tests whether correct or swapped DCS text changes frozen SDXL outputs "
                "toward real cluster identities without continuous image guidance. It does "
                "not test downstream training value."
            ),
        },
    )
    print(f"P4 evaluation complete: {output_dir}")


if __name__ == "__main__":
    main()
