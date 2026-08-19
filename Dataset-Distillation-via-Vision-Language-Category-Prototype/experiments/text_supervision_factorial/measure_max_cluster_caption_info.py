#!/usr/bin/env python3
"""Estimate maximum transferable caption-to-cluster information by sample splitting."""

import argparse
import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from audit_caption_interface import (
    load_cluster_labels,
    load_or_encode,
    parse_named_paths,
    read_jsonl_captions,
    write_csv,
)


REPRESENTATIONS = ("train_t77_mean_hidden", "train_t77_pooled")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--dataset", action="append", default=[], metavar="NAME=CAPTIONS.jsonl")
    parser.add_argument("--assignment", action="append", default=[], metavar="NAME=ASSIGNMENTS.csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--feature-cache-dir", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--split-seed", type=int, default=20260819)
    parser.add_argument("--inner-folds", type=int, default=2)
    parser.add_argument("--candidate-k", type=int, nargs="+", default=(1, 2, 4, 8, 16))
    parser.add_argument("--candidate-c", type=float, nargs="+", default=(0.1, 1.0, 10.0))
    parser.add_argument("--minimum-cluster-size", type=int, default=4)
    parser.add_argument("--max-iter", type=int, default=3000)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--force-features", action="store_true")
    return parser.parse_args()


def stable_seed(seed, key):
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


def split_half(rows, cluster_labels, seed, minimum_cluster_size):
    grouped = defaultdict(list)
    for index, label in enumerate(cluster_labels):
        grouped[label].append(index)
    a_indices, b_indices, excluded = [], [], []
    membership = []
    for label in sorted(grouped):
        indices = sorted(grouped[label], key=lambda index: rows[index]["relative"])
        if len(indices) < minimum_cluster_size:
            excluded.extend(indices)
            for index in indices:
                membership.append({
                    "record_id": rows[index]["record_id"], "dataset": rows[index]["dataset"],
                    "synset": rows[index]["synset"], "cluster_label": label,
                    "split": "excluded_sparse_cluster", "cluster_size": len(indices),
                })
            continue
        random.Random(stable_seed(seed, label)).shuffle(indices)
        b_count = len(indices) // 2
        b_group = indices[:b_count]
        a_group = indices[b_count:]
        if len(a_group) < 2 or len(b_group) < 2:
            raise AssertionError(f"Eligible cluster did not retain two examples per half: {label}")
        a_indices.extend(a_group)
        b_indices.extend(b_group)
        for split, selected in (("A", a_group), ("B", b_group)):
            for index in selected:
                membership.append({
                    "record_id": rows[index]["record_id"], "dataset": rows[index]["dataset"],
                    "synset": rows[index]["synset"], "cluster_label": label,
                    "split": split, "cluster_size": len(indices),
                })
    if set(a_indices) & set(b_indices):
        raise AssertionError("A/B split overlaps")
    return np.asarray(sorted(a_indices)), np.asarray(sorted(b_indices)), membership


def fit_predict_by_class(
    features, rows, cluster_labels, train_indices, test_indices, c_value, max_iter,
):
    from sklearn.linear_model import LogisticRegression

    output = {}
    train_by_class = defaultdict(list)
    test_by_class = defaultdict(list)
    for index in train_indices:
        train_by_class[rows[index]["synset"]].append(int(index))
    for index in test_indices:
        test_by_class[rows[index]["synset"]].append(int(index))
    for synset in sorted(test_by_class):
        train = np.asarray(train_by_class[synset])
        test = np.asarray(test_by_class[synset])
        train_labels = np.asarray([cluster_labels[index] for index in train])
        if len(set(train_labels)) < 2:
            raise ValueError(f"Not enough clusters to fit probe for {synset}")
        model = LogisticRegression(
            C=c_value, max_iter=max_iter, class_weight="balanced",
            solver="lbfgs", random_state=0,
        )
        model.fit(features[train], train_labels)
        probabilities = model.predict_proba(features[test])
        predictions = model.classes_[probabilities.argmax(axis=1)]
        class_to_column = {label: column for column, label in enumerate(model.classes_)}
        for position, index in enumerate(test):
            output[int(index)] = {
                "predicted_cluster": str(predictions[position]),
                "confidence": float(probabilities[position].max()),
                "probability_by_cluster": {
                    str(label): float(probabilities[position, column])
                    for label, column in class_to_column.items()
                },
                "cluster_count": len(model.classes_),
            }
    if set(output) != set(map(int, test_indices)):
        raise AssertionError("Probe predictions do not cover the requested test indices")
    return output


def crossfit_a_predictions(
    features, rows, cluster_labels, a_indices, folds, c_value, seed, max_iter,
):
    from sklearn.model_selection import StratifiedKFold

    output = {}
    by_class = defaultdict(list)
    for index in a_indices:
        by_class[rows[index]["synset"]].append(int(index))
    for class_offset, synset in enumerate(sorted(by_class)):
        indices = np.asarray(by_class[synset])
        labels = np.asarray([cluster_labels[index] for index in indices])
        counts = Counter(labels.tolist())
        if min(counts.values()) < folds:
            raise ValueError(f"A split cannot support {folds}-fold cross-fit in {synset}: {counts}")
        splitter = StratifiedKFold(
            n_splits=folds, shuffle=True, random_state=seed + class_offset
        )
        for train_position, test_position in splitter.split(features[indices], labels):
            fold_output = fit_predict_by_class(
                features, rows, cluster_labels,
                indices[train_position], indices[test_position],
                c_value, max_iter,
            )
            overlap = set(output) & set(fold_output)
            if overlap:
                raise AssertionError(f"Cross-fit predictions overlap: {overlap}")
            output.update(fold_output)
    if set(output) != set(map(int, a_indices)):
        raise AssertionError("A cross-fit did not predict every eligible A caption")
    return output


def select_top_k(rows, predictions, indices, k):
    groups = defaultdict(list)
    for index in indices:
        prediction = predictions[int(index)]
        groups[(rows[index]["synset"], prediction["predicted_cluster"])].append(int(index))
    selected = []
    for key in sorted(groups):
        ranked = sorted(
            groups[key],
            key=lambda index: (-predictions[index]["confidence"], rows[index]["relative"]),
        )
        selected.extend(ranked[:k])
    return np.asarray(sorted(selected))


def selection_metrics(rows, cluster_labels, predictions, eligible_indices, selected_indices):
    from sklearn.metrics import (
        accuracy_score, adjusted_mutual_info_score, balanced_accuracy_score, f1_score,
        normalized_mutual_info_score,
    )

    by_class = defaultdict(list)
    for index in selected_indices:
        by_class[rows[index]["synset"]].append(int(index))
    class_metrics = []
    for synset in sorted(by_class):
        indices = by_class[synset]
        true = [cluster_labels[index] for index in indices]
        predicted = [predictions[index]["predicted_cluster"] for index in indices]
        true_probabilities = [
            predictions[index]["probability_by_cluster"].get(cluster_labels[index], 0.0)
            for index in indices
        ]
        class_metrics.append({
            "synset": synset, "selected": len(indices),
            "top1": accuracy_score(true, predicted),
            "balanced_accuracy": balanced_accuracy_score(true, predicted),
            "macro_f1": f1_score(true, predicted, average="macro", zero_division=0),
            "normalized_mi": normalized_mutual_info_score(true, predicted),
            "adjusted_mi": adjusted_mutual_info_score(true, predicted),
            "mean_true_cluster_probability": statistics_fmean(true_probabilities),
        })
    if not class_metrics:
        raise ValueError("Selector retained no captions")
    return {
        "selected_captions": len(selected_indices),
        "eligible_captions": len(eligible_indices),
        "coverage": len(selected_indices) / len(eligible_indices),
        "source_classes": len(class_metrics),
        **{
            metric: statistics_fmean(row[metric] for row in class_metrics)
            for metric in (
                "top1", "balanced_accuracy", "macro_f1", "normalized_mi",
                "adjusted_mi", "mean_true_cluster_probability",
            )
        },
    }, class_metrics


def statistics_fmean(values):
    values = list(values)
    return float(sum(values) / len(values))


def digest_membership(rows, indices):
    digest = hashlib.sha256()
    for index in sorted(map(int, indices)):
        digest.update(rows[index]["record_id"].encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def main():
    args = parse_args()
    datasets = parse_named_paths(args.dataset, "--dataset")
    assignments = parse_named_paths(args.assignment, "--assignment")
    if set(datasets) != set(assignments):
        raise ValueError("--dataset and --assignment names must match")
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache = Path(args.feature_cache_dir).resolve() if args.feature_cache_dir else output / "feature_cache"
    cache.mkdir(parents=True, exist_ok=True)

    contexts = {}
    grid_rows = []
    membership_rows = []
    locks = {}
    for dataset in sorted(datasets):
        rows = read_jsonl_captions(dataset, datasets[dataset])
        cluster_labels, reason = load_cluster_labels(assignments[dataset], rows)
        if cluster_labels is None:
            raise RuntimeError(f"Cannot load {dataset} cluster labels: {reason}")
        features_by_mode = load_or_encode(rows, dataset, args, cache)
        a_indices, b_indices, membership = split_half(
            rows, cluster_labels, args.split_seed, args.minimum_cluster_size
        )
        membership_rows.extend(membership)
        best = None
        for representation in REPRESENTATIONS:
            features = features_by_mode[representation]
            for c_value in args.candidate_c:
                predictions = crossfit_a_predictions(
                    features, rows, cluster_labels, a_indices,
                    args.inner_folds, c_value, args.split_seed, args.max_iter,
                )
                for k in sorted(set(args.candidate_k)):
                    selected = select_top_k(rows, predictions, a_indices, k)
                    metrics, _ = selection_metrics(
                        rows, cluster_labels, predictions, a_indices, selected
                    )
                    candidate = {
                        "dataset": dataset, "representation": representation,
                        "probe_c": c_value, "top_k_per_predicted_cluster": k,
                        "selection_split": "A_cross_fitted", **metrics,
                    }
                    grid_rows.append(candidate)
                    rank = (
                        candidate["normalized_mi"], candidate["coverage"],
                        candidate["top1"], -k,
                    )
                    if best is None or rank > best[0]:
                        best = (rank, candidate)
        selected_config = best[1]
        locks[dataset] = {
            **selected_config,
            "split_seed": args.split_seed,
            "minimum_cluster_size": args.minimum_cluster_size,
            "inner_folds": args.inner_folds,
            "a_records": len(a_indices), "b_records": len(b_indices),
            "a_membership_sha256": digest_membership(rows, a_indices),
            "b_membership_sha256": digest_membership(rows, b_indices),
            "selection_used_b_labels": False,
        }
        contexts[dataset] = {
            "rows": rows, "cluster_labels": cluster_labels,
            "features": features_by_mode[selected_config["representation"]],
            "a_indices": a_indices, "b_indices": b_indices,
        }

    lock_path = output / "selection_lock.json"
    lock_path.write_text(
        json.dumps({
            "format_version": 1,
            "locked_before_b_evaluation": True,
            "selection_object": (
                "representation, logistic-regression C, and top-k per predicted cluster; "
                "all chosen exclusively from cross-fitted A predictions"
            ),
            "datasets": locks,
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    heldout_rows = []
    per_class_rows = []
    selected_caption_rows = []
    for dataset in sorted(contexts):
        context = contexts[dataset]
        config = locks[dataset]
        predictions = fit_predict_by_class(
            context["features"], context["rows"], context["cluster_labels"],
            context["a_indices"], context["b_indices"],
            config["probe_c"], args.max_iter,
        )
        selected = select_top_k(
            context["rows"], predictions, context["b_indices"],
            config["top_k_per_predicted_cluster"],
        )
        selected_metrics, class_metrics = selection_metrics(
            context["rows"], context["cluster_labels"], predictions,
            context["b_indices"], selected,
        )
        all_metrics, _ = selection_metrics(
            context["rows"], context["cluster_labels"], predictions,
            context["b_indices"], context["b_indices"],
        )
        heldout_rows.append({
            "dataset": dataset, "evaluation_split": "B_heldout",
            "representation_selected_on_a": config["representation"],
            "probe_c_selected_on_a": config["probe_c"],
            "top_k_selected_on_a": config["top_k_per_predicted_cluster"],
            **{f"selected_{key}": value for key, value in selected_metrics.items()},
            **{f"all_b_{key}": value for key, value in all_metrics.items()},
            "selection_lock": str(lock_path),
        })
        per_class_rows.extend({"dataset": dataset, "split": "B_heldout", **row} for row in class_metrics)
        for index in selected:
            row = context["rows"][index]
            prediction = predictions[int(index)]
            true_probability_raw = prediction["probability_by_cluster"].get(
                context["cluster_labels"][index], 0.0
            )
            true_probability = max(true_probability_raw, 1e-12)
            information_gain = math.log(true_probability * prediction["cluster_count"])
            selected_caption_rows.append({
                "dataset": dataset, "record_id": row["record_id"],
                "relative": row["relative"], "synset": row["synset"],
                "true_cluster": context["cluster_labels"][index],
                "predicted_cluster": prediction["predicted_cluster"],
                "confidence": prediction["confidence"],
                "true_cluster_probability": true_probability_raw,
                "information_gain_over_uniform_nats": information_gain,
                "correct": int(
                    context["cluster_labels"][index] == prediction["predicted_cluster"]
                ),
                "caption": row["text"],
            })

    write_csv(output / "a_selection_grid.csv", grid_rows)
    write_csv(output / "split_membership.csv", membership_rows)
    write_csv(output / "heldout_b_summary.csv", heldout_rows)
    write_csv(output / "heldout_b_per_class.csv", per_class_rows)
    write_csv(output / "heldout_b_selected_captions.csv", selected_caption_rows)
    summary = {
        "format_version": 1,
        "primary_estimand": (
            "class-macro normalized mutual information on B after representation, probe C, "
            "and top-k selector are frozen using only cross-fitted A predictions"
        ),
        "selection_bias_control": (
            "B cluster labels are not used for model fitting, hyperparameter selection, or "
            "caption selection; they are revealed only after selection_lock.json is written"
        ),
        "heldout_b": heldout_rows,
        "selection_lock": str(lock_path),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Saved sample-split maximum cluster-caption information to {output}")


if __name__ == "__main__":
    main()
