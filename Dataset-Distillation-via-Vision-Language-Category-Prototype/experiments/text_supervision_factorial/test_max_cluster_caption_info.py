import sys
import importlib.util
from pathlib import Path

import numpy as np
import pytest


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from measure_max_cluster_caption_info import (  # noqa: E402
    crossfit_a_predictions,
    fit_predict_by_class,
    select_top_k,
    selection_metrics,
    split_half,
)


def synthetic_cluster_data():
    rows, labels, features = [], [], []
    rng = np.random.default_rng(4)
    for class_index, synset in enumerate(("class_a", "class_b")):
        for cluster in range(3):
            center = np.zeros(8)
            center[class_index * 4 + cluster] = 5
            for image in range(8):
                rows.append({
                    "record_id": f"{synset}/{cluster}_{image}.jpg",
                    "relative": f"{synset}/{cluster}_{image}.jpg",
                    "dataset": "fixture", "synset": synset,
                    "text": f"caption {synset} cluster {cluster} image {image}",
                })
                labels.append(f"{synset}:{cluster}")
                features.append(center + rng.normal(0, 0.05, size=8))
    return rows, labels, np.asarray(features)


def test_half_split_is_disjoint_and_cluster_stratified():
    rows, labels, _ = synthetic_cluster_data()
    a_indices, b_indices, membership = split_half(rows, labels, seed=7, minimum_cluster_size=4)
    assert len(a_indices) == len(b_indices) == 24
    assert not (set(a_indices) & set(b_indices))
    counts = {}
    for row in membership:
        counts.setdefault((row["cluster_label"], row["split"]), 0)
        counts[(row["cluster_label"], row["split"])] += 1
    assert set(counts.values()) == {4}


@pytest.mark.skipif(importlib.util.find_spec("sklearn") is None, reason="scikit-learn unavailable")
def test_a_selected_probe_and_top_k_transfer_to_b():
    rows, labels, features = synthetic_cluster_data()
    a_indices, b_indices, _ = split_half(rows, labels, seed=7, minimum_cluster_size=4)
    a_predictions = crossfit_a_predictions(
        features, rows, labels, a_indices, folds=2,
        c_value=1.0, seed=9, max_iter=500,
    )
    selected_a = select_top_k(rows, a_predictions, a_indices, k=2)
    a_metrics, _ = selection_metrics(rows, labels, a_predictions, a_indices, selected_a)
    assert a_metrics["normalized_mi"] > 0.95

    b_predictions = fit_predict_by_class(
        features, rows, labels, a_indices, b_indices, c_value=1.0, max_iter=500
    )
    selected_b = select_top_k(rows, b_predictions, b_indices, k=2)
    b_metrics, _ = selection_metrics(rows, labels, b_predictions, b_indices, selected_b)
    assert b_metrics["normalized_mi"] > 0.95
    assert b_metrics["adjusted_mi"] > 0.9
