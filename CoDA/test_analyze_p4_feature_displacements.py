import numpy as np
import pytest

from analyze_p4_feature_displacements import calculate_rows, cosine, target_specificity


def test_cosine_and_zero_vector_handling():
    assert cosine(np.asarray([1.0, 0.0]), np.asarray([1.0, 0.0])) == pytest.approx(1.0)
    assert np.isnan(cosine(np.zeros(2), np.ones(2)))


def test_target_specificity_uses_strongest_alternative():
    displacement = np.asarray([1.0, 0.0])
    label = np.asarray([0.0, 0.0])
    ids = np.asarray([0, 1, 2])
    centroids = np.asarray([[1.0, 0.0], [0.8, 0.6], [-1.0, 0.0]])
    alignment, specificity = target_specificity(displacement, label, ids, centroids, 0)
    assert alignment == pytest.approx(1.0)
    assert specificity == pytest.approx(0.2)


def test_target_specificity_requires_target_centroid():
    with pytest.raises(ValueError, match="lacks target cluster"):
        target_specificity(
            np.asarray([1.0, 0.0]), np.zeros(2),
            np.asarray([0, 1]), np.asarray([[1.0, 0.0], [0.0, 1.0]]), 2,
        )


def test_calculate_rows_keeps_correct_and_swap_displacements_separate():
    record = {
        "spec": "imageA", "class_key": "imageA:n1", "class_id": "n1",
        "class_name": "class", "visual_cluster_id": 0,
        "shuffled_caption_cluster_id": 1, "generation_seed": 0, "image_seed": 10,
        "features": {
            ("i0g0", "label"): np.asarray([1.0, 0.0]),
            ("i0g0", "correct"): np.asarray([0.8, 0.2]),
            ("i0g0", "shuffled"): np.asarray([0.6, 0.4]),
            ("i1g0", "label"): np.asarray([0.9, 0.1]),
            ("i1g0", "correct"): np.asarray([0.8, 0.2]),
            ("i1g0", "shuffled"): np.asarray([0.5, 0.5]),
        },
    }
    probes = {
        "encoders": {"dino": {"classes": {"imageA:n1": {
            "centroid_cluster_ids": np.asarray([0, 1]),
            "centroids": np.asarray([[1.0, 0.0], [0.0, 1.0]]),
        }}}}
    }
    rows, vectors = calculate_rows([record], probes)
    assert len(rows) == len(vectors) == 1
    assert rows[0]["text_norm_i1g0"] < rows[0]["text_norm_i0g0"]
    assert rows[0]["swap_norm_i1g0"] > rows[0]["swap_norm_i0g0"]
    assert "features" not in rows[0]
