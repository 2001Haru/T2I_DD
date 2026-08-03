import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluate_p4_text_execution import margin_for, probe_scores, rank_for
from prepare_p4_text_execution import build_prompt_inputs


class P4PreparationTests(unittest.TestCase):
    def test_prompt_inputs_exclude_sparse_or_missing_clusters(self):
        assignments = []
        for cluster_id, count in ((0, 3), (1, 2), (2, 1)):
            for index in range(count):
                assignments.append(
                    {
                        "class_key": "imageA:n00000001",
                        "spec": "imageA",
                        "class_id": "n00000001",
                        "class_name": "example",
                        "cluster_id": cluster_id,
                        "included": True,
                    }
                )
        payload = {
            "classes": {
                "imageA:n00000001": {
                    "clusters": {
                        "0": {"caption": "first"},
                        "1": {"caption": "second"},
                        "2": {"caption": "sparse"},
                    }
                }
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            rows, eligible, excluded = build_prompt_inputs(
                assignments, payload, {"imageA"}, 1, Path(directory)
            )
            self.assertEqual(eligible["imageA:n00000001"], [0, 1])
            self.assertEqual([row["shuffled_caption_cluster_id"] for row in rows], [1, 0])
            self.assertEqual(excluded, [])
            indices = json.loads(
                (Path(directory) / "imageA_cluster_indices.json").read_text()
            )
            self.assertEqual(indices["n00000001"], [0, 1])


class P4MetricTests(unittest.TestCase):
    def test_centroid_margin_and_rank(self):
        payload = {
            "centroid_cluster_ids": np.asarray([2, 5, 8]),
            "centroids": np.eye(3, dtype=np.float32),
            "class_ids": np.asarray([2, 5, 8]),
            "scaler_mean": np.zeros(3, dtype=np.float32),
            "scaler_scale": np.ones(3, dtype=np.float32),
            "ridge_coef": np.eye(3, dtype=np.float32),
            "ridge_intercept": np.zeros(3, dtype=np.float32),
        }
        ids, scores = probe_scores(
            np.asarray([0.8, 0.2, 0.0], dtype=np.float32),
            payload,
            "nearest_centroid",
        )
        self.assertEqual(rank_for(ids, scores, 2), 1)
        self.assertAlmostEqual(margin_for(ids, scores, 2), scores[0] - scores[1])
        ids, scores = probe_scores(
            np.asarray([0.1, 0.2, 0.9], dtype=np.float32),
            payload,
            "linear_probe",
        )
        self.assertEqual(rank_for(ids, scores, 8), 1)


if __name__ == "__main__":
    unittest.main()
