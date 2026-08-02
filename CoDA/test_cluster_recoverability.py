import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from diagnose_cluster_recoverability import (
    evaluate_class,
    load_partition,
    matched_random_labels,
)


class ClusterRecoverabilityTest(unittest.TestCase):
    def test_matched_random_preserves_exact_occupancy(self):
        labels = np.asarray([0] * 7 + [1] * 4 + [2] * 9, dtype=np.int64)
        shuffled = matched_random_labels(labels, np.random.default_rng(17))
        np.testing.assert_array_equal(
            np.bincount(shuffled), np.bincount(labels)
        )
        self.assertFalse(np.array_equal(labels, shuffled))

    def test_separable_features_exceed_matched_random_null(self):
        rng = np.random.default_rng(11)
        labels = np.repeat(np.arange(3), 30)
        features = np.eye(3, dtype=np.float32)[labels]
        features = features + rng.normal(0.0, 0.03, size=features.shape)
        features = features / np.linalg.norm(features, axis=1, keepdims=True)
        result = evaluate_class(
            "imageA:n00000001",
            "imageA",
            "n00000001",
            "toy",
            features.astype(np.float32),
            labels,
            5,
            20,
            20,
            123,
            1.0,
        )
        true_value = result["true"]["linear_probe"]["macro_f1"]
        null_mean = np.mean(
            [row["linear_probe"]["macro_f1"] for row in result["null"]]
        )
        self.assertGreater(true_value, 0.95)
        self.assertGreater(true_value - null_mean, 0.4)

    def test_partition_uses_reconstructed_masks_and_keeps_voronoi_audit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            misc = root / "misc"
            cluster_dir = root / "clusters" / "imageA"
            misc.mkdir(parents=True)
            cluster_dir.mkdir(parents=True)
            (misc / "class_indices.txt").write_text("n00000001\n", encoding="utf-8")
            (misc / "imagenet-a.txt").write_text("n00000001\n", encoding="utf-8")
            (misc / "class_names.txt").write_text("toy class\n", encoding="utf-8")

            features = [
                np.asarray([0.0, 0.1], dtype=np.float32),
                np.asarray([0.2, 0.0], dtype=np.float32),
                np.asarray([9.8, 10.0], dtype=np.float32),
                np.asarray([10.0, 9.7], dtype=np.float32),
            ]
            paths = [str(root / f"image_{index}.png") for index in range(4)]
            with (cluster_dir / "features.pkl_0").open("wb") as handle:
                pickle.dump(
                    {"features": {0: features}, "paths": {0: paths}}, handle
                )
            with (cluster_dir / "centers_0.pkl").open("wb") as handle:
                pickle.dump(
                    {
                        0: np.asarray(
                            [[0.0, 0.0], [10.0, 10.0]], dtype=np.float32
                        )
                    },
                    handle,
                )

            args = SimpleNamespace(
                specs=["imageA"],
                cluster_root=str(root / "clusters"),
                misc_dir=str(misc),
                nclass=1,
                phase=0,
                features_cache_name="features.pkl",
                saved_clusters_base_name="centers.pkl",
                ipc=2,
            )
            reconstruction = np.asarray([1, 1, 0, -1], dtype=np.int64)
            audit = {
                "cluster_counts": [1, 2],
                "assigned_images": 3,
                "unassigned_images": 1,
                "coverage_fraction": 0.75,
                "initial_hdbscan_clusters": 2,
                "maximum_representative_match_rmse": 0.0,
                "representative_match_rmse": [0.0, 0.0],
                "cluster_origins": ["hdbscan_initial", "hdbscan_initial"],
                "representative_source_indices": [2, 0],
            }
            with patch(
                "diagnose_cluster_recoverability.reconstruct_class_partition",
                return_value=(reconstruction, audit),
            ):
                samples, metadata = load_partition(args)
            self.assertEqual([row["cluster_id"] for row in samples], [1, 1, 0, -1])
            self.assertEqual(
                [row["voronoi_cluster_id"] for row in samples], [0, 0, 1, 1]
            )
            self.assertEqual(
                [row["included_in_original_partition"] for row in samples],
                [True, True, True, False],
            )
            self.assertEqual(metadata[0]["cluster_counts"], [1, 2])
            self.assertEqual(metadata[0]["voronoi_cluster_counts"], [2, 2])
            self.assertEqual(metadata[0]["unassigned_images"], 1)


if __name__ == "__main__":
    unittest.main()
