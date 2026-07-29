import unittest

import numpy as np
import torch

from diagnose_cluster_members import (
    nearest_center_assignments,
    nearest_other_center_distances,
    summarize_cluster,
)


class ClusterMemberAuditTest(unittest.TestCase):
    def test_nearest_assignment_uses_per_dimension_rmse(self):
        centers = torch.tensor(
            [
                [[[-1.0, -1.0]]],
                [[[1.0, 1.0]]],
            ]
        )
        latents = torch.tensor(
            [
                [[[-0.5, -0.5]]],
                [[[0.75, 0.75]]],
            ]
        )

        cluster, distance, second, margin = nearest_center_assignments(
            latents, centers
        )

        self.assertEqual(cluster.tolist(), [0, 1])
        self.assertTrue(np.allclose(distance.numpy(), [0.5, 0.25]))
        self.assertTrue(torch.all(second > distance))
        self.assertTrue(torch.all(margin > 0))

    def test_other_center_distance_excludes_self(self):
        centers = np.asarray(
            [
                [[[0.0, 0.0]]],
                [[[1.0, 1.0]]],
                [[[3.0, 3.0]]],
            ]
        )

        distances = nearest_other_center_distances(centers)

        self.assertTrue(np.allclose(distances, [1.0, 1.0, 2.0]))

    def test_summary_exposes_nearest_gap(self):
        rows = [
            {"center_rmse": value, "assignment_margin_rmse": 0.1}
            for value in (0.8, 0.9, 1.0, 1.1)
        ]

        summary = summarize_cluster(rows, nearest_other_center_rmse=2.0)

        self.assertAlmostEqual(summary["nearest_center_rmse"], 0.8)
        self.assertAlmostEqual(summary["median_center_rmse"], 0.95)
        self.assertAlmostEqual(
            summary["nearest_member_to_other_center_ratio"], 0.4
        )
        self.assertGreater(summary["nearest_to_median_ratio"], 0.8)


if __name__ == "__main__":
    unittest.main()
