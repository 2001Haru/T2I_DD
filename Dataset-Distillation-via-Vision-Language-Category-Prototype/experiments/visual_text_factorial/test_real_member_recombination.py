import unittest

import numpy as np

from diagnose_real_member_recombination import (
    class_occupancy,
    cluster_size_percentiles,
    evaluate_heldout_member,
    grouped_assignments,
    intuitive_pair_metrics,
    mean_anchor,
)


class RealMemberAnchorTest(unittest.TestCase):
    def test_grouped_assignments_accepts_audit_column_name(self):
        rows = [
            {
                "synset": "class_a",
                "assigned_cluster": "2",
                "center_rmse": "0.5",
                "image_path": "example.png",
            }
        ]

        result = grouped_assignments(rows)

        self.assertIn(("class_a", 2), result)
        self.assertEqual(result[("class_a", 2)][0]["cluster_index"], 2)

    def test_mean_anchor_is_normalized(self):
        anchor = mean_anchor(
            np.asarray(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                ]
            )
        )

        self.assertAlmostEqual(float(np.linalg.norm(anchor)), 1.0)
        self.assertAlmostEqual(anchor[0], anchor[1])

    def test_heldout_retrieval_uses_own_anchor(self):
        anchors = {
            0: np.asarray([1.0, 0.0]),
            1: np.asarray([0.0, 1.0]),
        }

        result = evaluate_heldout_member(
            np.asarray([0.9, 0.1]),
            own_index=0,
            class_anchors=anchors,
        )

        self.assertEqual(result["retrieval_correct"], 1.0)
        self.assertEqual(result["retrieval_rank"], 1)
        self.assertGreater(result["own_anchor_margin"], 0.0)

    def test_cluster_size_percentile_orders_small_to_large(self):
        grouped = {
            ("class_a", 0): [{}] * 3,
            ("class_a", 1): [{}] * 7,
            ("class_a", 2): [{}] * 5,
        }

        result = cluster_size_percentiles(grouped)

        self.assertEqual(result[("class_a", 0)], 0.0)
        self.assertEqual(result[("class_a", 2)], 0.5)
        self.assertEqual(result[("class_a", 1)], 1.0)

    def test_class_occupancy_preserves_independent_class_unit(self):
        grouped = {
            ("class_a", 0): [{}] * 3,
            ("class_a", 1): [{}] * 7,
        }

        result = class_occupancy(grouped)["class_a"]

        self.assertEqual(result["minimum_cluster_size"], 3)
        self.assertEqual(result["maximum_cluster_size"], 7)
        self.assertAlmostEqual(result["minimum_cluster_fraction"], 0.3)
        self.assertAlmostEqual(result["maximum_to_minimum_cluster_size"], 7 / 3)

    def test_intuitive_metrics_distinguish_caption_and_visual_anchors(self):
        visual = np.asarray([1.0, 0.0])
        caption = np.asarray([0.0, 1.0])
        correct = visual.copy()
        shuffled = np.asarray([0.5, 0.5])
        metrics = {
            "projection_from_correct": 1.0,
            "pair_euclidean_distance": np.sqrt(2.0),
            "residual_from_correct": 0.0,
        }

        result = intuitive_pair_metrics(
            metrics,
            visual,
            caption,
            shuffled,
            correct,
        )

        self.assertGreater(result["caption_source_similarity_gain"], 0.0)
        self.assertLess(result["visual_target_similarity_change"], 0.0)
        self.assertGreater(result["unit_caption_pull_projection"], 0.0)
        self.assertEqual(result["off_axis_fraction"], 0.0)


if __name__ == "__main__":
    unittest.main()
