import unittest

import numpy as np

from diagnose_cross_modal_recombination import (
    aggregate_rows,
    projection_metrics,
    quadratic_fit,
)


class ProjectionMetricsTest(unittest.TestCase):
    def test_midpoint_is_mixup_like(self):
        source = np.asarray([1.0, 0.0])
        target = np.asarray([0.0, 1.0])
        generated = np.asarray([0.5, 0.5])
        correct = source.copy()

        result = projection_metrics(source, target, generated, correct)

        self.assertAlmostEqual(result["tau_from_source"], 0.5)
        self.assertAlmostEqual(result["tau_from_correct"], 0.5)
        self.assertAlmostEqual(result["residual_from_source"], 0.0)
        self.assertEqual(result["between_source_and_target"], 1.0)
        self.assertEqual(result["positive_caption_pull"], 1.0)

    def test_orthogonal_change_is_not_caption_pull(self):
        source = np.asarray([1.0, 0.0, 0.0])
        target = np.asarray([0.0, 1.0, 0.0])
        correct = np.asarray([1.0, 0.0, 0.0])
        generated = np.asarray([1.0, 0.0, 1.0])

        result = projection_metrics(source, target, generated, correct)

        self.assertAlmostEqual(result["tau_from_correct"], 0.0)
        self.assertAlmostEqual(result["direction_cosine_from_correct"], 0.0)
        self.assertGreater(result["residual_from_correct"], 0.0)
        self.assertEqual(result["positive_caption_pull"], 0.0)

    def test_aggregate_uses_equal_observation_weights(self):
        template = {
            field: 0.0
            for field in (
                "pair_cosine_distance",
                "pair_euclidean_distance",
                "generated_source_cosine_distance",
                "generated_target_cosine_distance",
                "correct_source_cosine_distance",
                "correct_target_cosine_distance",
                "tau_from_source",
                "tau_from_correct",
                "projection_from_source",
                "projection_from_correct",
                "direction_cosine_from_source",
                "direction_cosine_from_correct",
                "residual_from_source",
                "relative_residual_from_source",
                "residual_from_correct",
                "relative_residual_from_correct",
                "shuffled_correct_feature_distance",
                "target_cosine_improvement_vs_correct",
                "source_cosine_change_vs_correct",
                "target_distance_improvement_vs_correct",
                "source_distance_change_vs_correct",
                "between_source_and_target",
                "positive_caption_pull",
            )
        }
        rows = [
            {"shuffle_shift": 1, **template, "tau_from_correct": 0.25},
            {"shuffle_shift": 1, **template, "tau_from_correct": 0.75},
        ]

        result = aggregate_rows(rows, ("shuffle_shift",))

        self.assertEqual(result[0]["observations"], 2)
        self.assertAlmostEqual(result[0]["tau_from_correct"], 0.5)

    def test_quadratic_fit_detects_inverted_u(self):
        x_values = [-2.0, -1.0, 0.0, 1.0, 2.0]
        y_values = [0.0, 3.0, 4.0, 3.0, 0.0]

        result = quadratic_fit(x_values, y_values)

        self.assertLess(result["quadratic"], 0.0)
        self.assertAlmostEqual(result["r_squared"], 1.0)


if __name__ == "__main__":
    unittest.main()
