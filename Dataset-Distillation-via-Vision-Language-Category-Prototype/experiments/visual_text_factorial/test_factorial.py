import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from analyze_downstream_per_class import (
    aggregate_generation_seeds,
    pearson,
    rankdata,
    spearman,
)
from common import (
    condition_matrix,
    ensure_manifest,
    shuffled_prompt_index,
    stable_image_seed,
)
from diagnose_dino_coverage import class_metrics, condition_metadata
from diagnose_text_conditioning import pair_metrics
from summarize_results import condition_contrasts, parse_log
from summarize_shuffle_randomization import parse_shift_runs, rank_descending


class CommonTests(unittest.TestCase):
    def test_condition_matrix_has_exact_six_cells(self):
        self.assertEqual(
            [item["condition"] for item in condition_matrix()],
            [
                "no_visual_label",
                "no_visual_dcs",
                "no_visual_dcs_shuffled",
                "prototype_label",
                "prototype_dcs",
                "prototype_dcs_shuffled",
            ],
        )

    def test_cyclic_shuffle_is_a_derangement(self):
        mapped = [shuffled_prompt_index(index, 10, shift=1) for index in range(10)]
        self.assertEqual(mapped, [1, 2, 3, 4, 5, 6, 7, 8, 9, 0])
        self.assertTrue(all(index != mapped[index] for index in range(10)))

    def test_stable_image_seed_is_paired_and_distinct(self):
        self.assertEqual(stable_image_seed(1, 2, 3), stable_image_seed(1, 2, 3))
        self.assertNotEqual(stable_image_seed(1, 2, 3), stable_image_seed(1, 2, 4))

    def test_manifest_rejects_changed_configuration(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "condition"
            ensure_manifest(output, {"seed": 0})
            ensure_manifest(output, {"seed": 0}, resume=True)
            with self.assertRaises(RuntimeError):
                ensure_manifest(output, {"seed": 1}, resume=True)


class SummaryTests(unittest.TestCase):
    def test_parse_minimax_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "result.log"
            path.write_text(
                "noise\n(Repeat 3) Best, last acc:----[77.2, 78.0, 76.8] 77.3 0.5\n",
                encoding="utf-8",
            )
            self.assertEqual(parse_log(path), [77.2, 78.0, 76.8])

    def test_primary_cluster_correspondence_contrast(self):
        values = {
            "no_visual_label": [50.0, 51.0],
            "no_visual_dcs": [54.0, 55.0],
            "no_visual_dcs_shuffled": [52.0, 53.0],
            "prototype_label": [60.0, 61.0],
            "prototype_dcs": [65.0, 66.0],
            "prototype_dcs_shuffled": [61.0, 62.0],
        }
        contrasts = condition_contrasts(values)
        self.assertEqual(contrasts["no_visual_dcs_minus_shuffled"], [2.0, 2.0])
        self.assertEqual(contrasts["prototype_dcs_minus_shuffled"], [4.0, 4.0])
        self.assertEqual(
            contrasts["visual_x_cluster_correspondence_interaction"], [2.0, 2.0]
        )

    def test_shuffle_run_parser_and_rank(self):
        runs = parse_shift_runs(["2=/tmp/shift2", "7=/tmp/shift7"])
        self.assertEqual(sorted(runs), [2, 7])
        self.assertEqual(rank_descending(57.0, [57.0, 59.0, 56.0, 55.0]), 2)


class DiagnosticTests(unittest.TestCase):
    def test_condition_metadata_preserves_no_visual_prefix(self):
        self.assertEqual(
            condition_metadata("no_visual_dcs_shuffled", 4),
            ("no_visual", "shuffled_dcs", 4),
        )
        self.assertEqual(
            condition_metadata("prototype_dcs", 1),
            ("prototype", "correct_dcs", 0),
        )

    def test_identical_conditioning_has_zero_displacement(self):
        encoding = {
            "hidden": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            "mean_hidden": torch.tensor([0.5, 0.5]),
            "content_token_ids": [10, 11],
            "token_count": 4,
            "chunk_count": 1,
        }
        metrics = pair_metrics(encoding, encoding)
        self.assertAlmostEqual(metrics["symmetric_relative_l2"], 0.0)
        self.assertAlmostEqual(metrics["mean_hidden_cosine"], 1.0, places=6)
        self.assertAlmostEqual(metrics["token_jaccard"], 1.0)

    def test_class_metrics_reward_exact_real_coverage(self):
        real = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        exact = class_metrics(real, real, real_radius=0.1)
        collapsed = class_metrics(real, real[:1], real_radius=0.1)
        self.assertAlmostEqual(exact["real_to_synthetic_nn_mean"], 0.0)
        self.assertGreater(
            collapsed["real_to_synthetic_nn_mean"],
            exact["real_to_synthetic_nn_mean"],
        )
        self.assertGreater(
            exact["synthetic_pairwise_distance_mean"],
            collapsed["synthetic_pairwise_distance_mean"],
        )

    def test_rank_correlations_handle_ties(self):
        self.assertEqual(rankdata([3.0, 1.0, 1.0]), [3.0, 1.5, 1.5])
        self.assertAlmostEqual(pearson([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]), 1.0)
        self.assertAlmostEqual(spearman([1.0, 3.0, 2.0], [2.0, 6.0, 4.0]), 1.0)

    def test_downstream_rows_average_generation_seeds(self):
        template = {
            "visual_mode": "prototype",
            "shuffle_shift": 1,
            "synset": "n00000001",
            "correct_mean_accuracy": 50.0,
            "shuffled_mean_accuracy": 52.0,
            "downstream_gain_std_over_classifier_repeats": 1.0,
            "coverage_distance_improvement": 0.1,
            "fidelity_distance_improvement": 0.2,
            "coverage_fraction_change": 0.3,
            "precision_fraction_change": 0.4,
            "diversity_change": 0.5,
            "centroid_distance_improvement": 0.6,
            "conditioning_relative_l2": 1.0,
            "conditioning_mean_hidden_cosine": 0.8,
            "conditioning_token_jaccard": 0.4,
        }
        rows = [
            {**template, "generation_seed": 0, "downstream_gain": 2.0},
            {**template, "generation_seed": 1, "downstream_gain": 4.0},
        ]
        aggregated = aggregate_generation_seeds(rows)
        self.assertEqual(len(aggregated), 1)
        self.assertEqual(aggregated[0]["generation_seeds"], 2)
        self.assertEqual(aggregated[0]["downstream_gain"], 3.0)


if __name__ == "__main__":
    unittest.main()
