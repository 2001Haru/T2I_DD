import json
import tempfile
import unittest
from pathlib import Path

from common import (
    condition_matrix,
    ensure_manifest,
    shuffled_prompt_index,
    stable_image_seed,
)
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


if __name__ == "__main__":
    unittest.main()
