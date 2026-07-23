import json
import tempfile
import unittest
from pathlib import Path

from common import condition_matrix, ensure_manifest, stable_image_seed
from prepare_imagenette import materialize
from summarize_results import parse_log


class CommonTests(unittest.TestCase):
    def test_condition_matrix_has_exact_four_cells(self):
        matrix = condition_matrix("base", "tuned")
        self.assertEqual(
            [item["condition"] for item in matrix],
            ["frozen_label", "frozen_dcs", "finetuned_label", "finetuned_dcs"],
        )

    def test_stable_image_seed_is_deterministic_and_distinct(self):
        self.assertEqual(stable_image_seed(1, 2, 3), stable_image_seed(1, 2, 3))
        self.assertNotEqual(stable_image_seed(1, 2, 3), stable_image_seed(1, 2, 4))

    def test_manifest_rejects_changed_configuration(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "condition"
            ensure_manifest(output, {"seed": 0})
            ensure_manifest(output, {"seed": 0}, resume=True)
            with self.assertRaises(RuntimeError):
                ensure_manifest(output, {"seed": 1}, resume=True)


class PreparationTests(unittest.TestCase):
    def test_hardlink_materialization_is_idempotent(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.txt"
            destination = root / "nested" / "destination.txt"
            source.write_text("image", encoding="utf-8")
            self.assertTrue(materialize(source, destination, "hardlink"))
            self.assertFalse(materialize(source, destination, "hardlink"))


class SummaryTests(unittest.TestCase):
    def test_parse_minimax_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "result.log"
            path.write_text(
                "noise\n(Repeat 3) Best, last acc:----[77.2, 78.0, 76.8] 77.3 0.5\n",
                encoding="utf-8",
            )
            self.assertEqual(parse_log(path), [77.2, 78.0, 76.8])


if __name__ == "__main__":
    unittest.main()
