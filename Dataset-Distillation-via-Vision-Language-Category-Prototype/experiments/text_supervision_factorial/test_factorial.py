import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import build_unpaired_donors, condition_matrix, shuffled_prompt_index  # noqa: E402


class AssignmentTests(unittest.TestCase):
    def test_matrix_has_eighteen_unique_cells(self):
        conditions = [item["condition"] for item in condition_matrix()]
        self.assertEqual(len(conditions), 18)
        self.assertEqual(len(set(conditions)), 18)

    def test_unpaired_is_deranged_and_preserves_each_class(self):
        groups = {"a": [0, 1, 2, 3], "b": [4, 5, 6]}
        donors = build_unpaired_donors(groups, seed=17, epoch=2)
        self.assertTrue(all(index != donor for index, donor in enumerate(donors)))
        for indices in groups.values():
            self.assertEqual(sorted(donors[index] for index in indices), sorted(indices))
        self.assertEqual(donors, build_unpaired_donors(groups, seed=17, epoch=2))

    def test_shuffle_shift(self):
        self.assertEqual([shuffled_prompt_index(index, 4, 1) for index in range(4)], [1, 2, 3, 0])


class SummaryTests(unittest.TestCase):
    def test_summary_builds_primary_interaction(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evaluation = root / "evaluation" / "seed_0"
            evaluation.mkdir(parents=True)
            for item in condition_matrix():
                base = 50.0
                if item["supervision_mode"] == "matched_ft":
                    base += 2.0
                if item["prompt_mode"] == "correct":
                    base += 1.0
                (evaluation / f"{item['condition']}.log").write_text(
                    f"Best, last acc:----[{base}, {base + 1}] 0 0\n", encoding="utf-8"
                )
            output = root / "summary"
            subprocess.run(
                [sys.executable, str(HERE / "summarize_results.py"), "--evaluation-root", str(root / "evaluation"), "--output-dir", str(output)],
                check=True,
                capture_output=True,
                text=True,
            )
            payload = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertIn("matching_supervision_x_inference_correspondence", payload["aggregate_contrasts"])


if __name__ == "__main__":
    unittest.main()
