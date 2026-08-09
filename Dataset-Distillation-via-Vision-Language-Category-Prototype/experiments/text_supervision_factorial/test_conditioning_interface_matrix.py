import json
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_conditioning_interface_matrix import build_tasks  # noqa: E402


def write_model(path):
    path.mkdir(parents=True, exist_ok=True)
    (path / "model_index.json").write_text("{}", encoding="utf-8")
    (path / "training_summary.json").write_text("{}", encoding="utf-8")


class ConditioningInterfaceMatrixTests(unittest.TestCase):
    def make_args(self, root):
        base = root / "base"
        causal = root / "causal"
        generality = root / "generality"
        model = root / "sd15"
        model.mkdir()
        (model / "model_index.json").write_text("{}", encoding="utf-8")
        prototypes = base / "prototypes"
        prototypes.mkdir(parents=True)
        (prototypes / "text_supervision-ipc10-0.7-30-kmexpand1.json").write_text("{}", encoding="utf-8")
        (prototypes / "dcs.json").write_text("{}", encoding="utf-8")
        for ipc in (20, 50):
            artifacts = generality / "artifacts" / "nette" / f"ipc{ipc}"
            artifacts.mkdir(parents=True)
            (artifacts / f"nette-ipc{ipc}-0.7-30-kmexpand1.json").write_text("{}", encoding="utf-8")
            (artifacts / "dcs.json").write_text("{}", encoding="utf-8")
        write_model(base / "models" / "matched_ft")
        for seed in (0, 1):
            write_model(causal / "models" / f"train_seed_{seed}" / "empty_ft")
            if seed:
                write_model(causal / "models" / f"train_seed_{seed}" / "matched_ft")
        return Namespace(
            base_model=str(model), base_run_root=str(base), causal_run_root=str(causal),
            generality_run_root=str(generality), run_root=str(root / "run"),
            nette_data_root=str(root / "nette"), nette_caption_file=str(root / "nette.jsonl"),
            woof_data_root=str(root / "woof"), woof_caption_file=str(root / "woof.jsonl"),
            matrices=("A", "B", "C"), training_seeds=(0, 1), generation_seeds=(0, 1),
            guidance_scale=10.0, num_inference_steps=50, classifier_repeats=2,
            classifier_seed=0, train_batch_size=4, gradient_accumulation_steps=8,
            num_workers=2, mixed_precision="fp16", diffusers_src="",
        )

    def test_default_matrix_has_preregistered_cell_count_and_no_g(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = self.make_args(root)
            tasks, index = build_tasks(args, {})
            counts = {matrix: sum(row["matrix"] == matrix for row in index) for matrix in "ABC"}
            self.assertEqual(counts, {"A": 396, "B": 180, "C": 120})
            self.assertEqual(len(index), 696)
            self.assertTrue(tasks)
            self.assertEqual({row["prompt"] for row in index}, {"label", "correct", "shuffled"})

    def test_summary_averages_shuffle_realizations_before_correspondence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            index = []
            for prompt, shift, scores in (
                ("label", None, [50, 50]), ("correct", None, [60, 60]),
                ("shuffled", 1, [55, 55]), ("shuffled", 2, [57, 57]),
            ):
                log = root / f"{prompt}_{shift}.log"
                log.write_text(f"Best, last acc:----{scores}\n", encoding="utf-8")
                index.append({
                    "matrix": "A", "spec": "nette", "ipc": 10, "visual_mode": "prototype",
                    "strength": 0.7, "supervision": "matched_ft", "training_seed": 0,
                    "generation_seed": 0, "prompt": prompt, "shuffle_shift": shift,
                    "evaluation_log": str(log), "source": "fixture",
                })
            index_path = root / "index.json"
            index_path.write_text(json.dumps(index), encoding="utf-8")
            subprocess.run([
                sys.executable, str(HERE / "summarize_conditioning_interface_matrix.py"),
                "--evaluation-index", str(index_path), "--output-dir", str(root / "summary"),
            ], check=True, capture_output=True, text=True)
            summary = json.loads(
                (root / "summary" / "conditioning_interface_matrix_summary.json").read_text(encoding="utf-8")
            )
            correspondence = next(
                row for row in summary["contrasts"]
                if row["contrast"] == "correct_minus_shuffled_mean_robustness"
            )
            self.assertEqual(correspondence["mean"], 4.0)
            self.assertEqual(correspondence["training_generation_cells"], 1)

    def test_partial_summary_skips_missing_cells_without_partial_shift_average(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            index = []
            for prompt, shift, scores in (
                ("label", None, [50, 50]), ("correct", None, [60, 60]),
                ("shuffled", 1, [55, 55]), ("shuffled", 2, None),
            ):
                log = root / f"{prompt}_{shift}.log"
                if scores is not None:
                    log.write_text(f"Best, last acc:----{scores}\n", encoding="utf-8")
                index.append({
                    "matrix": "A", "spec": "nette", "ipc": 20, "visual_mode": "prototype",
                    "strength": 0.8, "supervision": "matched_ft", "training_seed": 0,
                    "generation_seed": 0, "prompt": prompt, "shuffle_shift": shift,
                    "evaluation_log": str(log), "source": "fixture",
                })
            index_path = root / "index.json"
            index_path.write_text(json.dumps(index), encoding="utf-8")
            subprocess.run([
                sys.executable, str(HERE / "summarize_conditioning_interface_matrix.py"),
                "--evaluation-index", str(index_path), "--output-dir", str(root / "summary"),
                "--allow-incomplete", "--matrices", "A", "--specs", "nette", "--ipcs", "10", "20",
            ], check=True, capture_output=True, text=True)
            summary = json.loads(
                (root / "summary" / "conditioning_interface_matrix_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["coverage"]["planned_evaluation_cells"], 4)
            self.assertEqual(summary["coverage"]["completed_evaluation_cells"], 3)
            names = {row["contrast"] for row in summary["contrasts"]}
            self.assertIn("correct_minus_shuffled_s1", names)
            self.assertNotIn("correct_minus_shuffled_mean_robustness", names)


if __name__ == "__main__":
    unittest.main()
