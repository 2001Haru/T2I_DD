import csv
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
            woof_phases=("ladder", "curve_ipc10_20", "curve_ipc50"),
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
            self.assertEqual(counts, {"A": 396, "B": 180, "C": 318})
            self.assertEqual(len(index), 894)
            self.assertTrue(tasks)
            self.assertEqual({row["prompt"] for row in index}, {"label", "correct", "shuffled"})

    def test_targeted_nette_and_phased_woof_counts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = self.make_args(root)
            args.matrices = ("C", "D")
            tasks, index = build_tasks(args, {})
            counts = {matrix: sum(row["matrix"] == matrix for row in index) for matrix in ("C", "D")}
            self.assertEqual(counts, {"C": 318, "D": 48})
            self.assertEqual(len(index), 366)
            self.assertEqual(len(index), len({
                (
                    row["matrix"], row["spec"], row["ipc"], row["visual_mode"], row["strength"],
                    row["supervision"], row["training_seed"], row["generation_seed"],
                    row["prompt"], row["shuffle_shift"],
                )
                for row in index
            }))

            args.woof_phases = ("curve_ipc50",)
            tasks, index = build_tasks(args, {})
            self.assertEqual(sum(row["matrix"] == "C" for row in index), 48)
            self.assertEqual(
                {row["phase"] for row in index if row["matrix"] == "C"},
                {"woof_strength_curve_ipc50"},
            )

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

    def test_formal_prompt_interface_interactions_use_paired_repeat_vectors(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            index = []

            def add_triplet(matrix, spec, ipc, strength, supervision, label, correct, shuffled):
                for prompt, scores in (("label", label), ("correct", correct), ("shuffled", shuffled)):
                    log = root / f"{matrix}_{spec}_{ipc}_{strength}_{supervision}_{prompt}.log"
                    log.write_text(f"Best, last acc:----{scores}\n", encoding="utf-8")
                    index.append({
                        "matrix": matrix, "spec": spec, "ipc": ipc, "visual_mode": "prototype",
                        "strength": strength, "supervision": supervision,
                        "training_seed": None if supervision == "frozen" else 0,
                        "generation_seed": 0, "prompt": prompt,
                        "shuffle_shift": 1 if prompt == "shuffled" else None,
                        "evaluation_log": str(log), "source": "fixture",
                    })

            # Woof checkpoint interaction at IPC10:
            # matched-frozen descriptive marginal = +6; correspondence = -8.
            add_triplet("C", "woof", 10, 0.7, "frozen", [50, 50], [54, 54], [46, 46])
            add_triplet("C", "woof", 10, 0.7, "matched_ft", [50, 50], [56, 56], [56, 56])

            # Cross-dataset IPC50 interaction and its change from strength 0.7 to 0.8.
            add_triplet("C", "woof", 50, 0.7, "matched_ft", [50, 50], [56, 56], [56, 56])
            add_triplet("D", "nette", 50, 0.7, "matched_ft", [50, 50], [60, 60], [54, 54])
            add_triplet("C", "woof", 50, 0.8, "matched_ft", [50, 50], [60, 60], [50, 50])
            add_triplet("D", "nette", 50, 0.8, "matched_ft", [50, 50], [58, 58], [54, 54])

            # New matrices carry both datasets under one matrix name and compare
            # Label-FT directly with Matched-FT.
            add_triplet("E", "woof", 50, 0.7, "label_ft", [50, 50], [52, 52], [54, 54])
            add_triplet("E", "woof", 50, 0.7, "unpaired_ft", [50, 50], [57, 57], [55, 55])
            add_triplet("E", "woof", 50, 0.7, "matched_ft", [50, 50], [58, 58], [56, 56])
            add_triplet("E", "nette", 50, 0.7, "label_ft", [50, 50], [56, 56], [54, 54])
            add_triplet("E", "nette", 50, 0.7, "matched_ft", [50, 50], [60, 60], [56, 56])

            # Complete R fixture for checkpoint x dataset x strength boundaries.
            for spec, strength, supervision, scores in (
                ("nette", 0.7, "label_ft", ([50, 50], [52, 52], [54, 54])),
                ("nette", 0.7, "unpaired_ft", ([50, 50], [57, 57], [55, 55])),
                ("nette", 0.7, "matched_ft", ([50, 50], [58, 58], [56, 56])),
                ("woof", 0.7, "label_ft", ([50, 50], [52, 52], [54, 54])),
                ("woof", 0.7, "unpaired_ft", ([50, 50], [55, 55], [55, 55])),
                ("woof", 0.7, "matched_ft", ([50, 50], [59, 59], [57, 57])),
                ("nette", 0.9, "label_ft", ([50, 50], [54, 54], [54, 54])),
                ("nette", 0.9, "unpaired_ft", ([50, 50], [59, 59], [57, 57])),
                ("nette", 0.9, "matched_ft", ([50, 50], [61, 61], [57, 57])),
                ("woof", 0.9, "label_ft", ([50, 50], [52, 52], [52, 52])),
                ("woof", 0.9, "unpaired_ft", ([50, 50], [56, 56], [54, 54])),
                ("woof", 0.9, "matched_ft", ([50, 50], [57, 57], [57, 57])),
            ):
                add_triplet("R", spec, 50, strength, supervision, *scores)

            index_path = root / "index.json"
            index_path.write_text(json.dumps(index), encoding="utf-8")
            subprocess.run([
                sys.executable, str(HERE / "summarize_conditioning_interface_matrix.py"),
                "--evaluation-index", str(index_path), "--output-dir", str(root / "summary"),
            ], check=True, capture_output=True, text=True)
            with (root / "summary" / "formal_interactions.csv").open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

            def value(analysis, contrast, effect, ipc, visual, matrix_left=None, spec_left=None):
                row = next(
                    row for row in rows
                    if row["analysis"] == analysis and row["contrast"] == contrast
                    and row["effect"] == effect and int(row["ipc"]) == ipc
                    and row["visual"] == visual
                    and (matrix_left is None or row["matrix_left"] == matrix_left)
                    and (spec_left is None or row["spec_left"] == spec_left)
                )
                return float(row["mean"])

            self.assertEqual(value(
                "checkpoint_prompt_interaction", "matched_ft_minus_frozen",
                "descriptive_marginal", 10, "strength_0.7",
            ), 6.0)
            self.assertEqual(value(
                "checkpoint_prompt_interaction", "matched_ft_minus_frozen",
                "correspondence", 10, "strength_0.7",
            ), -8.0)
            self.assertEqual(value(
                "dataset_interaction", "woof_minus_nette",
                "descriptive_marginal", 50, "strength_0.7", "C",
            ), -1.0)
            self.assertEqual(value(
                "dataset_interaction", "woof_minus_nette",
                "correspondence", 50, "strength_0.7", "C",
            ), -6.0)
            self.assertEqual(value(
                "dataset_by_strength_interaction",
                "(woof-nette)_strength_0.8_minus_(woof-nette)_strength_0.7",
                "correspondence", 50, "strength_0.8",
            ), 12.0)
            self.assertEqual(value(
                "checkpoint_prompt_interaction", "matched_ft_minus_label_ft",
                "correspondence", 50, "strength_0.7", "E", "woof",
            ), 4.0)
            self.assertEqual(value(
                "checkpoint_descriptive_average", "unpaired_ft_minus_label_ft",
                "descriptive_average", 50, "strength_0.7", "E", "woof",
            ), 3.0)
            self.assertEqual(value(
                "checkpoint_descriptive_average", "matched_ft_minus_unpaired_ft",
                "descriptive_average", 50, "strength_0.7", "E", "woof",
            ), 1.0)
            self.assertEqual(value(
                "checkpoint_correspondence_interaction", "unpaired_ft_minus_label_ft",
                "correspondence", 50, "strength_0.7", "E", "woof",
            ), 4.0)
            self.assertEqual(value(
                "checkpoint_correspondence_interaction", "matched_ft_minus_unpaired_ft",
                "correspondence", 50, "strength_0.7", "E", "woof",
            ), 0.0)
            self.assertEqual(value(
                "checkpoint_by_dataset_interaction",
                "unpaired_ft_minus_label_ft__woof_minus_nette",
                "descriptive_average", 50, "strength_0.7", "R", "woof",
            ), -1.0)
            self.assertEqual(value(
                "checkpoint_by_strength_interaction",
                "matched_ft_minus_unpaired_ft__strength_0.9_minus_strength_0.7",
                "correspondence", 50, "strength_0.9", "R", "woof",
            ), -4.0)
            self.assertEqual(value(
                "checkpoint_by_dataset_strength_interaction",
                "matched_ft_minus_unpaired_ft__(woof-nette)_strength_0.9_minus_strength_0.7",
                "correspondence", 50, "strength_0.9", "R", "woof-minus-nette",
            ), -6.0)
            generalized = [
                row for row in rows
                if row["analysis"] == "dataset_interaction"
                and row["matrix_left"] == "E" and row["matrix_right"] == "E"
            ]
            self.assertTrue(generalized)
            self.assertGreater(
                (root / "summary" / "conditioning_interface_matrix_summary.png").stat().st_size,
                10_000,
            )
            self.assertGreater(
                (root / "summary" / "checkpoint_statistical_boundaries.png").stat().st_size,
                10_000,
            )
            self.assertGreater(
                (root / "summary" / "checkpoint_heterogeneity_interactions.png").stat().st_size,
                10_000,
            )


if __name__ == "__main__":
    unittest.main()
