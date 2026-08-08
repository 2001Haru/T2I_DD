import json
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_strength_interaction import reusable_cell, strength_token  # noqa: E402


class StrengthInteractionTests(unittest.TestCase):
    def test_strength_token_is_stable(self):
        self.assertEqual(strength_token(0.7), "0p7")
        self.assertEqual(strength_token(1.0), "1")

    def test_reuse_requires_exact_generation_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            synthetic = root / "base" / "synthetic" / "seed_0" / "matched_ft_correct"
            synthetic.mkdir(parents=True)
            manifest = {
                "ipc": 10, "generation_seed": 0, "strength": 0.7, "guidance_scale": 10.0,
                "num_inference_steps": 50, "supervision_mode": "matched_ft", "prompt_mode": "correct",
            }
            (synthetic / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            (synthetic / "complete.json").write_text(json.dumps({"images": 100}), encoding="utf-8")
            log = root / "base" / "evaluation" / "seed_0" / "matched_ft_correct.log"
            log.parent.mkdir(parents=True)
            log.write_text("Best, last acc:----[60.0, 61.0]\n", encoding="utf-8")
            args = Namespace(
                disable_reuse_0p7=False, base_run_root=str(root / "base"),
                causal_run_root=str(root / "causal"), generality_run_root=str(root / "generality"),
                guidance_scale=10.0, num_inference_steps=50,
            )
            self.assertEqual(reusable_cell(args, 10, 0, 0, "correct"), (synthetic, log))
            manifest["strength"] = 0.8
            (synthetic / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            self.assertIsNone(reusable_cell(args, 10, 0, 0, "correct"))

    def test_summary_reports_prompt_utility_and_reference_interaction(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            index = []
            for strength, correct_gain in ((0.7, 1.0), (0.9, 3.0)):
                for training_seed in (0, 1):
                    for generation_seed in (0, 1):
                        for prompt, score in (("label", 60.0), ("correct", 60.0 + correct_gain)):
                            log = root / f"{strength}_{training_seed}_{generation_seed}_{prompt}.log"
                            log.write_text(f"Best, last acc:----[{score}, {score}]\n", encoding="utf-8")
                            index.append({
                                "ipc": 10, "strength": strength, "training_seed": training_seed,
                                "generation_seed": generation_seed, "prompt": prompt,
                                "supervision": "matched_ft", "source": "fixture",
                                "synthetic_dir": "fixture", "evaluation_log": str(log),
                            })
            index_path = root / "index.json"
            index_path.write_text(json.dumps(index), encoding="utf-8")
            subprocess.run([
                sys.executable, str(HERE / "summarize_strength_interaction.py"),
                "--evaluation-index", str(index_path), "--output-dir", str(root / "summary"),
            ], check=True, capture_output=True, text=True)
            summary = json.loads((root / "summary" / "strength_interaction_summary.json").read_text(encoding="utf-8"))
            utility = next(row for row in summary["contrasts"] if row["strength"] == 0.9)
            interaction = next(row for row in summary["interactions_relative_to_0p7"] if row["strength"] == 0.9)
            self.assertEqual(utility["mean"], 3.0)
            self.assertEqual(interaction["mean"], 2.0)


if __name__ == "__main__":
    unittest.main()
