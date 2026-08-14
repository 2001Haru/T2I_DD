import json
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from subset_specs import SUBSET_SYNSETS  # noqa: E402
from run_generality import build_tasks  # noqa: E402


class GeneralityTests(unittest.TestCase):
    def test_specs_are_disjoint_ten_class_subsets(self):
        self.assertEqual(len(SUBSET_SYNSETS["nette"]), 10)
        self.assertEqual(len(SUBSET_SYNSETS["woof"]), 10)
        self.assertFalse(set(SUBSET_SYNSETS["nette"]) & set(SUBSET_SYNSETS["woof"]))

    def test_summary_pairs_endpoint_policies(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            index = []
            for training_seed in (0, 1):
                for generation_seed in (0, 1):
                    for supervision, prompt, score in (
                        ("empty_ft", "label", 60.0),
                        ("matched_ft", "correct", 62.0),
                        ("matched_ft", "shuffled", 61.0),
                    ):
                        log = root / f"{training_seed}_{generation_seed}_{supervision}_{prompt}.log"
                        log.write_text(f"Best, last acc:----[{score}, {score + 1}] 0 0\n", encoding="utf-8")
                        index.append({
                            "phase": 1, "training_seed": training_seed, "generation_seed": generation_seed,
                            "supervision": supervision, "prompt": prompt, "ipc": 10, "spec": "nette",
                            "evaluation_log": str(log), "source": "fixture",
                        })
            index_path = root / "index.json"
            index_path.write_text(json.dumps(index), encoding="utf-8")
            subprocess.run([
                sys.executable, str(HERE / "summarize_generality.py"), "--evaluation-index", str(index_path),
                "--output-dir", str(root / "summary"),
            ], check=True, capture_output=True, text=True)
            summary = json.loads((root / "summary" / "generality_summary.json").read_text(encoding="utf-8"))
            endpoint = next(row for row in summary["contrasts"] if row["contrast"].startswith("endpoint_"))
            self.assertEqual(endpoint["mean"], 2.0)

    def test_minimal_ipc_stage_has_expected_cells(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base_model = root / "sd15"
            base_model.mkdir()
            base = root / "base"
            causal = root / "causal"
            for model in (
                base / "models" / "matched_ft",
                causal / "models" / "train_seed_0" / "empty_ft",
            ):
                model.mkdir(parents=True)
                (model / "model_index.json").write_text("{}", encoding="utf-8")
                (model / "training_summary.json").write_text("{}", encoding="utf-8")
            args = Namespace(
                run_root=str(root / "run"), nette_data_root=str(root / "nette"),
                nette_caption_file=str(root / "captions.jsonl"), base_model=str(base_model),
                base_run_root=str(base), causal_run_root=str(causal), nette_prototype=None, nette_dcs=None,
                woof_data_root=None, woof_caption_file=None, phases=("nette_ipc",), gpus="0,1",
                new_training_seeds=(2, 3), ipc_training_seeds=(0,), woof_training_seeds=(0, 1),
                generation_seeds=(0,), ipc_values=(20,), classifier_repeats=2, classifier_seed=0,
                train_batch_size=8, gradient_accumulation_steps=4, num_workers=2,
                mixed_precision="fp16", max_parallel_evals=1, retry_delay_seconds=1, max_retries=1,
                diffusers_src="",
            )
            tasks, index = build_tasks(args)
            self.assertEqual(len(tasks), 8)
            self.assertEqual(len(index), 4)
            self.assertEqual({row["prompt"] for row in index}, {"label", "correct", "shuffled"})


if __name__ == "__main__":
    unittest.main()
