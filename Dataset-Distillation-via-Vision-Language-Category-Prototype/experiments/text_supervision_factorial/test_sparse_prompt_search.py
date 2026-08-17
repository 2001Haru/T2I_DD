import json
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_sparse_caption_banks import build_banks  # noqa: E402
from common import build_sparse_bank_donors  # noqa: E402
from run_sparse_prompt_search import build_tasks  # noqa: E402


class SparsePromptSearchTests(unittest.TestCase):
    def test_sparse_assignment_is_balanced_deterministic_and_deranged(self):
        groups = {"a": list(range(13)), "b": list(range(13, 30))}
        banks = {"a": [0, 2, 5, 8], "b": [13, 17, 21, 29]}
        donors = build_sparse_bank_donors(groups, banks, seed=7, epoch=3)
        self.assertEqual(donors, build_sparse_bank_donors(groups, banks, seed=7, epoch=3))
        self.assertTrue(all(index != donor for index, donor in enumerate(donors)))
        for key, indices in groups.items():
            counts = [sum(donors[index] == source for index in indices) for source in banks[key]]
            self.assertLessEqual(max(counts) - min(counts), 1)

    def test_caption_banks_are_nested(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train = root / "train"
            metadata = train / "captions.jsonl"
            rows = []
            for synset in ("a", "b"):
                folder = train / synset
                folder.mkdir(parents=True)
                for index in range(10):
                    image = folder / f"{index}.png"
                    image.write_bytes(b"fixture")
                    rows.append({"file_name": f"{synset}/{index}.png", "text": f"caption {synset} {index}"})
            metadata.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            build_banks(train, metadata, root / "banks", (4, 8), (0, 1))
            for seed in (0, 1):
                small = json.loads((root / "banks" / f"bank_seed_{seed}" / "m_4.json").read_text())
                large = json.loads((root / "banks" / f"bank_seed_{seed}" / "m_8.json").read_text())
                for synset in ("a", "b"):
                    self.assertEqual(small["classes"][synset], large["classes"][synset][:4])

    def test_default_search_has_expected_task_count(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = Namespace(
                run_root=str(root), bank_seeds=(0, 1), budgets=(4, 8, 16, 32),
                generation_seeds=(0, 1), training_seed=0, data_root=str(root / "data"),
                caption_file=str(root / "captions.jsonl"), base_model=str(root / "sd15"),
                prototype=str(root / "prototype.json"), dcs=str(root / "dcs.json"),
                ipc=50, strength=0.8, classifier_repeats=2, classifier_seed=0,
                train_batch_size=8, gradient_accumulation_steps=4, num_workers=2,
                mixed_precision="fp16",
            )
            tasks, index = build_tasks(args)
            self.assertEqual(len(tasks), 56)
            self.assertEqual(len(index), 32)
            self.assertEqual(sum(task.kind == "train" for task in tasks.values()), 8)
            self.assertEqual(sum(task.kind == "generate" for task in tasks.values()), 16)
            self.assertEqual(sum(task.kind == "eval" for task in tasks.values()), 32)

    def test_high_budget_t77_search_has_expected_grid(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = Namespace(
                run_root=str(root), bank_seeds=(0, 1), budgets=(64, 128, 256, 512),
                generation_seeds=(0, 1), training_seed=0, data_root=str(root / "data"),
                caption_file=str(root / "captions.jsonl"), base_model=str(root / "sd15"),
                prototype=str(root / "prototype.json"), dcs=str(root / "dcs.json"),
                ipc=50, strength=0.8, classifier_repeats=2, classifier_seed=0,
                train_batch_size=8, gradient_accumulation_steps=4, num_workers=2,
                mixed_precision="fp16", prompts=("label", "bank_t77"),
            )
            tasks, index = build_tasks(args)
            self.assertEqual(len(tasks), 56)
            self.assertEqual(len(index), 32)
            self.assertEqual({row["budget"] for row in index}, {64, 128, 256, 512})
            self.assertEqual({row["prompt"] for row in index}, {"label", "bank_t77"})

    def test_boundary_t77_search_has_expected_grid(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = Namespace(
                run_root=str(root), bank_seeds=(0, 1), budgets=(16, 32),
                generation_seeds=(0, 1), training_seed=0, data_root=str(root / "data"),
                caption_file=str(root / "captions.jsonl"), base_model=str(root / "sd15"),
                prototype=str(root / "prototype.json"), dcs=str(root / "dcs.json"),
                ipc=50, strength=0.8, classifier_repeats=2, classifier_seed=0,
                train_batch_size=8, gradient_accumulation_steps=4, num_workers=2,
                mixed_precision="fp16", prompts=("label", "bank_t77"),
            )
            tasks, index = build_tasks(args)
            self.assertEqual(len(tasks), 28)
            self.assertEqual(len(index), 16)
            self.assertEqual(sum(task.kind == "train" for task in tasks.values()), 4)
            self.assertEqual(sum(task.kind == "generate" for task in tasks.values()), 8)
            self.assertEqual(sum(task.kind == "eval" for task in tasks.values()), 16)
            self.assertEqual({row["budget"] for row in index}, {16, 32})
            self.assertEqual({row["prompt"] for row in index}, {"label", "bank_t77"})

    def test_resume_manifest_can_change_gpu_allocation(self):
        source = (HERE / "run_sparse_prompt_search.py").read_text(encoding="utf-8")
        self.assertIn('if key != "gpus"', source)
        self.assertIn("Resume GPU allocation changed", source)

    def test_label_only_summary_completes_without_bank_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            log = root / "label.log"
            log.write_text("Best, last acc:----[78.0, 79.0]\n", encoding="utf-8")
            index = root / "index.json"
            index.write_text(json.dumps([{
                "bank_seed": 0, "budget": 4, "generation_seed": 0,
                "prompt": "label", "evaluation_log": str(log),
            }]), encoding="utf-8")
            output = root / "summary"
            subprocess.run([
                sys.executable, str(HERE / "summarize_sparse_prompt_search.py"),
                "--evaluation-index", str(index), "--output-dir", str(output),
                "--bootstrap-samples", "20",
            ], check=True)
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["available_prompts"], ["label"])
            self.assertEqual(summary["bank_minus_label"], [])
            self.assertEqual(summary["saturation"], [])

    def test_t77_bank_summary_uses_t77_condition(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = []
            for bank_seed in (0, 1):
                for generation_seed in (0, 1):
                    for budget in (64, 512):
                        for prompt, offset in (("label", 0), ("bank_t77", 1)):
                            log = root / f"b{bank_seed}_g{generation_seed}_m{budget}_{prompt}.log"
                            value = 70 + offset + budget / 512
                            log.write_text(
                                f"Best, last acc:----[{value}, {value + 1}]\n",
                                encoding="utf-8",
                            )
                            rows.append({
                                "bank_seed": bank_seed, "budget": budget,
                                "generation_seed": generation_seed, "prompt": prompt,
                                "evaluation_log": str(log),
                            })
            index = root / "index.json"
            index.write_text(json.dumps(rows), encoding="utf-8")
            output = root / "summary"
            subprocess.run([
                sys.executable, str(HERE / "summarize_sparse_prompt_search.py"),
                "--evaluation-index", str(index), "--output-dir", str(output),
                "--bootstrap-samples", "20",
            ], check=True)
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["bank_inference_prompt"], "bank_t77")
            self.assertTrue(all(
                row["contrast"] == "bank_t77_minus_label"
                for row in summary["bank_minus_label"]
            ))
            self.assertTrue((output / "saturation_vs_maximum.csv").is_file())

    def test_label_budget_summary_preserves_cross_budget_pairing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = []
            for bank_seed in (0, 1):
                for generation_seed in (0, 1):
                    for budget in (4, 8, 16, 32):
                        log = root / f"b{bank_seed}_g{generation_seed}_m{budget}.log"
                        offset = bank_seed + generation_seed
                        value = 70 + offset + {4: 0, 8: 1, 16: 2, 32: 3}[budget]
                        log.write_text(
                            f"Best, last acc:----[{value}.0, {value + 2}.0]\n",
                            encoding="utf-8",
                        )
                        rows.append({
                            "bank_seed": bank_seed, "budget": budget,
                            "generation_seed": generation_seed, "prompt": "label",
                            "evaluation_log": str(log),
                        })
            index = root / "index.json"
            index.write_text(json.dumps(rows), encoding="utf-8")
            output = root / "summary"
            subprocess.run([
                sys.executable, str(HERE / "summarize_sparse_prompt_search.py"),
                "--evaluation-index", str(index), "--output-dir", str(output),
                "--bootstrap-samples", "200",
            ], check=True)
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            result = summary["label_budget_analysis"][0]
            self.assertEqual(result["budgets"], "4,8,16,32")
            self.assertAlmostEqual(result["m_min_minus_m_max_mean_difference"], -3.0)
            self.assertAlmostEqual(result["log2_budget_slope"], 1.0)


if __name__ == "__main__":
    unittest.main()
