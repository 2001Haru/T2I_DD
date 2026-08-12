import json
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
                train_batch_size=4, gradient_accumulation_steps=8, num_workers=2,
                mixed_precision="fp16",
            )
            tasks, index = build_tasks(args)
            self.assertEqual(len(tasks), 56)
            self.assertEqual(len(index), 32)
            self.assertEqual(sum(task.kind == "train" for task in tasks.values()), 8)
            self.assertEqual(sum(task.kind == "generate" for task in tasks.values()), 16)
            self.assertEqual(sum(task.kind == "eval" for task in tasks.values()), 32)


if __name__ == "__main__":
    unittest.main()
