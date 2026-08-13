import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("run_sparse_generality", HERE / "run_sparse_generality.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class SparseGeneralityTest(unittest.TestCase):
    def test_child_command_fixes_bank_and_budget_and_propagates_spec(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(
                nette_data_root=str(root / "nette"), nette_caption_file=str(root / "nette.jsonl"),
                woof_data_root=str(root / "woof"), woof_caption_file=str(root / "woof.jsonl"),
                nette_prototype=str(root / "nette_proto.json"), nette_dcs=str(root / "nette_dcs.json"),
                woof_prototype=str(root / "woof_proto.json"), woof_dcs=str(root / "woof_dcs.json"),
                base_model=str(root / "model"), run_root=str(root / "run"), generation_seeds=(0, 1),
                ipc=50, strength=0.8, classifier_repeats=2, classifier_seed=0,
                retry_delay_seconds=120, diffusers_src="",
            )
            command, output = MODULE.child_command(args, "woof", 1, "1")
            self.assertEqual(command[command.index("--spec") + 1], "woof")
            self.assertEqual(command[command.index("--budgets") + 1], "4")
            self.assertEqual(command[command.index("--bank-seeds") + 1], "0")
            self.assertEqual(command[command.index("--training-seed") + 1], "1")
            self.assertEqual(command[command.index("--prompts") + 1], "label")
            self.assertEqual(command[command.index("--gpus") + 1], "1")
            self.assertEqual(output, root / "run" / "woof" / "train_seed_1")

    def test_cooling_retry_does_not_block_fresh_dataset(self):
        pending = [("nette", [], Path("nette")), ("woof", [], Path("woof"))]
        retry_at = {"nette": 200.0}
        selected = MODULE.pop_ready(pending, retry_at, now=100.0)
        self.assertEqual(selected[0], "woof")
        self.assertEqual(pending[0][0], "nette")


if __name__ == "__main__":
    unittest.main()
