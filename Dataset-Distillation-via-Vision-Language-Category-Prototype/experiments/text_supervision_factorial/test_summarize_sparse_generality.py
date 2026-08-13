import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "summarize_sparse_generality", HERE / "summarize_sparse_generality.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_log(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"Best, last acc:----{values}\n", encoding="utf-8")


class SparseGeneralitySummaryTest(unittest.TestCase):
    def test_merges_old_nette_and_new_cross_dataset_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sparse_new, sparse_old, controls = [], [], []
            for spec, training_seed, target in (
                ("nette", 1, sparse_new), ("woof", 0, sparse_new), ("nette", 0, sparse_old)
            ):
                log = root / f"sparse_{spec}_{training_seed}.log"
                write_log(log, [80.0, 82.0])
                target.append({
                    "spec": spec, "budget": 4, "bank_seed": 0,
                    "training_seed": training_seed, "generation_seed": 0,
                    "prompt": "label", "ipc": 50, "strength": 0.8,
                    "evaluation_log": str(log),
                })
            for spec, training_seed in (("nette", 0), ("nette", 1), ("woof", 0)):
                log = root / f"control_{spec}_{training_seed}.log"
                write_log(log, [78.0, 80.0])
                controls.append({
                    "spec": spec, "supervision": "label_ft", "visual_mode": "prototype",
                    "training_seed": training_seed, "generation_seed": 0,
                    "prompt": "label", "ipc": 50, "strength": 0.8,
                    "evaluation_log": str(log),
                })
            for name, rows in (("new.json", sparse_new), ("old.json", sparse_old), ("controls.json", controls)):
                (root / name).write_text(json.dumps(rows), encoding="utf-8")
            args = SimpleNamespace(
                sparse_index=str(root / "new.json"), old_nette_sparse_index=str(root / "old.json"),
                control_index=[str(root / "controls.json")],
            )
            rows = MODULE.load_rows(args)
            self.assertEqual(len(rows), 6)
            nette_sparse = [row for row in rows if row["spec"] == "nette" and row["method"] == "sparse_m4"]
            self.assertEqual({row["training_seed"] for row in nette_sparse}, {0, 1})
            contrast = MODULE.paired_contrast(
                nette_sparse,
                [row for row in rows if row["spec"] == "nette" and row["method"] == "label_ft"],
                100, __import__("random").Random(0),
            )
            self.assertEqual(contrast[0], 2.0)
            self.assertEqual(contrast[3], 2)


if __name__ == "__main__":
    unittest.main()
