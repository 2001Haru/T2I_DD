import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_sparse_checkpoint_controls import (  # noqa: E402
    CONFIG_FILES,
    SUMMARY_SUPERVISION,
    AUDIT_SUPERVISIONS,
    audit_checkpoints,
    build_tasks,
    checkpoint_path,
)


def write_checkpoint(path, supervision, seed):
    path.mkdir(parents=True, exist_ok=True)
    for relative in CONFIG_FILES:
        target = path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({"architecture": relative, "_name_or_path": str(path)}) + "\n")
    (path / "training_summary.json").write_text(json.dumps({
        "complete": True,
        "epochs": 8,
        "supervision": SUMMARY_SUPERVISION[supervision],
        "seed": seed,
        "sparse_bank": None,
    }) + "\n")


class SparseCheckpointControlTests(unittest.TestCase):
    def args(self, root):
        return Namespace(
            data_root=str(root / "data"), base_model=str(root / "sd15"),
            base_run_root=str(root / "base"), causal_run_root=str(root / "causal"),
            sparse_run_root=str(root / "sparse"), prototype=str(root / "prototype.json"),
            dcs=str(root / "dcs.json"), run_root=str(root / "controls"),
            training_seeds=(0, 1), generation_seeds=(0, 1), ipc=50, strength=0.8,
            classifier_repeats=2, classifier_seed=0,
        )

    def populate(self, args):
        for seed in args.training_seeds:
            for supervision in AUDIT_SUPERVISIONS:
                write_checkpoint(checkpoint_path(args, supervision, seed), supervision, seed)

    def test_checkpoint_paths_follow_historical_layout(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self.args(Path(temporary))
            self.assertEqual(checkpoint_path(args, "label_ft", 0), Path(args.base_run_root) / "models" / "label_ft")
            self.assertEqual(
                checkpoint_path(args, "empty_ft", 0),
                Path(args.causal_run_root) / "models" / "train_seed_0" / "empty_ft",
            )
            self.assertEqual(
                checkpoint_path(args, "matched_ft", 1),
                Path(args.causal_run_root) / "models" / "train_seed_1" / "matched_ft",
            )

    def test_audit_and_task_matrix_contain_no_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = self.args(root)
            self.populate(args)
            report = root / "audit.json"
            records = audit_checkpoints(args, report)
            self.assertEqual(len(records), 8)
            self.assertEqual(json.loads(report.read_text())["status"], "pass")
            tasks, index = build_tasks(args)
            self.assertEqual(len(tasks), 24)
            self.assertEqual(len(index), 12)
            self.assertEqual(sum(task.kind == "generate" for task in tasks.values()), 12)
            self.assertEqual(sum(task.kind == "eval" for task in tasks.values()), 12)
            self.assertFalse(any(task.kind == "train" for task in tasks.values()))

    def test_audit_rejects_protocol_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = self.args(root)
            self.populate(args)
            model = checkpoint_path(args, "empty_ft", 1)
            summary = json.loads((model / "training_summary.json").read_text())
            summary["epochs"] = 7
            (model / "training_summary.json").write_text(json.dumps(summary))
            with self.assertRaisesRegex(RuntimeError, "epochs=7"):
                audit_checkpoints(args, root / "audit.json")


if __name__ == "__main__":
    unittest.main()
