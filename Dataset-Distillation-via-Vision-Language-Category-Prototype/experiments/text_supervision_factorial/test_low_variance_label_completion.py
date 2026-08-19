import sys
import tempfile
from argparse import Namespace
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from run_low_variance_label_completion import build_tasks  # noqa: E402


def test_label_completion_has_three_cells_times_six_seeds():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        args = Namespace(
            run_root=str(root / "run"), data_root=str(root / "data"),
            base_model=str(root / "base"), prototype=str(root / "prototype.json"),
            dcs=str(root / "dcs.json"), matched_model=str(root / "matched"),
            unpaired_model=str(root / "unpaired"), bank_m4_model=str(root / "m4"),
            generation_seeds=tuple(range(6)), ipc=50, strength=0.8,
            classifier_repeats=3, classifier_seed=0, tail_k=10,
        )
        tasks, index = build_tasks(args)
        assert len(tasks) == 36
        assert sum(task.kind == "generate" for task in tasks.values()) == 18
        assert sum(task.kind == "eval" for task in tasks.values()) == 18
        assert len(index) == 18
        assert {row["checkpoint"] for row in index} == {
            "matched_ft", "unpaired_ft", "bank_m4"
        }
        assert {row["prompt"] for row in index} == {"label"}
        assert {row["generation_seed"] for row in index} == set(range(6))
