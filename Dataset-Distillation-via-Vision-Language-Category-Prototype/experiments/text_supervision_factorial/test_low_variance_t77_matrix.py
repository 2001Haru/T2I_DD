import importlib.util
import json
import sys
import tempfile
from argparse import Namespace
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

from run_low_variance_t77_matrix import (  # noqa: E402
    build_tasks,
    linear_paired_bootstrap,
    parse_metrics,
)


def test_matrix_has_preregistered_30_generation_and_48_evaluation_tasks():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        args = Namespace(
            run_root=str(root / "run"), data_root=str(root / "data"),
            base_model=str(root / "base"), prototype=str(root / "prototype.json"),
            dcs=str(root / "dcs.json"), matched_model=str(root / "matched"),
            unpaired_model=str(root / "unpaired"), bank_m4_model=str(root / "m4"),
            bank_m4_json=str(root / "m4.json"), bank_m64_model=str(root / "m64"),
            bank_m64_json=str(root / "m64.json"), label_model=str(root / "label"),
            generation_seeds=tuple(range(6)), ipc=50, strength=0.8,
            classifier_repeats=3, classifier_seed=0, tail_k=10,
        )
        tasks, index = build_tasks(args)
        assert len(tasks) == 78
        assert sum(task.kind == "generate" for task in tasks.values()) == 30
        assert sum(task.kind == "eval" for task in tasks.values()) == 48
        assert len(index) == 48
        assert {row["generation_seed"] for row in index} == set(range(6))


def test_best_tail_parser_and_matching_interaction_are_repeat_paired():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        rows = []
        values = {
            ("matched_ft", "correct_t77"): [62, 64, 66],
            ("unpaired_ft", "correct_t77"): [60, 62, 64],
            ("matched_ft", "shuffled_t77"): [61, 63, 65],
            ("unpaired_ft", "shuffled_t77"): [60, 62, 64],
        }
        for generation in range(6):
            for (checkpoint, prompt), best in values.items():
                log = root / f"g{generation}_{checkpoint}_{prompt}.log"
                tail = [value - 1 for value in best]
                log.write_text(
                    f"Best, last acc:----{best} 0 0\n"
                    f"Tail-10 val acc:----{tail} 0 0\n",
                    encoding="utf-8",
                )
                rows.append({
                    "checkpoint": checkpoint, "generation_seed": generation,
                    "prompt": prompt, "evaluation_log": str(log),
                })
        assert parse_metrics(rows[0]["evaluation_log"])["tail_k"] == 10
        terms = (
            (1.0, ("matched_ft", "correct_t77")),
            (-1.0, ("unpaired_ft", "correct_t77")),
            (-1.0, ("matched_ft", "shuffled_t77")),
            (1.0, ("unpaired_ft", "shuffled_t77")),
        )
        mean, lower, upper = linear_paired_bootstrap(rows, terms, "best", samples=50)
        assert mean == lower == upper == 1.0


def test_plotter_persists_machine_readable_validation_history():
    module_path = REPO / "04_evaluation" / "Minimax" / "misc" / "utils.py"
    spec = importlib.util.spec_from_file_location("minimax_utils_for_test", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with tempfile.TemporaryDirectory() as temporary:
        plotter = module.Plotter(temporary, nepoch=100, idx=2)
        plotter.update(10, 50.0, 40.0, 1.2, 1.3)
        plotter.update(20, 60.0, 45.0, 1.0, 1.1)
        csv_path = Path(temporary) / "curve_2.csv"
        json_path = Path(temporary) / "curve_2.json"
        assert csv_path.is_file() and json_path.is_file()
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        assert [row["acc_val"] for row in payload] == [40.0, 45.0]


def test_summary_contains_direct_m4_minus_m64_pairing():
    source = (HERE / "run_low_variance_t77_matrix.py").read_text(encoding="utf-8")
    assert '"bank_m4_minus_bank_m64"' in source
