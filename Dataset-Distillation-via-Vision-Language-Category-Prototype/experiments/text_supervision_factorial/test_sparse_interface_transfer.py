import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest

import run_sparse_interface_transfer as runner
import summarize_sparse_interface_transfer as summary


def write_eval(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"Best, last acc:----{values}\n", encoding="utf-8")


def test_reuse_key_normalizes_legacy_prompt_names():
    row = {
        "supervision": "matched_ft", "spec": "nette", "ipc": 50,
        "strength": 0.8, "training_seed": 1, "generation_seed": 0,
        "prompt": "shuffled_s1",
    }
    assert runner.reuse_key(row) == ("matched_ft", 1, 0, "shuffled")


def test_bank_semantic_hash_ignores_build_only_metadata(tmp_path):
    classes = {
        "n00000001": [
            {"relative": "n00000001/a.jpg", "caption": "caption a", "nested_rank": 0},
            {"relative": "n00000001/b.jpg", "caption": "caption b", "nested_rank": 1},
        ]
    }
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(json.dumps({
        "format_version": 1, "maximum_nested_budget": 32,
        "caption_file": "/old/path/nette.jsonl", "classes": classes,
    }), encoding="utf-8")
    second.write_text(json.dumps({
        "format_version": 1, "maximum_nested_budget": 4,
        "caption_file": "/new/path/nette.jsonl", "classes": classes,
    }), encoding="utf-8")
    assert runner.sha256(first) != runner.sha256(second)
    assert runner.bank_semantic_sha256(first) == runner.bank_semantic_sha256(second)


def test_attempts_survive_scheduler_restart(tmp_path):
    task = runner.Task("generate", 1, "generate", [], tmp_path, tmp_path / "task.log", lambda: False)
    (tmp_path / "scheduler_events.jsonl").write_text(
        "\n".join((
            json.dumps({"event": "launch", "task": "generate", "attempt": 1}),
            json.dumps({"event": "failure", "task": "generate", "attempt": 1}),
            json.dumps({"event": "launch", "task": "generate", "attempt": 2}),
        )) + "\n",
        encoding="utf-8",
    )
    runner.restore_attempt_counts(tmp_path, {task.name: task})
    assert task.attempts == 2


def test_returncode_reports_process_signals():
    assert runner.decoded_returncode(-9) == "SIGKILL"
    assert runner.decoded_returncode(137) == "SIGKILL"
    assert runner.decoded_returncode(1) == "normal_exit"


def test_gap_decomposition_closes(tmp_path, monkeypatch):
    means = {
        ("sparse_m4_ft", "label"): 78.80,
        ("sparse_m4_ft", "bank"): 79.10,
        ("sparse_m4_ft", "shuffled"): 80.00,
        ("sparse_m4_ft", "correct"): 80.50,
        ("matched_ft", "label"): 78.75,
        ("matched_ft", "bank"): 80.00,
        ("matched_ft", "shuffled"): 80.35,
        ("matched_ft", "correct"): 80.875,
    }
    index = []
    for family in runner.FAMILIES:
        for training_seed in (0, 1):
            for generation_seed in (0, 1):
                for prompt in summary.PROMPTS:
                    log = tmp_path / "logs" / family / str(training_seed) / str(generation_seed) / f"{prompt}.log"
                    value = means[(family, prompt)]
                    write_eval(log, [value, value])
                    index.append({
                        "checkpoint_family": family,
                        "training_seed": training_seed,
                        "generation_seed": generation_seed,
                        "prompt": prompt,
                        "evaluation_log": str(log),
                    })
    index_path = tmp_path / "evaluation_index.json"
    index_path.write_text(json.dumps(index), encoding="utf-8")
    output = tmp_path / "summary"
    monkeypatch.setattr(sys, "argv", [
        "summarize_sparse_interface_transfer.py", "--evaluation-index", str(index_path),
        "--output-dir", str(output), "--bootstrap-samples", "200",
    ])
    summary.main()
    result = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    decomposition = result["gap_decomposition"]
    assert decomposition["total_gap"] == pytest.approx(2.075)
    assert decomposition["checkpoint_gap_under_label"] == pytest.approx(-0.05)
    assert decomposition["descriptive_interface_value"] == pytest.approx(1.60)
    assert decomposition["cluster_correspondence_value"] == pytest.approx(0.525)
    assert decomposition["component_sum"] == pytest.approx(decomposition["total_gap"])
    contrasts = {row["contrast"]: row for row in result["paired_contrasts"]}
    assert contrasts["matched_bank_minus_correct"]["mean_difference"] == pytest.approx(-0.875)
    assert contrasts["matched_bank_minus_shuffled"]["mean_difference"] == pytest.approx(-0.35)
