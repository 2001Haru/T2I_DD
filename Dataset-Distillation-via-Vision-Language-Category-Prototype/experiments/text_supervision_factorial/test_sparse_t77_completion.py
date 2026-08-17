import json
import sys
import tempfile
from argparse import Namespace
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from run_sparse_t77_completion import bootstrap_fixed, build_tasks  # noqa: E402


def test_combined_completion_task_grid():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        train = root / "data" / "train"
        rows = []
        for synset in ("a", "b"):
            folder = train / synset
            folder.mkdir(parents=True)
            for index in range(10):
                image = folder / f"{index}.jpg"
                image.write_bytes(b"fixture")
                rows.append({
                    "file_name": f"{synset}/{index}.jpg",
                    "text": f"caption {synset} {index}",
                })
        caption_file = root / "captions.jsonl"
        caption_file.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        args = Namespace(
            run_root=str(root / "run"), data_root=str(root / "data"),
            caption_file=str(caption_file), base_model=str(root / "sd15"),
            prototype=str(root / "prototype.json"), dcs=str(root / "dcs.json"),
            generation_seeds=(0, 1), ipc=50, strength=0.8,
            dense_training_seeds=(0, 1),
            classifier_repeats=2, classifier_seed=0,
        )
        tasks, sparse_index, fixed_index = build_tasks(args)
        assert len(tasks) == 56
        assert len(sparse_index) == 16
        assert len(fixed_index) == 16
        assert sum(task.kind == "train" for task in tasks.values()) == 8
        assert {row["prompt"] for row in fixed_index} == {"correct_t77", "shuffled_t77"}
        assert {row["checkpoint_family"] for row in fixed_index} == {"matched_ft", "unpaired_ft"}
        assert {row["training_seed"] for row in fixed_index} == {0, 1}


def test_fixed_bootstrap_nests_training_generation_and_paired_repeats():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        rows = []
        for training_seed in (0, 1):
            for generation_seed in (0, 1):
                for prompt, values in (
                    ("correct_t77", [60 + training_seed, 62 + generation_seed]),
                    ("shuffled_t77", [59 + training_seed, 61 + generation_seed]),
                ):
                    log = root / f"s{training_seed}_g{generation_seed}_{prompt}.log"
                    log.write_text(f"Best, last acc:----{values}\n", encoding="utf-8")
                    rows.append({
                        "training_seed": training_seed,
                        "generation_seed": generation_seed,
                        "prompt": prompt,
                        "evaluation_log": str(log),
                    })
        shuffled = [row for row in rows if row["prompt"] == "shuffled_t77"]
        shuffled_mean, _, _ = bootstrap_fixed(shuffled, samples=20)
        contrast_mean, _, _ = bootstrap_fixed(rows, contrast=True, samples=20)
        assert shuffled_mean == 60.5
        assert contrast_mean == 1.0
