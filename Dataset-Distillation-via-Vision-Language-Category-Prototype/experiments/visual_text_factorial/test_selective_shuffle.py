import tempfile
import unittest
from pathlib import Path

from build_selective_shuffle import (
    build_hybrid_records,
    select_cluster_targets,
)


def cluster_rows(synset="class_a"):
    sizes = [30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
    return [
        {
            "synset": synset,
            "cluster_index": str(index),
            "assigned_images": str(size),
        }
        for index, size in enumerate(sizes)
    ]


def prompt_record(synset, index, prompt_source_index):
    return {
        "synset": synset,
        "class_index": 0,
        "image_index": index,
        "prototype_index": index,
        "prompt_source_index": prompt_source_index,
        "prompt": f"prompt {prompt_source_index}",
        "image_seed": index,
    }


class SelectiveShuffleTest(unittest.TestCase):
    def test_target_selection_finds_smallest_and_is_deterministic(self):
        rows = cluster_rows()

        first = select_cluster_targets(rows, 3, 123)
        second = select_cluster_targets(rows, 3, 123)

        self.assertEqual(first["class_a"]["small"], [0, 1, 2])
        self.assertEqual(first, second)
        self.assertEqual(len(first["class_a"]["random"]), 3)
        self.assertFalse(
            set(first["class_a"]["small"]) & set(first["class_a"]["random"])
        )

    def test_hybrid_uses_shuffled_only_for_selected_visual_clusters(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            correct_root = root / "correct"
            shuffled_root = root / "shuffled"
            correct_records = []
            shuffled_records = []
            for index in range(10):
                for condition_root, content in (
                    (correct_root, b"correct"),
                    (shuffled_root, b"shuffled"),
                ):
                    path = condition_root / "class_a" / f"image_{index:05d}.png"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(content)
                correct_records.append(prompt_record("class_a", index, index))
                shuffled_records.append(
                    prompt_record("class_a", index, (index + 1) % 10)
                )

            selections = {
                "class_a": {
                    "small": [0, 1, 2],
                    "random": [4, 6, 8],
                }
            }
            records = build_hybrid_records(
                correct_root,
                shuffled_root,
                {"prompt_records": correct_records},
                {"prompt_records": shuffled_records},
                selections,
                "small",
            )

            by_index = {record["image_index"]: record for record in records}
            self.assertEqual(by_index[0]["selected_source"], "shuffled")
            self.assertEqual(by_index[0]["prompt_source_index"], 1)
            self.assertEqual(by_index[3]["selected_source"], "correct")
            self.assertEqual(by_index[3]["prompt_source_index"], 3)


if __name__ == "__main__":
    unittest.main()
