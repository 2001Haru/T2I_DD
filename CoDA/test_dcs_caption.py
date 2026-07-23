import os
import sys
import tempfile
import unittest
import json

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from dcs_caption import (
    _flat_features,
    _read_jsonl,
    _repair_trailing_jsonl,
    _validate_or_write_rank_config,
    select_dcs_captions,
    tokenize,
)


class DcsCaptionTest(unittest.TestCase):
    def test_tokenize_keeps_only_lowercase_alpha_tokens(self):
        self.assertEqual(tokenize("A red-winged Bird, #2."), ["a", "red", "winged", "bird"])

    def test_selects_existing_caption_with_cluster_weighted_words(self):
        captions = [
            "A bird with a red crown and narrow black beak.",
            "A bird with a red crown and red wings.",
            "A bird with a blue tail and narrow beak.",
            "A bird with a blue tail and blue wings.",
        ]
        assignments = np.asarray([0, 0, 1, 1])
        selected, class_common, diagnostics = select_dcs_captions(
            captions=captions,
            assignments=assignments,
            class_name="bird",
            threshold=0.75,
            top_k=30,
            stop_words={"a", "with", "and"},
        )
        self.assertEqual(selected[0], captions[0])
        self.assertEqual(selected[1], captions[2])
        self.assertIn("bird", class_common)
        self.assertEqual(diagnostics[0]["member_count"], 2)

    def test_ties_are_deterministic(self):
        captions = ["first caption", "second caption"]
        selected, _, diagnostics = select_dcs_captions(
            captions=captions,
            assignments=np.asarray([0, 0]),
            class_name="object",
            threshold=0.0,
            top_k=30,
            stop_words=set(),
        )
        self.assertEqual(selected[0], "first caption")
        self.assertEqual(diagnostics[0]["selected_member_index"], 0)

    def test_repair_discards_only_interrupted_final_record(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "captions.jsonl")
            with open(path, "w", encoding="utf-8") as file:
                file.write('{"caption": "complete"}\n')
                file.write('{"caption": "interrupted"')
            _repair_trailing_jsonl(path)
            self.assertEqual(_read_jsonl(path), [{"caption": "complete"}])

    def test_flat_features_accepts_list_of_tensors(self):
        import torch

        features = [
            torch.arange(8, dtype=torch.float32).reshape(1, 2, 2, 2),
            torch.ones((1, 2, 2, 2), dtype=torch.float32),
        ]
        flattened = _flat_features(features)
        self.assertEqual(flattened.shape, (2, 8))
        np.testing.assert_array_equal(flattened[0], np.arange(8, dtype=np.float32))

    def test_old_world_size_does_not_block_single_gpu_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "captions.rank0.meta.json")
            expected = {
                "format_version": 1,
                "spec": "imageA",
                "model": "/models/llava",
                "instruction_template": "describe",
                "max_new_tokens": 128,
            }
            with open(path, "w", encoding="utf-8") as file:
                json.dump({**expected, "world_size": 2}, file)
            _validate_or_write_rank_config(path, expected)


if __name__ == "__main__":
    unittest.main()
