import json
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from audit_caption_interface import (
    audit_row,
    evaluate_features,
    infer_synset,
    load_cluster_labels,
    mask_class_mentions,
    load_corpora,
    ordering_diagnostic,
    probe_interface_deltas,
    summarize_audit,
    summarize_ordering,
)


class FakeTokenizer:
    model_max_length = 7
    all_special_ids = [0, 1, 2]

    def __call__(self, text, add_special_tokens=True, truncation=False, max_length=None, **_):
        content = [10 + index for index, _word in enumerate(text.split())]
        ids = ([1] + content + [2]) if add_special_tokens else content
        if truncation and max_length is not None and len(ids) > max_length:
            ids = ids[:max_length - 1] + [2]
        return {"input_ids": ids}

    def decode(self, ids, skip_special_tokens=True):
        del skip_special_tokens
        return " ".join("long" for _ in ids)


class CaptionInterfaceAuditTests(unittest.TestCase):
    def test_synset_inference_supports_directory_and_flat_woof_names(self):
        self.assertEqual(infer_synset("n01440764/example.JPEG"), "n01440764")
        self.assertEqual(
            infer_synset("n02096294_1424_n02096294.JPEG"),
            "n02096294",
        )
        self.assertEqual(infer_synset("not_an_imagenet_file.JPEG"), "")

    def test_cluster_assignments_support_flat_caption_names(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "assignments.csv"
            path.write_text(
                "image_path,assigned_cluster\n"
                "/data/train/n02096294/n02096294_1424_n02096294.JPEG,3\n",
                encoding="utf-8",
            )
            labels, reason = load_cluster_labels(path, [{
                "relative": "n02096294_1424_n02096294.JPEG",
                "synset": "n02096294",
            }])
        self.assertIsNone(reason)
        self.assertEqual(labels, ["n02096294:3"])

    def test_class_aliases_are_masked_without_removing_attributes(self):
        masked = mask_class_mentions(
            "A tench, Tinca tinca, is a long brown fish.", "n01440764"
        )
        self.assertNotIn("tench", masked.lower())
        self.assertNotIn("tinca", masked.lower())
        self.assertIn("long brown fish", masked.lower())

    def test_audit_uses_content_budget_and_reports_lost_tail(self):
        row = {
            "dataset": "nette", "condition": "matched_caption", "record_id": "x",
            "relative": "n01440764/a.jpg", "synset": "n01440764",
            "cluster_id": "", "text": "one two three four five six seven eight",
        }
        result = audit_row(FakeTokenizer(), row)
        self.assertEqual(result["content_tokens_full"], 8)
        self.assertEqual(result["train_visible_content_tokens"], 5)
        self.assertEqual(result["lost_content_tokens"], 3)
        self.assertEqual(result["inference_chunk_count"], 2)
        self.assertEqual(result["over_content_budget_75"], 1)

    def test_ordering_diagnostic_separates_ordering_from_shape_failure(self):
        # An ordinary shorter prompt selects the genuinely longer negative branch.
        result = ordering_diagnostic(
            FakeTokenizer(), "one two three four", "one two three four five six"
        )
        self.assertEqual(result["official_selected_branch"], "negative")
        self.assertEqual(result["official_would_shape_mismatch"], 0)

        # split(" ") counts the repeated-space empty field, while the tokenizer's
        # split() does not. The official tie selects positive although negative is
        # one CLIP token longer, which reproduces the unsafe branch precisely.
        result = ordering_diagnostic(
            FakeTokenizer(), "one  two", "one two three"
        )
        self.assertEqual(result["whitespace_words_prompt"], 3)
        self.assertEqual(result["official_selected_branch"], "positive")
        self.assertEqual(result["word_token_ordering_disagrees"], 1)
        self.assertEqual(result["official_would_shape_mismatch"], 1)

    def test_ordering_summary_reports_record_and_unique_counts(self):
        base = {
            "dataset": "nette", "condition": "label", "text": "same",
            "word_token_ordering_disagrees": 1,
            "official_branch_disagrees_with_token_max": 1,
            "official_would_shape_mismatch": 1,
        }
        summary = summarize_ordering([base, dict(base)])
        self.assertEqual(summary[0]["records"], 2)
        self.assertEqual(summary[0]["unique_texts"], 1)
        self.assertEqual(summary[0]["shape_mismatch_records"], 2)
        self.assertEqual(summary[0]["shape_mismatch_unique_texts"], 1)

    def test_load_corpora_preserves_correct_shuffled_marginal(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            captions = root / "captions.jsonl"
            captions.write_text(json.dumps({
                "file_name": "n01440764/a.jpg", "text": "a brown fish"
            }) + "\n", encoding="utf-8")
            dcs = root / "dcs.json"
            dcs.write_text(json.dumps({
                "n01440764": ["small brown fish", "large green fish"]
            }), encoding="utf-8")
            matched, rows = load_corpora({"nette": captions}, {"nette": dcs}, {})
        self.assertEqual(len(matched["nette"]), 1)
        correct = sorted(row["text"] for row in rows if row["condition"] == "correct_dcs")
        shuffled = sorted(row["text"] for row in rows if row["condition"] == "shuffled_dcs")
        self.assertEqual(correct, shuffled)

    def test_summary_is_grouped_by_dataset_and_condition(self):
        base = {
            "dataset": "nette", "condition": "label", "content_tokens_full": 2,
            "over_content_budget_75": 0, "over_encoded_budget_77": 0,
            "inference_chunk_count": 1, "lost_content_tokens": 0,
            "attribute_proxy_total": 0, "attribute_proxy_lost": 0,
            "first_sentence_content_tokens": 2, "content_prefix_match": 1,
        }
        summary = summarize_audit([base, {**base, "content_tokens_full": 4}])
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary[0]["content_tokens_mean"], 3.0)

    @unittest.skipUnless(importlib.util.find_spec("sklearn"), "scikit-learn is not installed")
    def test_probe_recovers_separable_labels(self):
        rng = np.random.default_rng(3)
        labels = np.asarray(["a"] * 20 + ["b"] * 20)
        features = np.concatenate([
            rng.normal(-2, 0.1, size=(20, 4)),
            rng.normal(2, 0.1, size=(20, 4)),
        ])
        summary, folds = evaluate_features(features, labels, 5, 7, 500)
        self.assertEqual(len(folds), 5)
        self.assertGreater(summary["top1"], 0.95)

    def test_probe_interface_delta_has_expected_sign(self):
        rows = []
        for mode, top1 in (("train_t77_pooled", 0.4), ("inference_chunked_pooled", 0.6)):
            rows.append({
                "dataset": "nette", "target": "class_id", "text_interface": mode,
                "top1": top1, "balanced_accuracy": top1,
                "macro_f1": top1, "normalized_mi": top1,
            })
        deltas = probe_interface_deltas(rows)
        top1 = next(row for row in deltas if row["metric"] == "top1")
        self.assertAlmostEqual(top1["chunked_minus_t77"], 0.2)


if __name__ == "__main__":
    unittest.main()
