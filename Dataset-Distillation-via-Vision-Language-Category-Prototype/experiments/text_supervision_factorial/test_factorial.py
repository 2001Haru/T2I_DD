import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import build_unpaired_donors, condition_matrix, shuffled_prompt_index  # noqa: E402
from generate_factorial import (  # noqa: E402
    epsilon_branch_metrics,
    first_sentence,
    get_pipeline_embeds,
    planned_guidance_timesteps,
    schedule_matched_noise,
    text_chunk_count,
    tokenmax_shared_length,
    variable_length_audit,
)


class Tokenized:
    def __init__(self, ids):
        self.input_ids = ids


class FakeTokenizer:
    model_max_length = 77
    pad_token_id = 0

    def __call__(self, text, return_tensors="pt", truncation=False, padding=None, max_length=None):
        ids = [1] + list(range(2, 2 + len(str(text).split()))) + [99]
        if truncation and max_length is not None:
            ids = ids[:max_length]
        if padding == "max_length":
            ids = ids + [self.pad_token_id] * (max_length - len(ids))
        return Tokenized(torch.tensor([ids], dtype=torch.long))


class WordCountMismatchTokenizer(FakeTokenizer):
    """Tokenize one whitespace word into more tokens than a three-word prompt."""

    def __call__(self, text, return_tensors="pt", truncation=False, padding=None, max_length=None):
        del return_tensors
        content = list(range(2, 10)) if text == "subword-heavy" else [2, 3, 4]
        ids = [1, *content, 99]
        if truncation and max_length is not None:
            ids = ids[:max_length]
        if padding == "max_length":
            ids = ids + [self.pad_token_id] * (max_length - len(ids))
        return Tokenized(torch.tensor([ids], dtype=torch.long))


class FakeEncoder:
    def __call__(self, ids):
        return (ids.float().unsqueeze(-1).repeat(1, 1, 3),)


class FakePipe:
    tokenizer = FakeTokenizer()
    text_encoder = FakeEncoder()


class FakeScheduler:
    order = 1

    def set_timesteps(self, steps, device=None):
        del device
        self.timesteps = torch.arange(steps - 1, -1, -1) * 100


class FakeScheduledPipe(FakePipe):
    scheduler = FakeScheduler()


class AssignmentTests(unittest.TestCase):
    def test_first_sentence_and_chunk_count(self):
        self.assertEqual(first_sentence("One sentence. Another sentence."), "One sentence.")
        self.assertEqual(first_sentence("No punctuation"), "No punctuation")
        self.assertEqual(text_chunk_count(FakeTokenizer(), "word " * 76), 2)

    def test_padding_control_extends_sequence_not_hidden_dimension(self):
        positive, negative = get_pipeline_embeds(
            FakePipe(), "class label", "negative", "cpu",
            policy="pad_extended", target_chunks=3,
        )
        self.assertEqual(tuple(positive.shape), (1, 231, 3))
        self.assertEqual(tuple(negative.shape), (1, 231, 3))
        self.assertTrue(torch.equal(positive[:, 77:], negative[:, 77:]))
        self.assertTrue(torch.count_nonzero(positive[:, 77:]) == 0)

    def test_t77_padding_control_keeps_one_block_content_and_empty_tail(self):
        positive, negative = get_pipeline_embeds(
            FakePipe(), "word " * 100, "negative", "cpu",
            policy="single_pad_extended", target_chunks=2,
        )
        single_positive, _ = get_pipeline_embeds(
            FakePipe(), "word " * 100, "negative", "cpu", policy="single"
        )
        self.assertEqual(tuple(positive.shape), (1, 154, 3))
        self.assertTrue(torch.equal(positive[:, :77], single_positive))
        self.assertTrue(torch.equal(positive[:, 77:], negative[:, 77:]))

    def test_raw_head_control_differs_from_full_only_in_positive_tail(self):
        prompt = "word " * 100
        full_positive, full_negative = get_pipeline_embeds(
            FakePipe(), prompt, "negative", "cpu", policy="chunked"
        )
        head_positive, head_negative = get_pipeline_embeds(
            FakePipe(), prompt, "negative", "cpu",
            policy="chunk_head_pad_extended", target_chunks=2,
        )
        self.assertEqual(tuple(head_positive.shape), (1, 154, 3))
        self.assertTrue(torch.equal(head_positive[:, :77], full_positive[:, :77]))
        self.assertTrue(torch.equal(head_negative, full_negative))
        self.assertTrue(torch.equal(head_positive[:, 77:], full_negative[:, 77:]))

    def test_raw_head_control_tracks_three_chunk_caption_per_prompt(self):
        prompt = "word " * 170
        self.assertEqual(text_chunk_count(FakeTokenizer(), prompt), 3)
        full_positive, full_negative = get_pipeline_embeds(
            FakePipe(), prompt, "negative", "cpu", policy="chunked"
        )
        head_positive, head_negative = get_pipeline_embeds(
            FakePipe(), prompt, "negative", "cpu",
            policy="chunk_head_pad_extended", target_chunks=3,
        )
        self.assertEqual(tuple(head_positive.shape), (1, 231, 3))
        self.assertTrue(torch.equal(head_positive[:, :77], full_positive[:, :77]))
        self.assertTrue(torch.equal(head_negative, full_negative))
        self.assertTrue(torch.equal(head_positive[:, 77:], full_negative[:, 77:]))

    def test_short_caption_c_d_f_g_are_identical(self):
        prompt = "short caption"
        c, c_negative = get_pipeline_embeds(
            FakePipe(), prompt, "negative", "cpu", policy="single"
        )
        d, d_negative = get_pipeline_embeds(
            FakePipe(), prompt, "negative", "cpu", policy="chunked"
        )
        f, f_negative = get_pipeline_embeds(
            FakePipe(), prompt, "negative", "cpu",
            policy="single_pad_extended", target_chunks=1,
        )
        g, g_negative = get_pipeline_embeds(
            FakePipe(), prompt, "negative", "cpu",
            policy="chunk_head_pad_extended", target_chunks=1,
        )
        self.assertTrue(torch.equal(c, d))
        self.assertTrue(torch.equal(c, f))
        self.assertTrue(torch.equal(c, g))
        self.assertTrue(torch.equal(c_negative, d_negative))
        self.assertTrue(torch.equal(c_negative, f_negative))
        self.assertTrue(torch.equal(c_negative, g_negative))

    def test_short_caption_a_e_are_identical(self):
        a, a_negative = get_pipeline_embeds(
            FakePipe(), "class label", "negative", "cpu", policy="single"
        )
        e, e_negative = get_pipeline_embeds(
            FakePipe(), "class label", "negative", "cpu",
            policy="pad_extended", target_chunks=1,
        )
        self.assertTrue(torch.equal(a, e))
        self.assertTrue(torch.equal(a_negative, e_negative))

    def test_wrapped_empty_control_restarts_bos_eos_in_every_tail_block(self):
        positive, negative = get_pipeline_embeds(
            FakePipe(), "class label", "negative", "cpu",
            policy="wrapped_empty_extended", target_chunks=3,
        )
        empty, _ = get_pipeline_embeds(
            FakePipe(), "", "", "cpu", policy="single"
        )
        self.assertEqual(tuple(positive.shape), (1, 231, 3))
        self.assertTrue(torch.equal(positive[:, 77:154], empty))
        self.assertTrue(torch.equal(positive[:, 154:], empty))
        self.assertTrue(torch.equal(positive[:, 77:], negative[:, 77:]))
        # Unlike a raw padding tail, the wrapped block contains BOS and EOS.
        self.assertGreater(torch.count_nonzero(positive[:, 77:]).item(), 0)

    def test_tokenmax_variable_uses_exact_shared_cfg_length(self):
        positive, negative = get_pipeline_embeds(
            FakePipe(), "word " * 80, "negative", "cpu",
            policy="tokenmax_variable",
        )
        # 80 content tokens plus BOS/EOS, not rounded up to 154 positions.
        self.assertEqual(tuple(positive.shape), (1, 82, 3))
        self.assertEqual(tuple(negative.shape), (1, 82, 3))

    def test_cfg_branch_metrics_are_recorded_separately(self):
        unconditional = torch.zeros(1, 1, 1, 2)
        conditional = torch.tensor([[[[3.0, 4.0]]]])
        metrics = epsilon_branch_metrics(torch.cat([unconditional, conditional]))
        self.assertEqual(metrics["epsilon_uncond"]["l2"], 0.0)
        self.assertEqual(metrics["epsilon_cond"]["l2"], 5.0)
        self.assertEqual(metrics["epsilon_residual"]["l2"], 5.0)
        self.assertAlmostEqual(metrics["epsilon_cond"]["rms"], (12.5 ** 0.5))

    def test_requested_guidance_timestep_is_not_mislabeled_when_schedule_skips_it(self):
        args = SimpleNamespace(
            guidance_diagnostic_timesteps=(200, 500, 800),
            num_inference_steps=10,
            strength=0.7,
            visual_mode="prototype",
            device="cpu",
        )
        mapping = planned_guidance_timesteps(FakeScheduledPipe(), args)
        self.assertEqual(mapping, {200: 200, 500: 500, 800: 600})

    def test_single_policy_is_exactly_one_block(self):
        positive, negative = get_pipeline_embeds(
            FakePipe(), "word " * 100, "negative", "cpu", policy="single"
        )
        self.assertEqual(tuple(positive.shape), (1, 77, 3))
        self.assertEqual(tuple(negative.shape), (1, 77, 3))

    def test_tokenmax_variable_policy_preserves_short_shared_length(self):
        prompt = "class label with several words"
        negative = "negative prompt"
        expected = tokenmax_shared_length(FakeTokenizer(), prompt, negative)
        positive, negative_embeds = get_pipeline_embeds(
            FakePipe(), prompt, negative, "cpu", policy="tokenmax_variable"
        )
        self.assertEqual(expected, 7)
        self.assertEqual(tuple(positive.shape), (1, expected, 3))
        self.assertEqual(tuple(negative_embeds.shape), (1, expected, 3))

    def test_tokenmax_variable_policy_uses_token_lengths_not_whitespace_counts(self):
        pipe = FakePipe()
        pipe.tokenizer = WordCountMismatchTokenizer()
        positive, negative = get_pipeline_embeds(
            pipe, "subword-heavy", "three word prompt", "cpu",
            policy="tokenmax_variable",
        )
        self.assertEqual(tuple(positive.shape), (1, 10, 3))
        self.assertEqual(tuple(negative.shape), (1, 10, 3))
        audit = variable_length_audit(
            pipe.tokenizer, "subword-heavy", "three word prompt"
        )
        self.assertTrue(audit["official_branch_disagrees_with_tokenmax"])
        self.assertTrue(audit["official_whitespace_would_shape_mismatch"])
        self.assertEqual(audit["tokenmax_shared_length"], 10)

    def test_matrix_has_eighteen_unique_cells(self):
        conditions = [item["condition"] for item in condition_matrix()]
        self.assertEqual(len(conditions), 18)
        self.assertEqual(len(set(conditions)), 18)

    def test_unpaired_is_deranged_and_preserves_each_class(self):
        groups = {"a": [0, 1, 2, 3], "b": [4, 5, 6]}
        donors = build_unpaired_donors(groups, seed=17, epoch=2)
        self.assertTrue(all(index != donor for index, donor in enumerate(donors)))
        for indices in groups.values():
            self.assertEqual(sorted(donors[index] for index in indices), sorted(indices))
        self.assertEqual(donors, build_unpaired_donors(groups, seed=17, epoch=2))

    def test_shuffle_shift(self):
        self.assertEqual([shuffled_prompt_index(index, 4, 1) for index in range(4)], [1, 2, 3, 0])

    def test_schedule_matched_noise_is_deterministic_and_does_not_consume_diffusion_rng(self):
        prototype = torch.zeros(4, 8, 8)
        first = schedule_matched_noise(prototype, 17, "cpu", dtype=torch.float32)
        second = schedule_matched_noise(prototype, 17, "cpu", dtype=torch.float32)
        other = schedule_matched_noise(prototype, 18, "cpu", dtype=torch.float32)
        self.assertTrue(torch.equal(first, second))
        self.assertFalse(torch.equal(first, other))
        self.assertEqual(tuple(first.shape), (1, 4, 8, 8))

        expected = torch.rand(3, generator=torch.Generator().manual_seed(17))
        diffusion_generator = torch.Generator().manual_seed(17)
        schedule_matched_noise(prototype, 17, "cpu", dtype=torch.float32)
        actual = torch.rand(3, generator=diffusion_generator)
        self.assertTrue(torch.equal(expected, actual))


class SummaryTests(unittest.TestCase):
    def test_summary_builds_primary_interaction(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evaluation = root / "evaluation" / "seed_0"
            evaluation.mkdir(parents=True)
            for item in condition_matrix():
                base = 50.0
                if item["supervision_mode"] == "matched_ft":
                    base += 2.0
                if item["prompt_mode"] == "correct":
                    base += 1.0
                (evaluation / f"{item['condition']}.log").write_text(
                    f"Best, last acc:----[{base}, {base + 1}] 0 0\n", encoding="utf-8"
                )
            output = root / "summary"
            subprocess.run(
                [sys.executable, str(HERE / "summarize_results.py"), "--evaluation-root", str(root / "evaluation"), "--output-dir", str(output)],
                check=True,
                capture_output=True,
                text=True,
            )
            payload = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertIn("matching_supervision_x_inference_correspondence", payload["aggregate_contrasts"])


if __name__ == "__main__":
    unittest.main()
