"""Small arithmetic tests for the P5 factorial contrasts.

Run this in the CoDA experiment environment, where the P4 evaluator's model
dependencies are installed.
"""

import unittest

from evaluate_p5_continuous_guidance import (
    EFFECT_METRICS,
    GEOMETRY_METRICS,
    geometry_three_way_rows,
    guidance_interaction_rows,
    summarize_interactions_by_generation_seed,
)


class P5InteractionTest(unittest.TestCase):
    def make_row(self, regime, value):
        row = {
            "encoder": "dino", "probe": "linear_probe", "spec": "imageA",
            "class_key": "imageA:n00000001", "class_id": "n00000001",
            "class_name": "example", "visual_cluster_id": 2,
            "caption_source_cluster_id": 5, "generation_seed": 0,
            "image_seed": 123, "visual_mode": regime,
        }
        row.update({metric: value for metric in EFFECT_METRICS})
        return row

    def test_guidance_and_three_way_signs(self):
        rows = [
            self.make_row("i0g0", 1.0), self.make_row("i0g1", 4.0),
            self.make_row("i1g0", 10.0), self.make_row("i1g1", 12.0),
        ]
        output = guidance_interaction_rows(rows)
        by_initialization = {row["initialization"]: row for row in output}
        self.assertEqual(by_initialization["i0"]["guidance_interaction_delta_target"], 3.0)
        self.assertEqual(by_initialization["i1"]["guidance_interaction_delta_target"], 2.0)
        self.assertEqual(by_initialization["i1_minus_i0"]["three_way_delta_target"], -1.0)

    def test_seed_summary_uses_matching_column(self):
        raw = guidance_interaction_rows([
            self.make_row("i0g0", 0.0), self.make_row("i0g1", 2.0),
            self.make_row("i1g0", 0.0), self.make_row("i1g1", 5.0),
        ])
        summary = summarize_interactions_by_generation_seed(raw)
        row = next(
            item for item in summary
            if item["initialization"] == "i1_minus_i0" and item["metric"] == "delta_pull"
        )
        self.assertEqual(row["mean"], 3.0)

    def test_geometry_ratio_of_ratios(self):
        base = {
            "spec": "imageA", "class_key": "imageA:n00000001",
            "class_id": "n00000001", "class_name": "example",
            "visual_cluster_id": 2, "generation_seed": 0, "image_seed": 123,
        }
        i0 = {**base, "initialization": "i0"}
        i1 = {**base, "initialization": "i1"}
        for metric in GEOMETRY_METRICS:
            i0[metric] = -0.4
            i1[metric] = -2.0
        row = geometry_three_way_rows([i0, i1])[0]
        self.assertAlmostEqual(row["three_way_text_norm_log_ratio"], -1.6)
        self.assertAlmostEqual(row["three_way_swap_norm_log_ratio"], -1.6)


if __name__ == "__main__":
    unittest.main()
