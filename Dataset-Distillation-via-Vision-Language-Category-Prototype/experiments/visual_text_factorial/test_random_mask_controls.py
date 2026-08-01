import unittest

from summarize_random_mask_controls import aggregate_by_generation, grouped_rows


class RandomMaskControlSummaryTest(unittest.TestCase):
    def test_generation_seed_is_the_aggregate_unit(self):
        rows = []
        for generation_seed, values in ((0, (1.0, 3.0)), (1, (-1.0, 1.0))):
            for shift, value in zip((1, 2), values):
                rows.append(
                    {
                        "mask_seed": 11,
                        "mask_role": "new",
                        "generation_seed": generation_seed,
                        "shuffle_shift": shift,
                        "mean_difference": value,
                    }
                )
        generation_rows = grouped_rows(
            rows,
            ("mask_seed", "mask_role", "generation_seed"),
            "shift_observations",
        )
        self.assertEqual(
            [row["mean_difference"] for row in generation_rows],
            [2.0, 0.0],
        )
        aggregate = aggregate_by_generation(generation_rows)
        self.assertEqual(aggregate[0]["generation_seed_observations"], 2)
        self.assertEqual(aggregate[0]["mean_over_generation_seed"], 1.0)


if __name__ == "__main__":
    unittest.main()
