import os
import tempfile
import unittest

import numpy as np
from PIL import Image

import gradient_candidate_selection as selection


class GradientCandidateSelectionTest(unittest.TestCase):
    def test_exact_combination_prefers_known_diffusion_target(self):
        combinations = selection._combination_matrix(2)
        real = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        diffusion = np.asarray([[2.0, 0.0], [0.0, 2.0]], dtype=np.float32)
        losses = selection._relative_combination_losses(
            real, diffusion, diffusion.copy(), combinations
        )
        self.assertEqual(int(np.argmin(losses)), 3)
        np.testing.assert_array_equal(combinations[3], [1.0, 1.0])

    def test_distance_weights_are_normalized_and_mild(self):
        weights = selection._neighbor_weights([0.0, 1.0, 2.0], "distance")
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=6)
        self.assertTrue(np.all(np.diff(weights) < 0.0))
        self.assertLess(float(weights[0] / weights[-1]), 3.0)

    def test_selection_copy_preserves_pair_indices(self):
        with tempfile.TemporaryDirectory() as directory:
            candidates = {"n00000001": []}
            for pair_index in range(2):
                pair = {}
                for source, color in (("real", "red"), ("diffusion", "blue")):
                    path = os.path.join(directory, f"{source}_{pair_index}.png")
                    Image.new("RGB", (4, 4), color=color).save(path)
                    pair[source] = path
                candidates["n00000001"].append(pair)

            output = os.path.join(directory, "selected")
            records = selection._copy_selection(
                output,
                ["n00000001"],
                candidates,
                {"n00000001": ["real", "diffusion"]},
            )
            self.assertEqual([row["pair_index"] for row in records], [0, 1])
            with Image.open(os.path.join(output, "n00000001", "0.png")) as image:
                self.assertEqual(image.getpixel((0, 0)), (255, 0, 0))
            with Image.open(os.path.join(output, "n00000001", "1.png")) as image:
                self.assertEqual(image.getpixel((0, 0)), (0, 0, 255))


if __name__ == "__main__":
    unittest.main()
