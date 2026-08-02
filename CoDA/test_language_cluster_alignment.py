import unittest

import numpy as np

from diagnose_language_cluster_alignment import retrieval_metrics


class LanguageClusterAlignmentTest(unittest.TestCase):
    def test_identity_similarity_has_perfect_bidirectional_retrieval(self):
        metrics = retrieval_metrics(np.eye(4, dtype=np.float32))
        self.assertEqual(metrics["text_to_image_top1"], 1.0)
        self.assertEqual(metrics["image_to_text_top1"], 1.0)
        self.assertEqual(metrics["bidirectional_mrr"], 1.0)
        self.assertGreater(metrics["diagonal_margin"], 0.0)

    def test_permuted_correspondence_changes_targets_but_not_matrix(self):
        similarity = np.eye(3, dtype=np.float32)
        metrics = retrieval_metrics(similarity, np.asarray([1, 2, 0]))
        self.assertEqual(metrics["bidirectional_top1"], 0.0)
        self.assertLess(metrics["diagonal_margin"], 0.0)


if __name__ == "__main__":
    unittest.main()
