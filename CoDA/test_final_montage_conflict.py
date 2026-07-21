import json
import os
import tempfile
import unittest

from summarize_final_montage_conflict import METHODS, summarize


CLASSES = ("n00000001", "n00000002")


def _write_result(path, values_by_seed):
    runs = []
    for classifier_seed, class_values in values_by_seed.items():
        runs.append({
            "training_seed": classifier_seed,
            "overall_top1": sum(class_values) / len(class_values),
            "classes": [
                {"class_id": class_id, "accuracy": value}
                for class_id, value in zip(CLASSES, class_values)
            ],
        })
    payload = {
        "runs": runs,
        "class_summary": [
            {"local_label": index, "class_id": class_id, "class_name": f"class {index}"}
            for index, class_id in enumerate(CLASSES)
        ],
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file)


class FinalMontageConflictSummaryTest(unittest.TestCase):
    def test_uses_classifier_seed_paired_differences(self):
        with tempfile.TemporaryDirectory() as directory:
            trained_root = os.path.join(directory, "trained")
            offsets = {
                "coda_baseline": 0.0,
                "montage_common_mode": 2.0,
                "montage_soft_alpha_0p5": 3.0,
                "montage_kappa_cap_0p3": 1.0,
            }
            for generation_seed in (0, 1):
                for method in METHODS:
                    offset = offsets[method]
                    path = os.path.join(
                        trained_root,
                        "imageC",
                        f"seed_{generation_seed}",
                        f"{method}-resnet_ap",
                        "per_class_accuracy_all_seeds.json",
                    )
                    _write_result(path, {
                        0: [70.0 + offset, 80.0 + offset],
                        1: [72.0 + offset, 82.0 + offset],
                    })

            output_dir = os.path.join(directory, "summary")
            payload = summarize(trained_root, ["imageC"], [0, 1], output_dir)
            comparisons = payload["subsets"]["imageC"]["paired_comparisons"]
            self.assertEqual(
                payload["protocol"]["held_out_confirmation_subset"], "imageC"
            )
            self.assertAlmostEqual(
                comparisons["montage_minus_baseline"]
                ["all_paired_classifier_runs"]["mean"],
                2.0,
            )
            self.assertAlmostEqual(
                comparisons["soft_minus_montage"]
                ["all_paired_classifier_runs"]["mean"],
                1.0,
            )
            self.assertAlmostEqual(
                comparisons["kappa_minus_montage"]
                ["all_paired_classifier_runs"]["mean"],
                -1.0,
            )
            self.assertTrue(os.path.isfile(
                os.path.join(output_dir, "paired_gains_by_generation_seed.png")
            ))
            with open(
                os.path.join(output_dir, "per_class_comparison.csv"),
                "r",
                encoding="utf-8",
            ) as file:
                self.assertEqual(sum(1 for _ in file), 3)

    def test_rejects_unpaired_classifier_seeds(self):
        with tempfile.TemporaryDirectory() as directory:
            trained_root = os.path.join(directory, "trained")
            for method in METHODS:
                path = os.path.join(
                    trained_root,
                    "imageC",
                    "seed_0",
                    f"{method}-resnet_ap",
                    "per_class_accuracy_all_seeds.json",
                )
                seeds = {0: [70.0, 80.0]}
                if method == "montage_common_mode":
                    seeds = {1: [70.0, 80.0]}
                _write_result(path, seeds)
            with self.assertRaisesRegex(ValueError, "identical classifier seeds"):
                summarize(
                    trained_root,
                    ["imageC"],
                    [0],
                    os.path.join(directory, "summary"),
                )


if __name__ == "__main__":
    unittest.main()
