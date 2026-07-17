import json
import os
import tempfile
import unittest

from summarize_paired_class_oracle import summarize


CLASSES = ("n00000001", "n00000002")


def _write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file)


def _selection_result(path):
    _write_json(path, {"training_seeds": [0, 1]})


def _paired_result(path, runs):
    payload_runs = []
    for seed, accuracies in runs.items():
        payload_runs.append({
            "training_seed": seed,
            "overall_top1": sum(accuracies) / len(accuracies),
            "classes": [
                {"class_id": class_id, "accuracy": accuracy}
                for class_id, accuracy in zip(CLASSES, accuracies)
            ],
        })
    _write_json(path, {"runs": payload_runs})


class PairedClassOracleTest(unittest.TestCase):
    def test_reports_seed_paired_interaction_gap(self):
        with tempfile.TemporaryDirectory() as directory:
            real_selection = os.path.join(directory, "selection", "real.json")
            diffusion_selection = os.path.join(directory, "selection", "diffusion.json")
            _selection_result(real_selection)
            _selection_result(diffusion_selection)
            manifest_path = os.path.join(directory, "oracle_manifest.json")
            _write_json(manifest_path, {
                "real_result": real_selection,
                "diffusion_result": diffusion_selection,
                "classes": [
                    {
                        "local_label": 0, "class_id": CLASSES[0],
                        "class_name": "class one", "selected_source": "diffusion",
                    },
                    {
                        "local_label": 1, "class_id": CLASSES[1],
                        "class_name": "class two", "selected_source": "real",
                    },
                ],
            })

            results_root = os.path.join(directory, "trained")
            method_runs = {
                "real": {2: [80.0, 60.0], 3: [82.0, 62.0]},
                "diffusion": {2: [70.0, 80.0], 3: [72.0, 82.0]},
                "class_oracle": {2: [65.0, 71.0], 3: [66.0, 64.0]},
            }
            for method, runs in method_runs.items():
                path = os.path.join(
                    results_root, "test", method, "seed_start_2", "resnet_ap",
                    "per_class_accuracy_all_seeds.json",
                )
                _paired_result(path, runs)

            output_dir = os.path.join(directory, "summary")
            payload = summarize([("test", manifest_path, results_root)], output_dir)
            gap = payload["subsets"]["test"]["summary"]["paired_interaction_gap"]
            self.assertEqual(payload["subsets"]["test"]["selection_training_seeds"], [0, 1])
            self.assertEqual(payload["subsets"]["test"]["evaluation_training_seeds"], [2, 3])
            self.assertAlmostEqual(gap["mean"], 0.5)
            self.assertTrue(os.path.isfile(os.path.join(output_dir, "paired_seed_results.csv")))
            self.assertTrue(os.path.isfile(os.path.join(output_dir, "paired_class_summary.csv")))


if __name__ == "__main__":
    unittest.main()
