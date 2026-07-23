import json
import os
import subprocess
import sys
import tempfile
import unittest


class SummarizeDcsTransferTest(unittest.TestCase):
    def test_reads_overall_top1_results(self):
        with tempfile.TemporaryDirectory() as directory:
            trained = os.path.join(directory, "trained")
            result_dir = os.path.join(
                trained, "imageA", "seed_0", "dcs-resnet_ap"
            )
            os.makedirs(result_dir)
            with open(
                os.path.join(result_dir, "per_class_accuracy_all_seeds.json"),
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(
                    {
                        "runs": [
                            {"training_seed": 0, "overall_top1": 76.0},
                            {"training_seed": 1, "overall_top1": 78.0},
                        ]
                    },
                    file,
                )
            output = os.path.join(directory, "summary")
            subprocess.run(
                [
                    sys.executable,
                    os.path.join(os.path.dirname(__file__), "summarize_dcs_transfer.py"),
                    "--trained-root",
                    trained,
                    "--output-dir",
                    output,
                    "--specs",
                    "imageA",
                    "--generation-seeds",
                    "0",
                    "--methods",
                    "dcs",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            with open(
                os.path.join(output, "experiment_summary.json"),
                "r",
                encoding="utf-8",
            ) as file:
                summary = json.load(file)
            self.assertEqual(summary["imageA"]["dcs"]["0"]["mean_accuracy"], 77.0)


if __name__ == "__main__":
    unittest.main()
