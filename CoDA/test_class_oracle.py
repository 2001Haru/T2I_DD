import argparse
import json
import os
import tempfile
import unittest

from PIL import Image

import build_class_oracle_dataset as oracle


def _result(class_rows, overall):
    return {
        "overall_top1": overall,
        "class_summary": [
            {
                "local_label": index,
                "class_id": class_id,
                "class_name": class_id,
                "mean": mean,
            }
            for index, (class_id, mean) in enumerate(class_rows)
        ],
    }


class ClassOracleTest(unittest.TestCase):
    def test_build_selects_whole_better_class_pool(self):
        with tempfile.TemporaryDirectory() as directory:
            real_dir = os.path.join(directory, "real")
            diffusion_dir = os.path.join(directory, "diffusion")
            classes = ("n00000001", "n00000002")
            for root, color in ((real_dir, "red"), (diffusion_dir, "blue")):
                for class_id in classes:
                    class_dir = os.path.join(root, class_id)
                    os.makedirs(class_dir)
                    for index in range(2):
                        Image.new("RGB", (4, 4), color=color).save(
                            os.path.join(class_dir, f"{index}.png")
                        )

            real_result = os.path.join(directory, "real.json")
            diffusion_result = os.path.join(directory, "diffusion.json")
            with open(real_result, "w", encoding="utf-8") as file:
                json.dump(_result([(classes[0], 80.0), (classes[1], 60.0)], [70.0]), file)
            with open(diffusion_result, "w", encoding="utf-8") as file:
                json.dump(_result([(classes[0], 70.0), (classes[1], 90.0)], [80.0]), file)

            output = os.path.join(directory, "oracle")
            args = argparse.Namespace(
                spec="test", real_dir=real_dir, diffusion_dir=diffusion_dir,
                real_result=real_result, diffusion_result=diffusion_result,
                output_dir=output, ipc=2, tie_policy="real",
            )
            metadata = oracle.build_oracle(args)
            self.assertEqual(metadata["selected_class_counts"], {"real": 1, "diffusion": 1})
            self.assertEqual(metadata["expected_independent_oracle_accuracy"], 85.0)
            with Image.open(os.path.join(output, classes[0], "0.png")) as image:
                self.assertEqual(image.getpixel((0, 0)), (255, 0, 0))
            with Image.open(os.path.join(output, classes[1], "0.png")) as image:
                self.assertEqual(image.getpixel((0, 0)), (0, 0, 255))


if __name__ == "__main__":
    unittest.main()
