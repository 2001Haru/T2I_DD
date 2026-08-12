import tempfile
import unittest
import sys
from argparse import Namespace
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_schedule_matched_followup import build_tasks


def write_model(path):
    path.mkdir(parents=True, exist_ok=True)
    (path / "model_index.json").write_text("{}", encoding="utf-8")
    (path / "training_summary.json").write_text("{}", encoding="utf-8")


class ScheduleMatchedFollowupTests(unittest.TestCase):
    def test_default_new_matrix_is_bounded_and_uses_all_controls(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = root / "base"
            causal = root / "causal"
            generality = root / "generality"
            interface = root / "interface"
            model = root / "sd15"
            model.mkdir()
            for artifact_root, name in (
                (generality / "artifacts" / "nette" / "ipc50", "nette"),
                (interface / "artifacts" / "woof" / "ipc50", "woof"),
            ):
                artifact_root.mkdir(parents=True)
                (artifact_root / f"{name}-ipc50-0.7-30-kmexpand1.json").write_text("{}", encoding="utf-8")
                (artifact_root / "dcs.json").write_text("{}", encoding="utf-8")
            for supervision in ("label_ft", "unpaired_ft", "matched_ft"):
                write_model(base / "models" / supervision)
                for seed in (0, 1):
                    write_model(causal / "models" / f"train_seed_{seed}" / supervision)
            for seed in (0, 1):
                for supervision in ("label_ft", "unpaired_ft", "matched_ft"):
                    write_model(interface / "models" / "woof" / f"train_seed_{seed}" / supervision)
            args = Namespace(
                nette_data_root=str(root / "nette"), woof_data_root=str(root / "woof"),
                base_model=str(model), base_run_root=str(base), causal_run_root=str(causal),
                generality_run_root=str(generality), interface_run_root=str(interface),
                woof_model_root=str(interface),
                run_root=str(root / "run"), specs=("nette", "woof"),
                training_seeds=(0, 1), generation_seeds=(0, 1),
                supervisions=("label_ft", "matched_ft"),
                guidance_scale=10.0, num_inference_steps=50, classifier_repeats=2,
                classifier_seed=0, diffusers_src="",
            )
            tasks, index = build_tasks(args, {})
            self.assertEqual(len(index), 192)
            self.assertEqual({row["visual_mode"] for row in index}, {"schedule_matched_noise", "pure_noise"})
            self.assertEqual(sum(row["visual_mode"] == "schedule_matched_noise" for row in index), 96)
            self.assertEqual(sum(row["visual_mode"] == "pure_noise" for row in index), 96)
            self.assertEqual(len(tasks), 288)

            args.supervisions = ("label_ft",)
            tasks, index = build_tasks(args, {})
            self.assertEqual(len(index), 96)
            self.assertEqual(len(tasks), 144)

            args.matrices = ("R",)
            args.supervisions = ("label_ft", "unpaired_ft")
            tasks, index = build_tasks(args, {})
            self.assertEqual(len(index), 96)
            self.assertEqual(len(tasks), 128)
            self.assertEqual({row["matrix"] for row in index}, {"R"})
            self.assertEqual({row["visual_mode"] for row in index}, {"prototype"})
            self.assertEqual({row["strength"] for row in index}, {0.7, 0.9})
            self.assertEqual({row["supervision"] for row in index}, {"label_ft", "unpaired_ft"})


if __name__ == "__main__":
    unittest.main()
