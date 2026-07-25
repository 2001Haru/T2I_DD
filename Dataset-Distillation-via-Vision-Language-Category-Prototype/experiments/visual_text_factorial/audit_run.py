import argparse
import json
import re
from pathlib import Path

from common import condition_matrix, sha256_file, shuffled_prompt_index
from summarize_results import parse_log


IMAGE_DIRECTORY_PATTERN = re.compile(r"ImageNet directory:\s*(.+)")


def parse_args():
    parser = argparse.ArgumentParser(description="Audit a completed visual x text run")
    parser.add_argument("--run-root", required=True)
    return parser.parse_args()


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def record_key(record):
    return record["synset"], int(record["image_index"])


def main():
    args = parse_args()
    run_root = Path(args.run_root).resolve()
    synthetic_root = run_root / "synthetic"
    evaluation_root = run_root / "evaluation"
    conditions = [item["condition"] for item in condition_matrix()]
    seed_dirs = sorted(
        synthetic_root.glob("seed_*"), key=lambda path: int(path.name.split("_")[-1])
    )
    if not seed_dirs:
        raise FileNotFoundError(f"No synthetic seed directories under {synthetic_root}")

    report = {"run_root": str(run_root), "seeds": {}, "warnings": []}
    for seed_dir in seed_dirs:
        generation_seed = int(seed_dir.name.split("_")[-1])
        manifests = {}
        records = {}
        invariant_fields = (
            "checkpoint",
            "prototype_sha256",
            "dcs_sha256",
            "generation_seed",
            "ipc",
            "strength",
            "guidance_scale",
            "num_inference_steps",
            "negative_prompt",
            "shuffle_strategy",
            "size",
            "seed_formula",
            "paired_noise_across_conditions",
        )
        reference_invariants = None
        seed_report = {"conditions": {}}

        for condition in conditions:
            condition_dir = seed_dir / condition
            manifest_path = condition_dir / "manifest.json"
            completion_path = condition_dir / "complete.json"
            if not manifest_path.is_file() or not completion_path.is_file():
                raise FileNotFoundError(f"Incomplete condition: {condition_dir}")
            manifest = load_json(manifest_path)
            completion = load_json(completion_path)
            if manifest["condition"] != condition:
                raise RuntimeError(
                    f"Condition manifest mismatch in {condition_dir}: {manifest['condition']}"
                )
            if int(manifest["generation_seed"]) != generation_seed:
                raise RuntimeError(f"Generation seed mismatch in {condition_dir}")
            current_invariants = {field: manifest[field] for field in invariant_fields}
            if reference_invariants is None:
                reference_invariants = current_invariants
            elif current_invariants != reference_invariants:
                raise RuntimeError(f"Cross-condition invariant mismatch for seed {generation_seed}")

            prototype_path = Path(manifest["prototype_path"])
            dcs_path = Path(manifest["dcs_path"])
            if sha256_file(prototype_path) != manifest["prototype_sha256"]:
                raise RuntimeError(f"Prototype hash changed since generation: {prototype_path}")
            if sha256_file(dcs_path) != manifest["dcs_sha256"]:
                raise RuntimeError(f"DCS hash changed since generation: {dcs_path}")

            prompt_records = manifest["prompt_records"]
            keyed_records = {record_key(record): record for record in prompt_records}
            if len(keyed_records) != len(prompt_records):
                raise RuntimeError(f"Duplicate prompt record in {condition_dir}")
            png_paths = sorted(condition_dir.glob("*/*.png"))
            expected_images = int(completion["images"])
            if len(prompt_records) != expected_images or len(png_paths) != expected_images:
                raise RuntimeError(
                    f"Image/record mismatch in {condition_dir}: "
                    f"records={len(prompt_records)}, png={len(png_paths)}, "
                    f"complete={expected_images}"
                )

            log_path = evaluation_root / f"seed_{generation_seed}" / f"{condition}.log"
            accuracies = parse_log(log_path)
            log_text = log_path.read_text(encoding="utf-8", errors="replace")
            directory_matches = IMAGE_DIRECTORY_PATTERN.findall(log_text)
            if directory_matches:
                logged_directory = directory_matches[-1].strip()
                if str(condition_dir) not in logged_directory:
                    report["warnings"].append(
                        f"seed={generation_seed} condition={condition}: "
                        f"log reports a different image directory: {logged_directory}"
                    )
            else:
                report["warnings"].append(
                    f"seed={generation_seed} condition={condition}: "
                    "evaluation log has no 'ImageNet directory' line"
                )

            manifests[condition] = manifest
            records[condition] = keyed_records
            seed_report["conditions"][condition] = {
                "images": expected_images,
                "classifier_accuracies": accuracies,
                "mean_accuracy": sum(accuracies) / len(accuracies),
            }

        reference_keys = set(records[conditions[0]])
        for condition in conditions[1:]:
            if set(records[condition]) != reference_keys:
                raise RuntimeError(
                    f"Prompt record key mismatch: {conditions[0]} vs {condition}"
                )

        dcs = load_json(manifests[conditions[0]]["dcs_path"])
        for key in sorted(reference_keys):
            reference = records[conditions[0]][key]
            for condition in conditions[1:]:
                candidate = records[condition][key]
                for field in ("synset", "image_index", "prototype_index", "image_seed"):
                    if candidate[field] != reference[field]:
                        raise RuntimeError(
                            f"Paired record mismatch for {key}, {condition}, field={field}"
                        )

            synset = reference["synset"]
            prototype_index = int(reference["prototype_index"])
            for visual_mode in ("no_visual", "prototype"):
                label_record = records[f"{visual_mode}_label"][key]
                dcs_record = records[f"{visual_mode}_dcs"][key]
                shuffled_record = records[f"{visual_mode}_dcs_shuffled"][key]
                if label_record["prompt_source_index"] is not None:
                    raise RuntimeError(f"Label prompt has a source index for {key}")
                if int(dcs_record["prompt_source_index"]) != prototype_index:
                    raise RuntimeError(f"Correct DCS mapping is wrong for {key}")
                expected_shuffled = shuffled_prompt_index(
                    prototype_index,
                    len(dcs[synset]),
                    manifests[f"{visual_mode}_dcs_shuffled"]["shuffle_strategy"]["shift"],
                )
                if int(shuffled_record["prompt_source_index"]) != expected_shuffled:
                    raise RuntimeError(f"Shuffled DCS mapping is wrong for {key}")
                if dcs_record["prompt"] != str(dcs[synset][prototype_index]):
                    raise RuntimeError(f"Correct DCS text is wrong for {key}")
                if shuffled_record["prompt"] != str(dcs[synset][expected_shuffled]):
                    raise RuntimeError(f"Shuffled DCS text is wrong for {key}")

        report["seeds"][str(generation_seed)] = seed_report

    output_path = run_root / "audit.json"
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"Audit passed for {len(seed_dirs)} generation seeds and "
        f"{len(conditions)} conditions. Report: {output_path}"
    )
    for warning in report["warnings"]:
        print(f"[WARNING] {warning}")


if __name__ == "__main__":
    main()
