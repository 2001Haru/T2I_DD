import argparse
import csv
import hashlib
import json
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path

from common import ensure_manifest, sha256_file
from diagnostic_common import atomic_write_json, load_json, parse_shift_runs


HYBRID_CONDITIONS = ("small3_shuffled", "random3_shuffled")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build selective-shuffle ImageNette datasets by mixing existing "
            "paired correct-DCS and shuffled-DCS images"
        )
    )
    parser.add_argument("--base-run-root", required=True)
    parser.add_argument(
        "--shuffle-run",
        action="append",
        default=[],
        metavar="SHIFT=RUN_ROOT",
    )
    parser.add_argument("--cluster-summary", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--generation-seeds", type=int, nargs="+", default=(0, 1))
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=HYBRID_CONDITIONS,
        default=HYBRID_CONDITIONS,
    )
    parser.add_argument("--selected-count", type=int, default=3)
    parser.add_argument("--random-target-seed", type=int, default=20260731)
    parser.add_argument(
        "--link-mode",
        choices=("symlink", "hardlink", "copy"),
        default="symlink",
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def read_csv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def stable_synset_seed(seed, synset):
    payload = f"{int(seed)}:{synset}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def select_cluster_targets(cluster_rows, selected_count, random_target_seed):
    grouped = defaultdict(list)
    for row in cluster_rows:
        grouped[row["synset"]].append(
            {
                "cluster_index": int(float(row["cluster_index"])),
                "assigned_images": int(float(row["assigned_images"])),
            }
        )

    selections = {}
    for synset, rows in sorted(grouped.items()):
        rows.sort(key=lambda row: row["cluster_index"])
        if selected_count <= 0 or selected_count >= len(rows):
            raise ValueError(
                f"selected-count must be in [1, {len(rows) - 1}] for {synset}"
            )
        indices = [row["cluster_index"] for row in rows]
        if indices != list(range(len(rows))):
            raise ValueError(f"Non-contiguous cluster indices for {synset}: {indices}")

        small = sorted(
            rows,
            key=lambda row: (row["assigned_images"], row["cluster_index"]),
        )[:selected_count]
        small_indices = sorted(row["cluster_index"] for row in small)
        random_pool = [index for index in indices if index not in small_indices]
        generator = random.Random(stable_synset_seed(random_target_seed, synset))
        random_indices = sorted(generator.sample(random_pool, selected_count))
        selections[synset] = {
            "small": small_indices,
            "random": random_indices,
            "small_cluster_sizes": {
                str(row["cluster_index"]): row["assigned_images"] for row in small
            },
            "all_cluster_sizes": {
                str(row["cluster_index"]): row["assigned_images"] for row in rows
            },
            "random_selection_pool": "clusters excluding the selected smallest set",
            "small_random_overlap": 0,
        }
    return selections


def source_condition_root(base_run_root, shuffle_runs, shift, generation_seed):
    if shift == 1:
        root = Path(base_run_root)
    else:
        if shift not in shuffle_runs:
            raise KeyError(f"No shuffled run provided for shift {shift}")
        root = Path(shuffle_runs[shift])
    return root / "synthetic" / f"seed_{generation_seed}" / "prototype_dcs_shuffled"


def load_source_pair(base_run_root, shuffle_runs, shift, generation_seed):
    correct_root = (
        Path(base_run_root)
        / "synthetic"
        / f"seed_{generation_seed}"
        / "prototype_dcs"
    )
    shuffled_root = source_condition_root(
        base_run_root, shuffle_runs, shift, generation_seed
    )
    for root in (correct_root, shuffled_root):
        if not (root / "complete.json").is_file():
            raise FileNotFoundError(f"Incomplete source condition: {root}")
        if not (root / "manifest.json").is_file():
            raise FileNotFoundError(f"Missing source manifest: {root}")

    correct_manifest = load_json(correct_root / "manifest.json")
    shuffled_manifest = load_json(shuffled_root / "manifest.json")
    manifest_shift = int(shuffled_manifest["shuffle_strategy"]["shift"])
    if manifest_shift != shift:
        raise RuntimeError(
            f"Expected shuffled shift {shift}, found {manifest_shift}: {shuffled_root}"
        )
    return correct_root, shuffled_root, correct_manifest, shuffled_manifest


def record_key(record):
    return record["synset"], int(record["image_index"])


def source_image(root, record):
    return (
        Path(root)
        / record["synset"]
        / f"image_{int(record['image_index']):05d}.png"
    ).resolve()


def build_hybrid_records(
    correct_root,
    shuffled_root,
    correct_manifest,
    shuffled_manifest,
    selections,
    target_kind,
):
    correct_records = {
        record_key(record): record for record in correct_manifest["prompt_records"]
    }
    shuffled_records = {
        record_key(record): record for record in shuffled_manifest["prompt_records"]
    }
    if set(correct_records) != set(shuffled_records):
        raise RuntimeError("Correct and shuffled datasets contain different image keys")

    output = []
    for key in sorted(correct_records):
        correct = correct_records[key]
        shuffled = shuffled_records[key]
        visual_index = int(correct["prototype_index"])
        if int(shuffled["prototype_index"]) != visual_index:
            raise RuntimeError(f"Visual prototype mismatch for {key}")
        selected = visual_index in selections[correct["synset"]][target_kind]
        source_record = shuffled if selected else correct
        source_root = shuffled_root if selected else correct_root
        path = source_image(source_root, source_record)
        if not path.is_file():
            raise FileNotFoundError(f"Missing source image: {path}")
        output.append(
            {
                "synset": correct["synset"],
                "class_index": int(correct["class_index"]),
                "image_index": int(correct["image_index"]),
                "prototype_index": visual_index,
                "target_kind": target_kind,
                "target_selected": selected,
                "selected_source": "shuffled" if selected else "correct",
                "prompt_source_index": source_record["prompt_source_index"],
                "prompt": source_record["prompt"],
                "image_seed": int(source_record["image_seed"]),
                "source_image": str(path),
            }
        )
    return output


def materialize_image(source, destination, link_mode, resume):
    source = Path(source).resolve()
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if not resume:
            raise FileExistsError(f"Destination exists without --resume: {destination}")
        if link_mode in ("symlink", "hardlink") and os.path.samefile(
            source, destination
        ):
            return
        if link_mode == "copy" and destination.stat().st_size == source.stat().st_size:
            return
        raise RuntimeError(
            f"Existing destination does not match expected source: {destination}"
        )

    temporary = destination.with_name(destination.name + ".tmp")
    if temporary.exists() or temporary.is_symlink():
        temporary.unlink()
    if link_mode == "symlink":
        os.symlink(source, temporary)
    elif link_mode == "hardlink":
        os.link(source, temporary)
    else:
        shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def hybrid_manifest(
    args,
    shift,
    generation_seed,
    condition,
    target_kind,
    records,
    selections,
    correct_root,
    shuffled_root,
):
    selected_by_class = {
        synset: values[target_kind] for synset, values in sorted(selections.items())
    }
    return {
        "schema_version": 1,
        "condition": condition,
        "generation_seed": generation_seed,
        "shuffle_shift": shift,
        "selected_count_per_class": args.selected_count,
        "random_target_seed": args.random_target_seed,
        "link_mode": args.link_mode,
        "cluster_summary": str(Path(args.cluster_summary).resolve()),
        "cluster_summary_sha256": sha256_file(args.cluster_summary),
        "correct_source_root": str(Path(correct_root).resolve()),
        "shuffled_source_root": str(Path(shuffled_root).resolve()),
        "correct_manifest_sha256": sha256_file(Path(correct_root) / "manifest.json"),
        "shuffled_manifest_sha256": sha256_file(
            Path(shuffled_root) / "manifest.json"
        ),
        "selected_clusters_by_class": selected_by_class,
        "selection_details": selections,
        "prompt_records": records,
    }


def build_condition(
    args,
    output_dir,
    shift,
    generation_seed,
    condition,
    target_kind,
    records,
    selections,
    correct_root,
    shuffled_root,
):
    manifest = hybrid_manifest(
        args,
        shift,
        generation_seed,
        condition,
        target_kind,
        records,
        selections,
        correct_root,
        shuffled_root,
    )
    ensure_manifest(output_dir, manifest, resume=args.resume)
    for record in records:
        destination = (
            Path(output_dir)
            / record["synset"]
            / f"image_{int(record['image_index']):05d}.png"
        )
        materialize_image(
            record["source_image"], destination, args.link_mode, args.resume
        )
    expected = len(records)
    actual = len(list(Path(output_dir).glob("*/*.png")))
    if actual != expected:
        raise RuntimeError(
            f"Hybrid image count mismatch in {output_dir}: {actual} != {expected}"
        )
    atomic_write_json(
        Path(output_dir) / "complete.json",
        {
            "condition": condition,
            "generation_seed": generation_seed,
            "shuffle_shift": shift,
            "images": expected,
        },
    )


def main():
    args = parse_args()
    if args.selected_count != 3:
        raise ValueError(
            "This prespecified experiment uses exactly three selected clusters; "
            "use selected-count=3 and a new method name for other doses."
        )
    shuffle_runs = parse_shift_runs(args.shuffle_run)
    shifts = [1, *sorted(shuffle_runs)]
    selections = select_cluster_targets(
        read_csv(args.cluster_summary),
        args.selected_count,
        args.random_target_seed,
    )
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        output_root / "target_selections.json",
        {
            "selected_count": args.selected_count,
            "random_target_seed": args.random_target_seed,
            "selections": selections,
        },
    )

    for shift in shifts:
        for generation_seed in args.generation_seeds:
            (
                correct_root,
                shuffled_root,
                correct_manifest,
                shuffled_manifest,
            ) = load_source_pair(
                args.base_run_root,
                shuffle_runs,
                shift,
                generation_seed,
            )
            for condition, target_kind in (
                ("small3_shuffled", "small"),
                ("random3_shuffled", "random"),
            ):
                if condition not in args.conditions:
                    continue
                records = build_hybrid_records(
                    correct_root,
                    shuffled_root,
                    correct_manifest,
                    shuffled_manifest,
                    selections,
                    target_kind,
                )
                output_dir = (
                    output_root
                    / f"shift_{shift}"
                    / "synthetic"
                    / f"seed_{generation_seed}"
                    / condition
                )
                build_condition(
                    args,
                    output_dir,
                    shift,
                    generation_seed,
                    condition,
                    target_kind,
                    records,
                    selections,
                    correct_root,
                    shuffled_root,
                )
                print(f"Built {condition}: shift={shift}, seed={generation_seed}")


if __name__ == "__main__":
    main()
