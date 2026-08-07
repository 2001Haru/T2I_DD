#!/usr/bin/env python3
"""Prepare ImageNette or ImageWoof and LLaVA questions from ImageNet dataset.json."""

import argparse
import filecmp
import json
import os
import shutil
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT / "03_distiilation"))
from classes import IMAGENET2012_CLASSES  # noqa: E402
from subset_specs import SUBSET_SYNSETS  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", choices=sorted(SUBSET_SYNSETS), required=True)
    parser.add_argument("--source-root", required=True, help="ImageNet root containing dataset.json")
    parser.add_argument("--validation-root", required=True, help="ImageNet val root organized by synset")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--link-mode", choices=("symlink", "hardlink", "copy"), default="symlink")
    parser.add_argument("--questions-out", default=None)
    parser.add_argument("--overwrite-questions", action="store_true")
    return parser.parse_args()


def materialize(source, destination, mode):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        try:
            same = os.path.samefile(source, destination)
        except OSError:
            same = mode == "copy" and destination.is_file() and filecmp.cmp(source, destination, shallow=False)
        if not same:
            raise RuntimeError(f"Destination collision: {destination}")
        return False
    if mode == "symlink":
        destination.symlink_to(source.resolve())
    elif mode == "hardlink":
        os.link(source, destination)
    else:
        shutil.copy2(source, destination)
    return True


def main():
    args = parse_args()
    source_root = Path(args.source_root).resolve()
    validation_root = Path(args.validation_root).resolve()
    output_root = Path(args.output_root).resolve()
    questions = Path(args.questions_out).resolve() if args.questions_out else output_root / "llava_questions.jsonl"
    if questions.exists() and not args.overwrite_questions:
        raise FileExistsError(f"Question file exists: {questions}; pass --overwrite-questions")
    payload = json.loads((source_root / "dataset.json").read_text(encoding="utf-8"))
    ordered_synsets = list(IMAGENET2012_CLASSES)
    wanted = set(SUBSET_SYNSETS[args.spec])
    records = []
    created = 0
    for relative, label in payload.get("labels", []):
        synset = ordered_synsets[int(label)]
        if synset not in wanted:
            continue
        source = source_root / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = output_root / "train" / synset / source.name
        created += int(materialize(source, destination, args.link_mode))
        records.append((destination.relative_to(output_root / "train").as_posix(), synset))
    if {synset for _, synset in records} != wanted:
        raise RuntimeError(f"Training subset is incomplete for {args.spec}")
    for synset in SUBSET_SYNSETS[args.spec]:
        source_dir = validation_root / synset
        if not source_dir.is_dir():
            raise FileNotFoundError(source_dir)
        for source in sorted(source_dir.iterdir()):
            if source.is_file():
                materialize(source, output_root / "val" / synset / source.name, args.link_mode)
    questions.parent.mkdir(parents=True, exist_ok=True)
    with questions.open("w", encoding="utf-8") as handle:
        for question_id, (relative, synset) in enumerate(records):
            class_name = IMAGENET2012_CLASSES[synset]
            item = {
                "question_id": question_id,
                "image": relative,
                "text": (
                    f"Describe the physical appearance of the {class_name} in the image. "
                    "Include details about its shape, posture, color, and any distinct features."
                ),
                "category": "detail",
            }
            handle.write(json.dumps(item, ensure_ascii=True) + "\n")
    print(f"Prepared {len(records)} {args.spec} training images ({created} new files/links).")
    print(f"LLaVA questions: {questions}")


if __name__ == "__main__":
    main()
