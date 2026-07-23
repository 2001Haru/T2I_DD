import argparse
import filecmp
import json
import os
import shutil
import sys
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
DISTILLATION_DIR = REPO_ROOT / "03_distiilation"
sys.path.insert(0, str(DISTILLATION_DIR))

from classes import IMAGENET2012_CLASSES  # noqa: E402


IMAGENETTE_SYNSETS = (
    "n01440764",
    "n02102040",
    "n02979186",
    "n03000684",
    "n03028079",
    "n03394916",
    "n03417042",
    "n03425413",
    "n03445777",
    "n03888257",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare ImageNette from an ImageNet dataset.json without duplicating the image archive."
    )
    parser.add_argument("--source-root", required=True, help="Directory containing dataset.json and image files")
    parser.add_argument("--output-root", required=True, help="ImageNette root to create")
    parser.add_argument(
        "--validation-root",
        default=None,
        help="Optional ImageNet val directory organized by synset folders",
    )
    parser.add_argument("--link-mode", choices=("symlink", "hardlink", "copy"), default="symlink")
    parser.add_argument("--questions-out", default=None, help="LLaVA question JSONL path")
    parser.add_argument("--overwrite-questions", action="store_true")
    return parser.parse_args()


def materialize(source, destination, mode):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        try:
            same_file = os.path.samefile(source, destination)
        except OSError:
            same_file = False
        if mode == "copy" and destination.is_file():
            same_file = filecmp.cmp(source, destination, shallow=False)
        if not same_file:
            raise RuntimeError(f"Destination collision: {destination}")
        return False
    if mode == "symlink":
        destination.symlink_to(source.resolve())
    elif mode == "hardlink":
        os.link(source, destination)
    else:
        shutil.copy2(source, destination)
    return True


def load_dataset_entries(source_root):
    dataset_json = source_root / "dataset.json"
    with dataset_json.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    labels = payload.get("labels")
    if not isinstance(labels, list):
        raise ValueError(f"Expected a labels list in {dataset_json}")
    return labels


def write_questions(path, records, overwrite=False):
    if path.exists() and not overwrite:
        raise FileExistsError(f"Question file exists: {path}. Pass --overwrite-questions to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for question_id, record in enumerate(records):
            prompt = (
                f"Describe the physical appearance of the {record['class_name']} in the image. "
                "Include details about its shape, posture, color, and any distinct features."
            )
            item = {
                "question_id": question_id,
                "image": record["relative_path"],
                "text": prompt,
                "category": "detail",
            }
            handle.write(json.dumps(item, ensure_ascii=True) + "\n")


def main():
    args = parse_args()
    source_root = Path(args.source_root).resolve()
    output_root = Path(args.output_root).resolve()
    train_root = output_root / "train"
    val_root = output_root / "val"
    questions_path = Path(args.questions_out).resolve() if args.questions_out else output_root / "llava_questions.jsonl"

    ordered_synsets = list(IMAGENET2012_CLASSES.keys())
    wanted = set(IMAGENETTE_SYNSETS)
    records = []
    created = 0

    for relative_path, label in load_dataset_entries(source_root):
        label = int(label)
        if label < 0 or label >= len(ordered_synsets):
            raise ValueError(f"ImageNet label out of range: {label}")
        synset = ordered_synsets[label]
        if synset not in wanted:
            continue
        source = source_root / relative_path
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = train_root / synset / source.name
        created += int(materialize(source, destination, args.link_mode))
        records.append(
            {
                "relative_path": destination.relative_to(train_root).as_posix(),
                "class_name": IMAGENET2012_CLASSES[synset],
            }
        )

    missing = wanted - {Path(record["relative_path"]).parts[0] for record in records}
    if missing:
        raise RuntimeError(f"No training images found for ImageNette synsets: {sorted(missing)}")

    if args.validation_root:
        validation_root = Path(args.validation_root).resolve()
        for synset in IMAGENETTE_SYNSETS:
            source_dir = validation_root / synset
            if not source_dir.is_dir():
                raise FileNotFoundError(source_dir)
            destination_dir = val_root / synset
            destination_dir.mkdir(parents=True, exist_ok=True)
            for source in sorted(source_dir.iterdir()):
                if source.is_file():
                    materialize(source, destination_dir / source.name, args.link_mode)

    write_questions(questions_path, records, overwrite=args.overwrite_questions)
    print(f"Prepared {len(records)} ImageNette training images ({created} new links/files).")
    print(f"LLaVA questions: {questions_path}")
    if args.validation_root:
        print(f"Validation subset: {val_root}")


if __name__ == "__main__":
    main()
