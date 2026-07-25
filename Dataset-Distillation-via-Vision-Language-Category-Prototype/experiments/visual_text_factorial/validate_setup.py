import argparse
import json
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MODEL_COMPONENTS = (
    "model_index.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "tokenizer/tokenizer_config.json",
    "unet/config.json",
    "vae/config.json",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Validate the visual x text factorial inputs")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--prototype", required=True)
    parser.add_argument("--dcs", required=True)
    return parser.parse_args()


def image_count(root):
    return sum(
        1
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def validate_model(reference):
    root = Path(reference)
    if not root.is_dir():
        print(f"[OK] Frozen model: remote Hugging Face reference {reference}")
        return
    missing = [relative for relative in MODEL_COMPONENTS if not (root / relative).is_file()]
    if missing:
        raise RuntimeError(f"Frozen model is incomplete; missing: {missing}")
    print(f"[OK] Frozen model: {root.resolve()}")


def validate_json_pair(prototype_path, dcs_path):
    with Path(prototype_path).open("r", encoding="utf-8") as handle:
        prototypes = json.load(handle)
    with Path(dcs_path).open("r", encoding="utf-8") as handle:
        dcs = json.load(handle)
    if set(prototypes) != set(dcs):
        raise RuntimeError("Prototype and DCS class keys differ")
    if len(prototypes) != 10:
        raise RuntimeError(f"Expected 10 ImageNette classes, found {len(prototypes)}")
    for synset in prototypes:
        if len(prototypes[synset]) != 10 or len(dcs[synset]) != 10:
            raise RuntimeError(
                f"Expected 10 prototypes and DCS prompts for {synset}; "
                f"found {len(prototypes[synset])} and {len(dcs[synset])}"
            )
        if any(not str(value).strip() for value in dcs[synset]):
            raise RuntimeError(f"Empty DCS prompt found for {synset}")
    print("[OK] Fixed IPC-10 prototype and DCS artifacts")
    return set(prototypes)


def main():
    args = parse_args()
    data_root = Path(args.data_root).resolve()
    expected_classes = validate_json_pair(args.prototype, args.dcs)
    for split in ("train", "val"):
        path = data_root / split
        if not path.is_dir():
            raise FileNotFoundError(path)
        actual_classes = {item.name for item in path.iterdir() if item.is_dir()}
        missing_classes = sorted(expected_classes - actual_classes)
        if missing_classes:
            raise RuntimeError(f"{path} is missing prototype classes: {missing_classes}")
        count = image_count(path)
        if count == 0:
            raise RuntimeError(f"No images found under {path}")
        print(f"[OK] {split}: {count} images")
    validate_model(args.base_model)


if __name__ == "__main__":
    main()
