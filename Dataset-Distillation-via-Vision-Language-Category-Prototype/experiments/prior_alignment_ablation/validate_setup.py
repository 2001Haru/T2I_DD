import argparse
import json
from pathlib import Path

from prepare_imagenette import IMAGENETTE_SYNSETS


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
    parser = argparse.ArgumentParser(description="Validate data and model layout before the ablation")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--finetuned-model", required=True)
    return parser.parse_args()


def image_paths(root):
    return {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }


def validate_model(reference, label, require_local):
    root = Path(reference)
    if not root.is_dir():
        if require_local:
            raise FileNotFoundError(f"{label} must be a local Diffusers directory: {reference}")
        print(f"[OK] {label}: remote Hugging Face reference {reference}")
        return
    missing = [relative for relative in MODEL_COMPONENTS if not (root / relative).is_file()]
    if missing:
        raise RuntimeError(f"{label} is not a complete Diffusers pipeline; missing: {missing}")
    print(f"[OK] {label}: {root.resolve()}")


def validate_data(data_root):
    train_root = data_root / "train"
    val_root = data_root / "val"
    metadata_path = train_root / "metadata.jsonl"
    for path in (train_root, val_root):
        if not path.is_dir():
            raise FileNotFoundError(path)
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)

    for split_root in (train_root, val_root):
        actual = {path.name for path in split_root.iterdir() if path.is_dir()}
        missing = set(IMAGENETTE_SYNSETS) - actual
        if missing:
            raise RuntimeError(f"{split_root} is missing ImageNette classes: {sorted(missing)}")

    train_images = image_paths(train_root)
    metadata_images = set()
    with metadata_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            if "file_name" not in item or not str(item.get("text", "")).strip():
                raise ValueError(f"Invalid metadata record at line {line_number}")
            metadata_images.add(str(item["file_name"]).replace("\\", "/"))
    missing_metadata = train_images - metadata_images
    unknown_metadata = metadata_images - train_images
    if missing_metadata or unknown_metadata:
        raise RuntimeError(
            "Image/metadata mismatch: "
            f"{len(missing_metadata)} images without text, {len(unknown_metadata)} unknown metadata paths"
        )
    print(
        f"[OK] ImageNette: {len(train_images)} train images, "
        f"{len(image_paths(val_root))} validation images, complete captions"
    )


def main():
    args = parse_args()
    validate_data(Path(args.data_root).resolve())
    validate_model(args.base_model, "frozen model", require_local=False)
    validate_model(args.finetuned_model, "author fine-tuned model", require_local=True)


if __name__ == "__main__":
    main()
