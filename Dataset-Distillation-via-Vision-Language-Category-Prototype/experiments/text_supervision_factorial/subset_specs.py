from pathlib import Path


SUBSET_SYNSETS = {
    "nette": (
        "n01440764", "n02102040", "n02979186", "n03000684", "n03028079",
        "n03394916", "n03417042", "n03425413", "n03445777", "n03888257",
    ),
    "woof": (
        "n02086240", "n02087394", "n02088364", "n02089973", "n02093754",
        "n02096294", "n02099601", "n02105641", "n02111889", "n02115641",
    ),
}


def validate_subset(root, spec, caption_file=None):
    root = Path(root).resolve()
    expected = set(SUBSET_SYNSETS[spec])
    for split in ("train", "val"):
        split_root = root / split
        if not split_root.is_dir():
            raise FileNotFoundError(split_root)
        observed = {path.name for path in split_root.iterdir() if path.is_dir()}
        if observed != expected:
            raise RuntimeError(
                f"{spec} {split} synsets differ: missing={sorted(expected - observed)}, "
                f"extra={sorted(observed - expected)}"
            )
    if caption_file is not None and not Path(caption_file).resolve().is_file():
        raise FileNotFoundError(Path(caption_file).resolve())
    return root
