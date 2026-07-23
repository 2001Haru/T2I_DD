import importlib
import importlib.metadata
import sys


MODULES = (
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("diffusers", "diffusers"),
    ("transformers", "transformers"),
    ("accelerate", "accelerate"),
    ("safetensors", "safetensors"),
    ("numpy", "numpy"),
    ("sklearn", "scikit-learn"),
    ("nltk", "nltk"),
    ("tqdm", "tqdm"),
    ("matplotlib", "matplotlib"),
    ("PIL", "Pillow"),
)


def package_version(distribution):
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def main():
    missing = []
    loaded = {}
    for module_name, distribution in MODULES:
        try:
            loaded[module_name] = importlib.import_module(module_name)
            print(f"[OK] {distribution} {package_version(distribution)}")
        except Exception as error:
            missing.append((distribution, f"{type(error).__name__}: {error}"))
            print(f"[MISSING/BROKEN] {distribution}: {type(error).__name__}: {error}")

    if missing:
        packages = " ".join(sorted({item[0] for item in missing}))
        print("\nInstall the pilot requirements with:", file=sys.stderr)
        print(
            "python -m pip install -r "
            "experiments/prior_alignment_ablation/requirements-pilot.txt",
            file=sys.stderr,
        )
        print(f"Affected packages: {packages}", file=sys.stderr)
        raise SystemExit(1)

    torch = loaded["torch"]
    if not torch.cuda.is_available():
        print("[BROKEN] PyTorch cannot see CUDA", file=sys.stderr)
        raise SystemExit(1)
    print(f"[OK] CUDA {torch.version.cuda}; GPUs={torch.cuda.device_count()}")

    try:
        from diffusers import AutoencoderKL, StableDiffusionImg2ImgPipeline  # noqa: F401
    except Exception as error:
        print(f"[BROKEN] Required Diffusers APIs cannot be imported: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    print("[OK] Required Diffusers APIs")

    nltk = loaded["nltk"]
    missing_resources = []
    for resource in ("tokenizers/punkt_tab/english", "corpora/stopwords"):
        try:
            nltk.data.find(resource)
        except LookupError:
            missing_resources.append(resource)
    if missing_resources:
        print(f"[MISSING] NLTK resources: {missing_resources}", file=sys.stderr)
        print("Run: python -m nltk.downloader punkt_tab stopwords", file=sys.stderr)
        raise SystemExit(1)
    print("[OK] NLTK punkt_tab and stopwords")


if __name__ == "__main__":
    main()
