import importlib
import importlib.metadata
import sys


MODULES = (
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("diffusers", "diffusers"),
    ("transformers", "transformers"),
    ("numpy", "numpy"),
    ("PIL", "Pillow"),
)


def package_version(distribution):
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def main():
    loaded = {}
    missing = []
    for module_name, distribution in MODULES:
        try:
            loaded[module_name] = importlib.import_module(module_name)
            print(f"[OK] {distribution} {package_version(distribution)}")
        except Exception as error:
            missing.append((distribution, error))
            print(f"[MISSING/BROKEN] {distribution}: {type(error).__name__}: {error}")
    if missing:
        packages = " ".join(sorted({distribution for distribution, _ in missing}))
        print(f"Missing or broken packages: {packages}", file=sys.stderr)
        raise SystemExit(1)

    torch = loaded["torch"]
    if not torch.cuda.is_available():
        print("[BROKEN] PyTorch cannot see CUDA", file=sys.stderr)
        raise SystemExit(1)
    print(f"[OK] CUDA {torch.version.cuda}; GPUs={torch.cuda.device_count()}")

    try:
        from diffusers import StableDiffusionImg2ImgPipeline  # noqa: F401
    except Exception as error:
        print(f"[BROKEN] StableDiffusionImg2ImgPipeline import failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    print("[OK] StableDiffusionImg2ImgPipeline")


if __name__ == "__main__":
    main()
