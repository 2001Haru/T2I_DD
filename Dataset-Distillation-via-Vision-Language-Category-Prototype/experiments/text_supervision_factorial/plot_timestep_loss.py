import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for mode in ("label_ft", "unpaired_ft", "matched_ft"):
        path = Path(args.model_root) / mode / "timestep_loss_epochs.csv"
        with path.open("r", encoding="utf-8") as handle:
            records = list(csv.DictReader(handle))
        epochs = sorted({int(row["epoch"]) for row in records})
        losses = []
        for epoch in epochs:
            selected = [row for row in records if int(row["epoch"]) == epoch and row["loss"]]
            count = sum(int(row["samples"]) for row in selected)
            losses.append(sum(float(row["loss"]) * int(row["samples"]) for row in selected) / count)
        axes[0].plot(epochs, losses, marker="o", label=mode)
        final = [row for row in records if int(row["epoch"]) == epochs[-1] and row["loss"]]
        centers = [(int(row["timestep_low"]) + int(row["timestep_high"])) / 2 for row in final]
        axes[1].plot(centers, [float(row["loss"]) for row in final], marker="o", label=mode)
    axes[0].set(title="Training loss by epoch", xlabel="Epoch", ylabel="MSE")
    axes[1].set(title="Final-epoch loss by diffusion timestep", xlabel="Training timestep", ylabel="MSE")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)


if __name__ == "__main__":
    main()
