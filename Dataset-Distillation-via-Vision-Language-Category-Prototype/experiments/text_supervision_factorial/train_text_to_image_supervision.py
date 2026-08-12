import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm.auto import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DISTILLATION_DIR = REPO_ROOT / "03_distiilation"
import sys

sys.path.insert(0, str(DISTILLATION_DIR))
from classes import IMAGENET2012_CLASSES  # noqa: E402
from common import build_sparse_bank_donors, build_unpaired_donors  # noqa: E402


class LocalEMAModel:
    """Diffusers-compatible EMA without the legacy transformers.deepspeed check."""

    def __init__(self, parameters, decay=0.9999, min_decay=0.0):
        parameters = list(parameters)
        self.shadow_params = [parameter.detach().clone() for parameter in parameters]
        self.decay = float(decay)
        self.min_decay = float(min_decay)
        self.optimization_step = 0

    def get_decay(self):
        step = max(0, self.optimization_step - 1)
        if step <= 0:
            return 0.0
        value = (1 + step) / (10 + step)
        return max(self.min_decay, min(value, self.decay))

    @torch.no_grad()
    def step(self, parameters):
        parameters = list(parameters)
        if len(parameters) != len(self.shadow_params):
            raise ValueError("EMA parameter count changed during training")
        self.optimization_step += 1
        one_minus_decay = 1.0 - self.get_decay()
        for index, (shadow, parameter) in enumerate(zip(self.shadow_params, parameters)):
            if shadow.device != parameter.device or shadow.dtype != parameter.dtype:
                shadow = shadow.to(device=parameter.device, dtype=parameter.dtype)
                self.shadow_params[index] = shadow
            if parameter.requires_grad:
                shadow.sub_(one_minus_decay * (shadow - parameter.detach()))
            else:
                shadow.copy_(parameter.detach())

    @torch.no_grad()
    def copy_to(self, parameters):
        for shadow, parameter in zip(self.shadow_params, parameters):
            parameter.data.copy_(shadow.to(device=parameter.device, dtype=parameter.dtype))

    def to(self, device=None, dtype=None):
        self.shadow_params = [
            parameter.to(device=device, dtype=dtype if parameter.is_floating_point() else None)
            for parameter in self.shadow_params
        ]

    def state_dict(self):
        return {
            "decay": self.decay,
            "min_decay": self.min_decay,
            "optimization_step": self.optimization_step,
            "shadow_params": self.shadow_params,
        }

    def load_state_dict(self, state_dict):
        target_devices = [parameter.device for parameter in self.shadow_params]
        target_dtypes = [parameter.dtype for parameter in self.shadow_params]
        self.decay = float(state_dict["decay"])
        self.min_decay = float(state_dict["min_decay"])
        self.optimization_step = int(state_dict["optimization_step"])
        loaded = state_dict["shadow_params"]
        if len(loaded) != len(target_devices):
            raise ValueError("EMA checkpoint parameter count differs from the current UNet")
        self.shadow_params = [
            parameter.detach().to(device=device, dtype=dtype).clone()
            for parameter, device, dtype in zip(loaded, target_devices, target_dtypes)
        ]


def parse_args():
    parser = argparse.ArgumentParser(description="Train SD1.5 with controlled text supervision")
    parser.add_argument("--pretrained-model", required=True)
    parser.add_argument("--train-root", required=True)
    parser.add_argument("--caption-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--supervision",
        choices=("empty", "constant", "label", "matched", "unpaired", "sparse_unpaired"),
        required=True,
    )
    parser.add_argument(
        "--constant-prompt",
        default="A natural photo.",
        help="Shared non-class text used by constant supervision.",
    )
    parser.add_argument("--sparse-bank", default=None)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--num-train-epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--lr-scheduler", default="constant")
    parser.add_argument("--lr-warmup-steps", type=int, default=0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--mixed-precision", choices=("no", "fp16", "bf16"), default="fp16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--checkpointing-steps", type=int, default=500)
    parser.add_argument("--checkpoints-total-limit", type=int, default=2)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--loss-log-steps", type=int, default=50)
    parser.add_argument("--timestep-bins", type=int, default=10)
    parser.add_argument("--random-flip", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--use-ema", action="store_true")
    return parser.parse_args()


def read_records(train_root, caption_file):
    train_root = Path(train_root).resolve()
    image_paths = [
        path for path in train_root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    ]
    by_basename = defaultdict(list)
    for image_path in image_paths:
        by_basename[image_path.name].append(image_path)
    rows = []
    seen = set()
    with Path(caption_file).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            relative = str(item.get("file_name", "")).replace("\\", "/")
            caption = str(item.get("text", "")).strip()
            path = train_root / relative
            if relative and not path.is_file() and len(Path(relative).parts) == 1:
                matches = by_basename.get(relative, [])
                if len(matches) == 1:
                    path = matches[0]
                    relative = path.relative_to(train_root).as_posix()
            if not relative or not caption or not path.is_file():
                raise ValueError(f"Invalid caption row {line_number}: {relative}")
            synset = Path(relative).parts[0]
            if synset not in IMAGENET2012_CLASSES:
                raise ValueError(f"Unknown synset at row {line_number}: {synset}")
            if relative in seen:
                raise ValueError(f"Duplicate image in caption metadata: {relative}")
            seen.add(relative)
            rows.append({"path": path, "relative": relative, "synset": synset, "caption": caption})
    images = {path.relative_to(train_root).as_posix() for path in image_paths}
    if images != seen:
        raise RuntimeError(f"Caption/image mismatch: {len(images - seen)} missing, {len(seen - images)} unknown")
    return rows


class SupervisionDataset(Dataset):
    def __init__(self, rows, tokenizer, supervision, resolution, random_flip, seed, constant_prompt,
                 sparse_bank=None):
        self.rows = rows
        self.tokenizer = tokenizer
        self.supervision = supervision
        self.seed = seed
        self.constant_prompt = constant_prompt
        self.epoch = 0
        self.caption_donors = list(range(len(rows)))
        transform_list = [transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR)]
        transform_list.append(transforms.CenterCrop(resolution))
        if random_flip:
            transform_list.append(transforms.RandomHorizontalFlip())
        transform_list.extend([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
        self.transform = transforms.Compose(transform_list)
        self.class_indices = defaultdict(list)
        for index, row in enumerate(rows):
            self.class_indices[row["synset"]].append(index)
        self.sparse_bank_sources = None
        if supervision == "sparse_unpaired":
            if sparse_bank is None:
                raise ValueError("sparse_unpaired supervision requires --sparse-bank")
            by_relative = {row["relative"]: index for index, row in enumerate(rows)}
            classes = sparse_bank.get("classes", {})
            if set(classes) != set(self.class_indices):
                raise ValueError("Sparse bank classes differ from the training classes")
            self.sparse_bank_sources = {}
            for synset, entries in classes.items():
                sources = []
                for entry in entries:
                    relative = str(entry["relative"]).replace("\\", "/")
                    if relative not in by_relative:
                        raise ValueError(f"Sparse-bank source is absent: {relative}")
                    source = by_relative[relative]
                    if self.rows[source]["caption"] != str(entry["caption"]).strip():
                        raise ValueError(f"Sparse-bank caption differs from metadata: {relative}")
                    sources.append(source)
                self.sparse_bank_sources[synset] = sources
        self.set_epoch(0)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)
        self.caption_donors = list(range(len(self.rows)))
        if self.supervision == "unpaired":
            self.caption_donors = build_unpaired_donors(self.class_indices, self.seed, self.epoch)
        elif self.supervision == "sparse_unpaired":
            self.caption_donors = build_sparse_bank_donors(
                self.class_indices, self.sparse_bank_sources, self.seed, self.epoch
            )

    def assignment_audit(self):
        return {
            "epoch": self.epoch,
            "supervision": self.supervision,
            "images": len(self.rows),
            "self_pairs": sum(i == donor for i, donor in enumerate(self.caption_donors)),
            "caption_multiset_preserved_by_class": self.supervision != "sparse_unpaired" and all(
                sorted(self.caption_donors[i] for i in indices) == sorted(indices)
                for indices in self.class_indices.values()
            ),
            "sparse_bank_balanced_by_class": (
                all(
                    max([self.caption_donors[i] for i in indices].count(source) for source in self.sparse_bank_sources[key])
                    - min([self.caption_donors[i] for i in indices].count(source) for source in self.sparse_bank_sources[key]) <= 1
                    for key, indices in self.class_indices.items()
                ) if self.sparse_bank_sources else None
            ),
            "assignment_sha256": __import__("hashlib").sha256(
                json.dumps(self.caption_donors).encode("utf-8")
            ).hexdigest(),
        }

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(row["path"]) as image:
            pixel_values = self.transform(ImageOps.exif_transpose(image).convert("RGB"))
        if self.supervision == "empty":
            text = ""
        elif self.supervision == "constant":
            text = self.constant_prompt
        elif self.supervision == "label":
            text = IMAGENET2012_CLASSES[row["synset"]]
        else:
            text = self.rows[self.caption_donors[index]]["caption"]
        input_ids = self.tokenizer(
            text,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids[0]
        return {"pixel_values": pixel_values, "input_ids": input_ids}


class LossBins:
    def __init__(self, bins, timesteps, device):
        self.bins = bins
        self.timesteps = timesteps
        self.device = device
        self.reset()

    def reset(self):
        self.loss_sum = torch.zeros(self.bins, device=self.device, dtype=torch.float64)
        self.count = torch.zeros(self.bins, device=self.device, dtype=torch.float64)

    def add(self, timesteps, losses):
        indices = torch.clamp((timesteps.long() * self.bins) // self.timesteps, max=self.bins - 1)
        self.loss_sum.scatter_add_(0, indices, losses.detach().double())
        self.count.scatter_add_(0, indices, torch.ones_like(losses, dtype=torch.float64))

    def reduce_payload(self, accelerator):
        sums = accelerator.reduce(self.loss_sum, reduction="sum").cpu()
        counts = accelerator.reduce(self.count, reduction="sum").cpu()
        rows = []
        for index in range(self.bins):
            low = index * self.timesteps // self.bins
            high = (index + 1) * self.timesteps // self.bins - 1
            rows.append({
                "bin": index,
                "timestep_low": low,
                "timestep_high": high,
                "loss": float(sums[index] / counts[index]) if counts[index] else None,
                "samples": int(counts[index]),
            })
        return rows

    def has_samples(self):
        return bool(self.count.sum().item())


def collate(examples):
    return {
        "pixel_values": torch.stack([item["pixel_values"] for item in examples]).contiguous().float(),
        "input_ids": torch.stack([item["input_ids"] for item in examples]),
    }


def checkpoint_limit(output_dir, limit):
    if not limit:
        return
    checkpoints = sorted(
        (path for path in Path(output_dir).glob("checkpoint-*") if path.is_dir()),
        key=lambda path: int(path.name.split("-")[-1]),
    )
    for path in checkpoints[:-limit]:
        import shutil
        shutil.rmtree(path)


def append_jsonl(path, payload):
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def sanitize_resume_outputs(output_dir, global_step, first_epoch):
    interval_path = Path(output_dir) / "timestep_loss_intervals.jsonl"
    if interval_path.is_file():
        kept = []
        for line in interval_path.read_text(encoding="utf-8").splitlines():
            if line.strip() and int(json.loads(line)["global_step"]) <= global_step:
                kept.append(line)
        interval_path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    epoch_path = Path(output_dir) / "timestep_loss_epochs.csv"
    if epoch_path.is_file():
        with epoch_path.open("r", encoding="utf-8", newline="") as handle:
            rows = [row for row in csv.DictReader(handle) if int(row["global_step"]) <= global_step]
        fields = ("epoch", "global_step", "bin", "timestep_low", "timestep_high", "loss", "samples")
        with epoch_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    audit_path = Path(output_dir) / "caption_assignment_audit.jsonl"
    if audit_path.is_file():
        kept = [
            line for line in audit_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and int(json.loads(line)["epoch"]) < first_epoch
        ]
        audit_path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    project = ProjectConfiguration(project_dir=str(output_dir), logging_dir=str(output_dir / "logs"))
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=None if args.mixed_precision == "no" else args.mixed_precision,
        project_config=project,
    )
    set_seed(args.seed)

    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model, subfolder="scheduler")
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    rows = read_records(args.train_root, args.caption_file)
    sparse_bank = None
    if args.supervision == "sparse_unpaired":
        if not args.sparse_bank:
            raise ValueError("--sparse-bank is required for sparse_unpaired supervision")
        sparse_bank = json.loads(Path(args.sparse_bank).read_text(encoding="utf-8"))
    dataset = SupervisionDataset(
        rows,
        tokenizer,
        args.supervision,
        args.resolution,
        args.random_flip,
        args.seed,
        args.constant_prompt,
        sparse_bank,
    )
    dataloader = DataLoader(
        dataset,
        shuffle=True,
        collate_fn=collate,
        batch_size=args.train_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=False,
    )
    optimizer = torch.optim.AdamW(unet.parameters(), lr=args.learning_rate)
    updates_per_epoch = math.ceil(
        len(dataloader) / (args.gradient_accumulation_steps * accelerator.num_processes)
    )
    max_steps = args.num_train_epochs * updates_per_epoch
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=max_steps * accelerator.num_processes,
    )
    ema = LocalEMAModel(unet.parameters()) if args.use_ema else None
    if ema is not None:
        accelerator.register_for_checkpointing(ema)

    unet, optimizer, dataloader, lr_scheduler = accelerator.prepare(unet, optimizer, dataloader, lr_scheduler)
    prepared_updates_per_epoch = math.ceil(len(dataloader) / args.gradient_accumulation_steps)
    if prepared_updates_per_epoch != updates_per_epoch:
        updates_per_epoch = prepared_updates_per_epoch
        max_steps = args.num_train_epochs * updates_per_epoch
    weight_dtype = torch.float16 if accelerator.mixed_precision == "fp16" else (
        torch.bfloat16 if accelerator.mixed_precision == "bf16" else torch.float32
    )
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    if ema is not None:
        ema.to(accelerator.device)

    global_step = 0
    first_epoch = 0
    resume_batches = 0
    resume_path = None
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint == "latest":
            candidates = sorted(
                output_dir.glob("checkpoint-*"), key=lambda path: int(path.name.split("-")[-1])
            )
            resume_path = candidates[-1] if candidates else None
        else:
            resume_path = Path(args.resume_from_checkpoint)
        if resume_path:
            accelerator.load_state(str(resume_path))
            if ema is not None:
                ema.to(accelerator.device)
            global_step = int(resume_path.name.split("-")[-1])
            first_epoch = global_step // updates_per_epoch
            resume_batches = (global_step % updates_per_epoch) * args.gradient_accumulation_steps
            if accelerator.is_main_process:
                sanitize_resume_outputs(output_dir, global_step, first_epoch)
            accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        config = vars(args).copy()
        config.update({
            "world_size": accelerator.num_processes,
            "effective_batch_size": args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps,
            "training_images": len(rows),
            "scheduler_timesteps": noise_scheduler.config.num_train_timesteps,
        })
        (output_dir / "training_config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    interval_bins = LossBins(args.timestep_bins, noise_scheduler.config.num_train_timesteps, accelerator.device)
    epoch_rows = []
    existing_epoch_path = output_dir / "timestep_loss_epochs.csv"
    if accelerator.is_main_process and existing_epoch_path.is_file():
        with existing_epoch_path.open("r", encoding="utf-8", newline="") as handle:
            epoch_rows = list(csv.DictReader(handle))
    progress_bar = tqdm(
        total=max_steps,
        initial=global_step,
        disable=not accelerator.is_local_main_process,
        desc=f"{args.supervision} optimizer steps",
        dynamic_ncols=True,
    )
    for epoch in range(first_epoch, args.num_train_epochs):
        dataset.set_epoch(epoch)
        if accelerator.is_main_process:
            append_jsonl(output_dir / "caption_assignment_audit.jsonl", dataset.assignment_audit())
        unet.train()
        epoch_bins = LossBins(args.timestep_bins, noise_scheduler.config.num_train_timesteps, accelerator.device)
        epoch_dataloader = dataloader
        if epoch == first_epoch and resume_batches:
            epoch_dataloader = accelerator.skip_first_batches(dataloader, resume_batches)
        for batch in epoch_dataloader:
            with accelerator.accumulate(unet):
                with torch.no_grad():
                    latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                    encoder_hidden_states = text_encoder(batch["input_ids"])[0]
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                prediction = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                target = noise_scheduler.get_velocity(latents, noise, timesteps) if noise_scheduler.config.prediction_type == "v_prediction" else noise
                sample_losses = F.mse_loss(prediction.float(), target.float(), reduction="none").mean(dim=(1, 2, 3))
                loss = sample_losses.mean()
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                interval_bins.add(timesteps, sample_losses)
                epoch_bins.add(timesteps, sample_losses)

            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)
                progress_bar.set_postfix(
                    epoch=epoch + 1,
                    loss=f"{loss.detach().item():.4f}",
                    lr=f"{lr_scheduler.get_last_lr()[0]:.2e}",
                )
                if global_step % args.loss_log_steps == 0:
                    payload = {"epoch": epoch, "global_step": global_step, "bins": interval_bins.reduce_payload(accelerator)}
                    if accelerator.is_main_process:
                        append_jsonl(output_dir / "timestep_loss_intervals.jsonl", payload)
                    interval_bins.reset()
                if ema is not None:
                    ema.step(unet.parameters())
                if args.checkpointing_steps and global_step % args.checkpointing_steps == 0:
                    accelerator.save_state(str(output_dir / f"checkpoint-{global_step}"))
                    checkpoint_limit(output_dir, args.checkpoints_total_limit)

        bins = epoch_bins.reduce_payload(accelerator)
        if interval_bins.has_samples():
            payload = {
                "epoch": epoch,
                "global_step": global_step,
                "interval_kind": "epoch_tail",
                "bins": interval_bins.reduce_payload(accelerator),
            }
            if accelerator.is_main_process:
                append_jsonl(output_dir / "timestep_loss_intervals.jsonl", payload)
            interval_bins.reset()
        if accelerator.is_main_process:
            for row in bins:
                row.update({"epoch": epoch, "global_step": global_step})
                epoch_rows.append(row)
            with (output_dir / "timestep_loss_epochs.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=("epoch", "global_step", "bin", "timestep_low", "timestep_high", "loss", "samples"))
                writer.writeheader()
                writer.writerows(epoch_rows)

    progress_bar.close()

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        trained_unet = accelerator.unwrap_model(unet)
        if ema is not None:
            ema.copy_to(trained_unet.parameters())
        pipeline = StableDiffusionPipeline.from_pretrained(
            args.pretrained_model,
            unet=trained_unet,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            safety_checker=None,
            requires_safety_checker=False,
        )
        pipeline.save_pretrained(output_dir)
        summary = {
            "complete": True,
            "global_steps": global_step,
            "epochs": args.num_train_epochs,
            "supervision": args.supervision,
            "seed": args.seed,
            "sparse_bank": str(Path(args.sparse_bank).resolve()) if args.sparse_bank else None,
            "loss_files": ["timestep_loss_intervals.jsonl", "timestep_loss_epochs.csv"],
        }
        (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    accelerator.end_training()


if __name__ == "__main__":
    main()
