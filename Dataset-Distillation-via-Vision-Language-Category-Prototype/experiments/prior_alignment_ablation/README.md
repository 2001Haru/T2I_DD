# Prior-alignment fine-tuning ablation

This experiment asks whether diffusion fine-tuning is necessary for
dataset-specific text to improve VLCP-style dataset distillation. It uses a
paired 2x2 design:

| condition | diffusion model | generation prompt |
|---|---|---|
| `frozen_label` | frozen SD 1.5 | ImageNet class name |
| `frozen_dcs` | frozen SD 1.5 | VLCP DCS |
| `finetuned_label` | author VLCP checkpoint | ImageNet class name |
| `finetuned_dcs` | author VLCP checkpoint | VLCP DCS |

All four cells reuse one fixed prototype JSON and receive matched generation
noise. The primary statistic is:

```text
(finetuned_dcs - finetuned_label) - (frozen_dcs - frozen_label)
```

A consistently positive interaction means that dataset-domain fine-tuning
makes DCS more useful. A generic improvement in both fine-tuned cells is only a
fine-tuning main effect and does not establish that text needs fine-tuning.

## Recommended pilot

Use the author's `ImageNette_seed0` checkpoint and prepared ImageNette dataset.
Run one generation seed and three paired classifier repeats. This avoids LLaVA
inference and all diffusion training. Add generation seed 1 only if the first
interaction is large enough to justify confirmation.

## 1. Download author resources

The author release is linked from the project README:

- Google Drive: <https://drive.google.com/drive/folders/1qENjFnh--QmGsceC0jFaUpGCindK7nCB?usp=sharing>
- GoFile mirror: <https://gofile.me/6UrIQ/zaQO2ICY5>

Download only these two items instead of the complete release:

1. The original **ImageNette dataset with text metadata**. It must contain
   `train/`, `val/`, and `train/metadata.jsonl`.
2. The fine-tuned model **ImageNette_seed0**. It must be a complete Diffusers
   pipeline containing `model_index.json`, `unet/`, `vae/`, `text_encoder/`,
   `tokenizer/`, and `scheduler/`.

Selecting those two folders in the Google Drive or GoFile web interface is the
safest option. If only terminal access is available, `gdown` can download the
shared Google Drive folder, but this may fetch the much larger full release:

```bash
python -m pip install -U gdown
mkdir -p /linxi/downloads/VLCP_release
gdown --folder \
  "https://drive.google.com/drive/folders/1qENjFnh--QmGsceC0jFaUpGCindK7nCB" \
  --output /linxi/downloads/VLCP_release \
  --remaining-ok
```

Place them as follows (the extracted archive may add one extra parent folder):

```text
/linxi/dataset/ImageNette_VLCP/
  train/
    metadata.jsonl
    n01440764/
    ...
  val/
    n01440764/
    ...

/linxi/models/VLCP/ImageNette_seed0/
  model_index.json
  scheduler/
  text_encoder/
  tokenizer/
  unet/
  vae/
```

Locate the true model root after extraction with:

```bash
find /linxi/models/VLCP -name model_index.json -print
```

`FINETUNED_MODEL` must point to the parent of `model_index.json`, not to a
training `checkpoint-500/` directory and not to the archive's outer folder.

## 2. Download frozen SD 1.5

VLCP uses Stable Diffusion 1.5, not SDXL. To match the paper, use the exact base
repository named in its scripts. The repository requires accepting the model
license on Hugging Face once:

1. Open <https://huggingface.co/benjamin-paine/stable-diffusion-v1-5> and accept access.
2. Create a read token in Hugging Face settings.
3. On the cloud machine run:

```bash
python -m pip install -U huggingface_hub
hf auth login

hf download benjamin-paine/stable-diffusion-v1-5 \
  --local-dir /linxi/models/VLCP/stable-diffusion-v1-5
```

The public mirror `stable-diffusion-v1-5/stable-diffusion-v1-5` can be used if
access to the author-named mirror is unavailable, but the author-named model is
preferred for the cleanest comparison.

## 3. Environment

The experiment does not use VLCP's copied custom Diffusers package. It uses the
standard `AutoencoderKL` and `StableDiffusionImg2ImgPipeline`, which implement
the same scaled VAE encoding and img2img-from-latent operations used here.

Inside the existing CUDA/PyTorch environment install missing packages without
reinstalling PyTorch:

```bash
python -m pip install \
  "diffusers>=0.30,<0.39" \
  "transformers>=4.40,<5" \
  "accelerate>=0.30" \
  safetensors huggingface_hub \
  scikit-learn scipy nltk matplotlib ipdb

python -m nltk.downloader punkt_tab stopwords
```

## 4. Validate paths before GPU work

```bash
cd /linxi/T2I_DD/Dataset-Distillation-via-Vision-Language-Category-Prototype

python experiments/prior_alignment_ablation/validate_setup.py \
  --data-root /linxi/dataset/ImageNette_VLCP \
  --base-model /linxi/models/VLCP/stable-diffusion-v1-5 \
  --finetuned-model /linxi/models/VLCP/ImageNette_seed0
```

The validator checks all ten synsets, image-caption coverage, validation data,
and the required Diffusers components. Do not start generation until all three
lines report `[OK]`.

## 5. Run the low-cost pilot

```bash
export DATA_ROOT=/linxi/dataset/ImageNette_VLCP
export BASE_MODEL=/linxi/models/VLCP/stable-diffusion-v1-5
export FINETUNED_MODEL=/linxi/models/VLCP/ImageNette_seed0
export RUN_ID=author_checkpoint_pilot_v0
export RUN_ROOT=/linxi/T2I_DD/vlcp_ablation_runs/$RUN_ID

FINETUNE=false \
GENERATION_SEEDS=0 \
bash experiments/prior_alignment_ablation/run_ablation.sh
```

This performs only:

1. One VAE pass over ImageNette, LOF, KMeans, and DCS selection.
2. Four matched synthetic datasets of 100 images each.
3. Three ResNet-AP classifier repeats per condition.
4. Automatic computation of main effects and the interaction.

No `TRAIN_SCRIPT` is required and the author model is never modified. The
prototype is extracted once with the frozen model's VAE and reused in all four
conditions. The official VLCP training script freezes the VAE, so this removes
prototype variation without removing the intended UNet fine-tuning effect.

## 6. Resume and confirm

Resume a matching interrupted run with:

```bash
FINETUNE=false \
GENERATION_SEEDS=0 \
RESUME=true \
bash experiments/prior_alignment_ablation/run_ablation.sh
```

The same exported paths and `RUN_ID` must still be present. Matching manifests
are reused; incompatible generation settings and incomplete classifier logs are
rejected instead of appended.

If seed 0 gives a meaningful positive interaction, add a second generation
seed in the same run:

```bash
FINETUNE=false \
GENERATION_SEEDS="0 1" \
RESUME=true \
bash experiments/prior_alignment_ablation/run_ablation.sh
```

Results are written to `$RUN_ROOT/summary/summary.json` and `summary.csv`.

## Fallback: build ImageNette from the existing ImageNet archive

Use this only if the author's prepared ImageNette data cannot be downloaded.

```bash
python experiments/prior_alignment_ablation/prepare_imagenette.py \
  --source-root /zhangchi/imagenet_512/images \
  --validation-root /linxi/dataset/imagenet/validation/val \
  --output-root /linxi/dataset/imagenette_vlcp
```

This creates links rather than copying ImageNet and writes
`llava_questions.jsonl`. Generate descriptions with LLaVA and merge them:

```bash
python /path/to/LLaVA/llava/eval/model_vqa.py \
  --model-path /linxi/models/CoDA/llava \
  --question-file /linxi/dataset/imagenette_vlcp/llava_questions.jsonl \
  --image-folder /linxi/dataset/imagenette_vlcp/train \
  --answers-file /linxi/dataset/imagenette_vlcp/llava_answers.jsonl

python experiments/prior_alignment_ablation/merge_llava_answers.py \
  --questions /linxi/dataset/imagenette_vlcp/llava_questions.jsonl \
  --answers /linxi/dataset/imagenette_vlcp/llava_answers.jsonl \
  --output /linxi/dataset/imagenette_vlcp/train/metadata.jsonl
```

This fallback tests the same mechanism, but captions may differ from the
author's exact release due to LLaVA version and decoding differences.
