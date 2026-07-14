**[ICLR 2026]** This repository contains the official implementation of the paper: **"CoDA: From Text-to-Image Diffusion Models to Training-Free Dataset Distillation"**.

## 🔥 News
- **[2026.01.26]** 🎉 We are thrilled to announce that **CoDA** has been accepted to **ICLR 2026**!
- **[2025.12.04]** CoDA is released on [arXiv](https://arxiv.org/abs/2512.03844).

## 📖 Introduction

CoDA is a novel dataset distillation framework leveraging an off-the-shelf text-to-image model (SDXL). Instead of relying on diffusion models pre-trained on the target dataset (e.g., utilizing an ImageNet-trained DiT to distill ImageNet), we introduce "Distribution Discovery" and "Distribution Alignment" to bridge the distribution gap between general generative priors and specific domains. This achieves SOTA performance without the prohibitive cost of pre-training, establishing CoDA as a truly universal solution capable of performing dataset distillation tasks on any arbitrary dataset.

## 🛠️ Requirements

To install the required dependencies, run:

```
pip install -r requirements.txt
```

## 🚀 Usage

Please make sure to navigate to the project root directory first:

```
cd CoDA
scripts/CoDA.sh
```

## Cluster-caption extension

This repository also supports the cluster-aware prompt variant. It captions each
representative image in `real_images/<class_id>/<index>.png` with LLaVA, then
uses the matching caption together with the original CoDA class name while SDXL
generates the corresponding synthetic image.

The default caption model is the Transformers-compatible
`llava-hf/llava-1.5-7b-hf`. It is downloaded only when
`--generate_cluster_captions` is used, and is cached by Hugging Face on the
cloud machine.

```bash
# Original CoDA prompt: the first ImageNet class descriptor only.
python CoDA_main.py ... --calcu_features --calcu_cluster --generate_images

# Cluster-aware prompt: caption representatives, then use the matching caption.
python CoDA_main.py ... --calcu_features --calcu_cluster \
    --generate_cluster_captions --generate_images --use_cluster_captions
```

Captions are stored in `cluster_captions.json` inside the experiment directory.
They are validated before generation and reused on later runs. Original CoDA
images remain in `generated_images/`; caption-conditioned images default to
`generated_images_vlm_caption/`. The latter behavior can be overridden with
`--generated_images_dirname` for ablations, and a different caption manifest
can be selected with `--cluster_caption_file`.

## ImageNet dataset.json layout

CoDA automatically supports ImageNet sources stored as numbered image shards
with a `dataset.json` label manifest, such as `00000/img00000000.png` paired
with an ImageNet-1K class index. No image relocation or LMDB conversion is
needed. Set `IMAGENET_TRAIN_FOLDER` in `scripts/CoDA.sh` directly to the
directory that contains both the numbered folders and `dataset.json`.

The manifest is treated as one split. Downstream evaluation still requires a
separate validation root, configured by `IMAGENET_VAL_FOLDER`; it may use the
same `dataset.json` layout or the conventional `<wnid>/<image>` layout. Do not
point the validation setting at the training manifest.

## Timing and isolated outputs

Each run writes a method-specific timing record under
`results/.../timings/coda_baseline.json` or
`results/.../timings/vlm_caption.json`. Timings include feature extraction,
clustering, caption generation, synthetic image generation, and downstream
training. After both methods have completed, `scripts/CoDA.sh` prints a JSON
comparison that reports caption time relative to baseline generation time.

Downstream checkpoints and logs are isolated under method-specific directories
in `trained_results/.../coda_baseline-*` and `trained_results/.../vlm_caption-*`.

## Model download validation

`MODEL_FOLDER` must contain complete Diffusers repositories at
`sdxl-base/` and `sdxl-refiner/`, each including `model_index.json`. CoDA now
stops before generation when either directory is partial, and
`scripts/CoDA.sh` stops immediately after any failed stage so invalid or
incomplete synthetic data is never passed to downstream training.

## Class-focused caption prompts

The current caption variant explicitly names the target class when querying
LLaVA and asks only for its physical appearance. The resulting SDXL prompt is
`An natural photo of a {class_name}, {caption}, centered object.`. Runs from
`scripts/CoDA.sh` use the isolated method tag `vlm_caption_class_focused`, so
captions, generated images, timings, and classifier outputs do not overwrite
the earlier generic-caption ablation.

Set `RUN_DOWNSTREAM_TRAINING=false` and `GENERATE_IMAGES=false` to generate and
inspect captions before committing GPU time to SDXL generation and classifier
training.

## Guidance conflict diagnostics

Run `scripts/guidance_conflict_sweep.sh` to regenerate baseline, generic-caption
v0, and class-focused-caption v1 with identical seeds while recording
per-sample, per-step guidance measurements. Each timestamped run stores raw CSV
values, JSON summaries, per-variant plots, and a combined comparison under
`results/.../guidance_conflict_runs/<UTC timestamp>/`; earlier experiments are
never overwritten.

The recorded directions are `g_text = epsilon_conditional -
epsilon_unconditional` and `g_img = delta_epsilon_CoDA`, both in SDXL noise
prediction space before CFG scaling. The diagnostics include cosine similarity
and `q_t = ||g_text||_2 / ||g_img||_2`, as well as the negative projection
ratio `kappa_t`, which measures how much text guidance is cancelled by image
guidance. The sweep defaults to generation seed 1 while retaining evaluation
seed 0, then trains all three downstream classifiers. Set `REFERENCE_RUN_DIR`
to the earlier seed-0 run directory to create a six-curve cross-seed plot.

If generation completed but downstream training did not, resume only the three
classifier runs with `RUN_ID=<completed run> bash scripts/train_guidance_run.sh`.

## Conflict-aware projection ablation

The optional `--conflict_projection_alpha` removes only the component of CoDA
image guidance that opposes text guidance. `0` is exactly the original method,
`0.5` removes half of that negative component, and `1.0` removes it completely.
Gamma remains unchanged. Guidance CSV and plots store both the interaction
before projection and the effective interaction used by the sampler.

Run the focused-caption matrix at generation seeds 0 and 1 with alpha 0.5 and
1.0 using:

```bash
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
bash scripts/conflict_projection_sweep.sh
```

Results are isolated under `results/.../projection_runs/<RUN_ID>/seed_<SEED>`
and `trained_results/projection_runs/imageA/<RUN_ID>`. Existing baseline and
unprojected focused-caption runs are reused for the final analysis and are not
regenerated. Optionally set `SEED0_REFERENCE_RUN_DIR` and
`SEED1_REFERENCE_RUN_DIR` to their existing guidance run directories; the
comparison plots will then contain baseline, alpha 0, alpha 0.5, and alpha 1.0.
To generate without classifier training, set
`RUN_DOWNSTREAM_TRAINING=false`. Resume only downstream training with:

```bash
RUN_ID=<completed projection run> bash scripts/train_projection_run.sh
```

## Fixed ImageB validation experiment

`scripts/imageB_projection_experiment.sh` uses the repository's predefined
`misc/imagenet-b.txt` classes without resampling or manual filtering. It first
validates train/validation counts and paths, computes ImageB features and
clusters once, generates one focused-caption manifest, and runs the following
matrix at generation seeds 0 and 1:

- original CoDA baseline;
- focused caption with projection alpha 0;
- focused caption with projection alpha 0.5.

Run the complete experiment with:

```bash
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
bash scripts/imageB_projection_experiment.sh
```

The timestamped outputs are stored under `results/imageB/.../imageB_runs/` and
`trained_results/imageB_runs/`. Every classifier GPU/training seed writes its
best-epoch `per_class_accuracy_best.json`; the method directory also contains
`per_class_accuracy_all_seeds.json` with the per-class mean and standard
deviation. The run's `summary/` directory contains the complete method table,
paired improvements, and `per_class_comparison.csv`. If preparation already exists, reuse it with
`CALCULATE_FEATURES=false CALCULATE_CLUSTER=false GENERATE_CAPTIONS=false`.
Resume only downstream training with:

```bash
RUN_ID=<completed ImageB run> bash scripts/train_imageB_run.sh
```

## Class-level conflict diagnosis and kappa cap

For a completed fixed ImageB run, relate each class's pre-projection guidance
conflict to its downstream accuracy with:

```bash
RUN_ID=<completed ImageB run> bash scripts/analyze_imageB_guidance.sh
```

The resulting `class_guidance_diagnostics/` directory contains raw class-level
CSV statistics, Pearson/Spearman correlations, four annotated accuracy/conflict
scatter plots, and ten-class cosine/kappa curves. Denoising steps are divided
into normalized early, middle, and late thirds.

The alternative `--conflict_projection_kappa_cap` leaves image guidance intact
when its cancellation ratio is below the cap and removes only the excess. It is
mutually exclusive with fixed-alpha projection. After reviewing the diagnostic
results, run the preregistered cap of 0.3 on the existing ImageA or ImageB
clusters and captions with:

```bash
SPEC=imageA bash scripts/kappa_cap_experiment.sh
SPEC=imageB bash scripts/kappa_cap_experiment.sh
```

Both generation seeds 0 and 1 are used by default. Resume training without
regeneration using:

```bash
SPEC=imageB RUN_ID=<completed cap run> bash scripts/train_kappa_cap_run.sh
```
