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

## Multi-image local-mode caption experiment

The `montage_neighbors` caption mode retrieves the four nearest training images
to each saved cluster center in the original SDXL VAE feature space. It stores
them as a `2x2` montage and asks LLaVA for physical attributes shared across
multiple tiles. The montage is used only by LLaVA; SDXL still receives CoDA's
original representative as image guidance. Each caption manifest records the
four source paths and squared VAE distances for inspection and provenance.
LLaVA is instructed to return a single description of at most 35 words without
referring to the montage. The manifest preserves its response in `raw_captions`
and stores a layout-reference-free, length-normalized version in `captions`.
Before sampling, the complete SDXL prompt is also checked against both CLIP
tokenizers and the caption is shortened further if either 77-token limit would
be exceeded. Each generated dataset writes `prompt_records_gpu*.json` with the
effective prompt and both token counts for direct verification.

The controlled experiment compares original CoDA, the existing single-image
focused caption, and the four-image common-mode caption. Projection is disabled
for all three methods, and generation seeds 0 and 1 are used by default:

```bash
SPEC=imageA bash scripts/multiview_caption_experiment.sh
SPEC=imageB bash scripts/multiview_caption_experiment.sh
```

Existing feature and cluster artifacts are reused by default. Set
`CALCULATE_FEATURES=true CALCULATE_CLUSTER=true` only for a new class subset.
Before SDXL generation, inspect `caption_montages/` and
`captions_montage_common_mode.json` inside the timestamped run directory.
Outputs are isolated under `results/.../multiview_caption_runs/<RUN_ID>` and
`trained_results/multiview_caption_runs/<SPEC>/<RUN_ID>`. To generate without
training, set `RUN_DOWNSTREAM_TRAINING=false`; resume only training with:

```bash
SPEC=imageA RUN_ID=<completed multiview run> \
bash scripts/train_multiview_caption_run.sh
```

To pause for caption inspection before spending time on SDXL, use a fixed ID:

```bash
MULTIVIEW_RUN_ID=imageA_multiview_v0 SPEC=imageA \
RUN_GENERATION=false RUN_DOWNSTREAM_TRAINING=false \
bash scripts/multiview_caption_experiment.sh

MULTIVIEW_RUN_ID=imageA_multiview_v0 SPEC=imageA RESUME_RUN=true \
GENERATE_CAPTIONS=false bash scripts/multiview_caption_experiment.sh
```

Resume mode still refuses to overwrite any existing generated dataset.
To replace captions from a paused pre-generation run after changing the prompt
policy, add `OVERWRITE_CAPTIONS=true` when resuming.

## Final prompt and refinement controls

The final control matrix separates representative selection, VAE fidelity,
diffusion refinement, and text complexity across ImageA and ImageB. ImageA uses
generation seed 0 and ImageB uses seed 1 by default, and each evaluates:

- the selected real representatives;
- direct VAE reconstruction of the saved representative latents;
- CoDA with an empty prompt;
- CoDA with the generic prompt `a natural photo`;
- original CoDA with the class-name prompt;
- CoDA with the montage caption.

ImageA reuses both generated datasets and classifier results for class-prompt
and montage conditions from `imageA_multiview_v0/seed_0`. It generates only the
empty and generic conditions. ImageB generates all four diffusion conditions,
while real representatives and VAE reconstructions never run diffusion.

```bash
bash scripts/final_prompt_controls_experiment.sh
```

Set `RUN_DOWNSTREAM_TRAINING=false` to finish generation first. Resume or retry
only incomplete classifier runs with the printed run ID:

```bash
RUN_ID=<final control run ID> bash scripts/train_final_prompt_controls.sh
```

The summary is stored in
`trained_results/final_prompt_controls/<RUN_ID>/summary/`, including overall
method differences for both subsets and a per-class CSV. Override
`IMAGEA_REFERENCE_RUN_ID`, `IMAGEA_SEED`, or `IMAGEB_SEED` only when selecting a
different completed reference run.

## SDXL prior compatibility diagnostic

The Prior Compatibility Score (PCS) tests whether SDXL's class-conditioned
denoiser is a useful local model for a saved CoDA representative. For the same
representative latent, timestep, and sampled noise, it compares conditional and
unconditional noise-prediction errors:

```text
PCS = (MSE_unconditional - MSE_class_conditioned) / MSE_unconditional
```

A positive value means that the ImageNet class name helps SDXL explain that
noisy representative. The primary downstream target is fixed in advance as
`class_prompt_mean - vae_reconstruction_mean`. This is the cleanest available
comparison because both methods start from the same saved cluster center, while
only the class-prompt condition runs diffusion refinement. No PCS threshold is
fit on the 20 classes: the diagnostic reports rank correlation and separately
tests the natural rule `PCS > 0`.

Run ImageA and ImageB concurrently on two GPUs after the final prompt-control
summary has been produced:

```bash
FINAL_CONTROL_RUN_ID=final_prompt_controls_v0 \
PCS_RUN_ID=pcs_v0 \
bash scripts/pcs_diagnostic_experiment.sh
```

Set `FINAL_CONTROL_RUN_ID` to the directory name under
`trained_results/final_prompt_controls/` that contains
`summary/per_class_comparison.csv`. GPU assignment can be changed with
`PCS_GPU_A` and `PCS_GPU_B`. The default evaluates all ten representatives per
class at eight fixed training timesteps and does not generate images or train a
classifier. Follow progress with:

```bash
tail -f results/pcs_diagnostics/pcs_v0/imageA.log
tail -f results/pcs_diagnostics/pcs_v0/imageB.log
```

Each subset stores `pcs_raw.csv`, `pcs_per_class.csv`,
`pcs_per_class_timestep.csv`, `pcs_per_timestep.csv`, and `pcs_config.json`. The combined `analysis/`
directory stores the merged class table, Pearson/Spearman statistics, the fixed
zero-threshold decision test, and `pcs_accuracy_relationship.png`. Treat the
separate ImageA and ImageB Spearman correlations for `delta_class_vs_vae` as
the main result; the normalized combined correlation and other accuracy deltas
are supporting diagnostics. `pcs_timestep_correlations.png` checks whether the
relationship is concentrated at a particular noise level; because it compares
multiple timesteps, treat it as exploratory rather than the primary test.

The analyzer also computes the symmetric logarithmic score directly from the
saved unconditional and conditional MSE values:

```text
PCS_log = mean(log(MSE_unconditional) - log(MSE_class_conditioned))
```

An existing PCS run does not need any new SDXL inference. Reanalyze `pcs_v0`
without overwriting its original `analysis/` directory using:

```bash
PCS_RUN_ID=pcs_v0 bash scripts/reanalyze_pcs_log.sh
```

The new outputs are stored in `results/pcs_diagnostics/pcs_v0/analysis_log/`.
The command prints the old and logarithmic Spearman correlations together with
the fixed `PCS_log > 0` selection accuracy. Detailed results include
`pcs_log_per_class.csv`, `pcs_log_accuracy_relationship.png`, and a two-panel
linear/logarithmic timestep comparison.

## Gradient-guided candidate selection

The gradient-selection experiment treats each real CoDA representative and its
matching class-prompt Diffusion output as a pair. For each pair it builds an
independent local target from 32 same-class ImageNet training images in the
cached SDXL-VAE feature space. The representative source itself is always
excluded from this neighborhood, preventing a trivial self-match.

For four random ResNet-AP-10 initializations and four augmentation draws, the
selector analytically computes the final-layer cross-entropy gradient from the
network feature and softmax residual. It records the candidate-to-neighborhood
cosine score `G`, then enumerates all `2^10` ways to choose exactly one image
from each real/Diffusion pair. The chosen set minimizes mean relative gradient
error across the 16 runs. A random control is constructed with exactly the same
number of Diffusion images per class as the selected set.

The default paths reuse `final_prompt_controls_v0` and the existing ImageA
multiview baseline. No SDXL or LLaVA inference is run:

```bash
GRADIENT_SELECTION_RUN_ID=gm_v0 \
FINAL_CONTROL_RUN_ID=final_prompt_controls_v0 \
bash scripts/gradient_selection_experiment.sh
```

ImageA and ImageB gradient computation runs concurrently on GPUs 0 and 1.
Progress and failures are written to:

```bash
tail -f results/gradient_selection_runs/gm_v0/imageA.log
tail -f results/gradient_selection_runs/gm_v0/imageB.log
```

Set `RUN_DOWNSTREAM_TRAINING=false` to stop after selection. Resume only the
classifier stage with:

```bash
GRADIENT_SELECTION_RUN_ID=gm_v0 \
FINAL_CONTROL_RUN_ID=final_prompt_controls_v0 \
bash scripts/train_gradient_selection_run.sh
```

Each subset stores `neighbor_manifest.json`, raw and summarized pair scores,
`class_diagnostics.csv`, diagnostic plots, and the two image folders
`selected_gm/` and `selected_random_matched/`. Classifier results and the
cross-subset summary are stored under
`trained_results/gradient_selection_runs/<RUN_ID>/`. Existing run directories
and incomplete classifier directories are never overwritten.

The preregistered primary configuration is `K=32`, class-standardized VAE L2
distance, uniform neighborhood weights, four network seeds, and four
augmentations. `NEIGHBOR_COUNT`, `MODEL_SEEDS`, `GM_AUGMENTATIONS`, and
`GM_BATCH_SIZE` are configurable, but ImageA/B accuracy should not be used to
choose the reported configuration. Keep ImageC untouched for confirmation.

## Diagnostic class oracle

The class-oracle experiment asks whether independently best class pools remain
best after they are combined into one dataset. For each class, it uses the
existing downstream per-class means to select all ten real representatives or
all ten class-prompt Diffusion images. This deliberately leaks validation
performance and therefore is an optimistic diagnostic, not a selection method.

Run the build and two downstream classifier evaluations with:

```bash
export CLASS_ORACLE_RUN_ID=class_oracle_v0
export FINAL_CONTROL_RUN_ID=final_prompt_controls_v0
bash scripts/class_oracle_experiment.sh
```

The expected independent oracle accuracy is the class average
`mean_c(max(Acc_real_c, Acc_diffusion_c))`. After training the combined oracle
dataset, the summary reports:

```text
interaction_gap = actual_oracle_accuracy - expected_independent_oracle_accuracy
```

A negative gap means that independently favorable class choices do not compose
without loss. The output also reports oracle gain over the better all-real or
all-Diffusion endpoint. Datasets and manifests are stored in
`results/class_oracle_runs/<RUN_ID>/`; trained classifiers, per-class
interaction shifts, and `class_oracle_interaction_gap.png` are stored in
`trained_results/class_oracle_runs/<RUN_ID>/`.

If dataset construction finishes but training is interrupted, resume without
rebuilding it:

```bash
export CLASS_ORACLE_RUN_ID=class_oracle_v0
export FINAL_CONTROL_RUN_ID=final_prompt_controls_v0
bash scripts/train_class_oracle_run.sh
```

### Cross-fitted paired validation

The first oracle estimate selects the larger of two noisy endpoint means, so its
expected accuracy can be optimistically biased. Validate the fixed class choices
with disjoint classifier seeds while training all-real, all-Diffusion, and the
same hybrid dataset under identical seeds:

```bash
export CLASS_ORACLE_RUN_ID=class_oracle_v0
export PAIRED_ORACLE_RUN_ID=class_oracle_v0_paired_v0
export FINAL_CONTROL_RUN_ID=final_prompt_controls_v0
export EVAL_SEED_STARTS="2 4"
bash scripts/paired_class_oracle_validation.sh
```

With two visible GPUs, each seed start launches a pair of classifier seeds, so
the default evaluates seeds `2,3,4,5`. These must not overlap the selection
seeds recorded by the oracle endpoint files. No images are generated or copied;
the script only retrains classifiers against the existing real, Diffusion, and
hybrid directories.

For an interrupted run, reuse completed method/seed pairs with:

```bash
export CLASS_ORACLE_RUN_ID=class_oracle_v0
export PAIRED_ORACLE_RUN_ID=class_oracle_v0_paired_v0
export EVAL_SEED_STARTS="2 4"
export RESUME_RUN=true
bash scripts/paired_class_oracle_validation.sh
```

Results are isolated under
`trained_results/paired_class_oracle_runs/<PAIRED_ORACLE_RUN_ID>/`. The summary
contains paired seed and class CSV files, JSON statistics, and plots. Its primary
quantity is the same-seed difference between hybrid accuracy and the class-wise
selected endpoint accuracy, which removes the original max-of-noisy-means
comparison from evaluation.

## Final Montage and conflict-control confirmation

This experiment returns to the two explicit-text settings that showed weak but
potentially useful signals. It evaluates four nested conditions on ImageA,
ImageB, and the held-out ImageC subset, using generation seeds 0 and 1:

- original CoDA with its class-name prompt;
- the four-neighbor Montage common-mode caption;
- Montage plus Soft Projection with `alpha=0.5`;
- Montage plus Kappa Cap with `tau=0.3`.

Soft Projection and Kappa Cap are separate arms because both modify the same
conflicting image-guidance component. ImageA and ImageB are development subsets;
ImageC is a one-time confirmation subset and should not be used for another
round of tuning.

The default run automatically reuses completed ImageA baseline/Montage results
from `imageA_multiview_v0` and ImageB seed-1 baseline/Montage results from
`final_prompt_controls_v0`. Missing conditions, including all ImageC conditions,
are generated and trained normally:

```bash
export FINAL_MONTAGE_RUN_ID=final_montage_conflict_v0
bash scripts/final_montage_conflict_experiment.sh
```

With the default references this reuses 6 of the 24 dataset conditions and runs
the remaining 18. If interrupted, use the same ID and enable resume mode:

```bash
export FINAL_MONTAGE_RUN_ID=final_montage_conflict_v0
export RESUME_RUN=true
bash scripts/final_montage_conflict_experiment.sh
```

If interruption leaves partial downstream-training directories, resume mode
moves the base and per-GPU directories into
`incomplete_classifier_archives/` and restarts only that classifier condition
from epoch 0. Completed generated datasets and classifier conditions are still
reused. Set `ARCHIVE_INCOMPLETE_CLASSIFIERS=false` to retain the stricter
stop-on-partial behavior. Direct checkpoint continuation is intentionally not
used because the current training code does not restore the learning-rate
scheduler state and therefore would not be equivalent to an uninterrupted run.

The script validates each reused dataset before linking it into the new run,
archives partial classifier outputs, prepares missing ImageC feature/cluster
artifacts, and records all new outputs under independent run directories. Set
`IMAGEA_REFERENCE_RUN_ID`, `IMAGEB_REFERENCE_RUN_ID`, or
`IMAGEB_REFERENCE_SEED` when the completed reference IDs differ. Arbitrary
existing conditions can be supplied with variables such as
`IMAGEB_SEED0_CODA_BASELINE_DATA_DIR` and the corresponding `_RESULT_DIR`.

The final report is written to
`trained_results/final_montage_conflict_runs/<RUN_ID>/summary/`. It reports
classifier-seed-paired differences for Montage versus baseline, Soft Projection
versus Montage, and Kappa Cap versus Montage, both per generation seed and per
subset. ImageC is labeled as held out in the JSON summary and plots.

## VLCP DCS transfer to CoDA

The DCS transfer keeps CoDA clustering, SDXL image guidance, and classifier
evaluation unchanged. It changes only the text condition:

1. LLaVA captions every real training image in ImageA/B/C once.
2. Each image is assigned to its nearest saved CoDA representative in the
   original SDXL VAE feature space.
3. Following VLCP, words present in at least `0.7` of all class captions are
   removed, the remaining words are counted per cluster, and the existing
   cluster caption with maximum weighted coverage of the top 30 words is used.
4. SDXL receives
   `An natural photo of a {class_name}, {caption}, centered object.`

The nearest-center assignment is necessary because historical CoDA artifacts
store final representatives but not the post-processed HDBSCAN memberships.
The DCS manifest records this choice and all selected source paths.

Install the one additional text dependency and its corpus:

```bash
pip install nltk==3.9.1
python -m nltk.downloader stopwords
```

Run ImageA/B/C with generation seeds 0 and 1:

```bash
export DCS_TRANSFER_RUN_ID=dcs_transfer_v0
bash scripts/dcs_transfer_experiment.sh
```

By default only the new `dcs` arm is generated and trained, because matching
CoDA baselines already exist from earlier experiments. To regenerate paired
baselines inside the same run, set:

```bash
export METHODS="coda_baseline dcs"
```

Caption shards are appended under
`results/dcs_caption_cache/<spec>/vlcp_dcs_class_aware/`, so interruption during
the expensive LLaVA pass is resumable. Generation and classifier artifacts are
isolated under `dcs_transfer_runs/<RUN_ID>/`. Resume the same run with:

```bash
export DCS_TRANSFER_RUN_ID=dcs_transfer_v0
export RESUME_RUN=true
bash scripts/dcs_transfer_experiment.sh
```

The caption stage defaults to one GPU to avoid the host-RAM spike caused by
loading two LLaVA-7B replicas simultaneously. SDXL generation and classifier
evaluation still use both visible GPUs. Caption caches are independent of world
size, so a partially completed two-GPU caption run can resume on one GPU without
discarding either rank shard. Set `DCS_CAPTION_GPU_COUNT=2` and
`DCS_CAPTION_VISIBLE_DEVICES=0,1` only when the host has enough RAM.

Set `RUN_DCS_CAPTIONING=false` after caption caches are complete to rebuild only
the DCS selection manifest. The defaults match VLCP ImageNet parameters:
`DCS_THRESHOLD=0.7`, `DCS_TOP_K=30`, and no extra caption word truncation.

For a compute-limited pilot, caption only the 50 nearest assigned images per
representative:

```bash
export SPECS=imageA
export DCS_MAX_IMAGES_PER_CLUSTER=50
export DCS_TRANSFER_RUN_ID=dcs_transfer_imageA_m50_v0
bash scripts/dcs_transfer_experiment.sh
```

This caps ImageA at 500 captions instead of roughly 13,000. It is a
cluster-balanced approximation to VLCP DCS rather than the full-data method.
The manifest records original and sampled cluster sizes. `0` restores the full
method. `DCS_CAPTION_BATCH_SIZE=2` can improve throughput on GPUs with spare
memory, but the default remains `1` for constrained V100 servers.
