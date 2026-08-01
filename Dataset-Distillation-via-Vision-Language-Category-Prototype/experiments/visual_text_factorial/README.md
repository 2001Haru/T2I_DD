# Visual x Text Factorial on ImageNette

This experiment asks whether cluster-specific text contributes information
independently of a VLCP visual prototype, or only through an interaction with
prototype initialization. It uses the frozen SD 1.5 checkpoint and reuses the
fixed IPC-10 prototype and DCS JSON files from the prior-alignment pilot. It
does not run LLaVA, rebuild captions, or perform diffusion fine-tuning.

## Conditions

The six cells are:

| Visual condition | Text condition | Output name |
| --- | --- | --- |
| no visual information | class label | `no_visual_label` |
| no visual information | correct cluster DCS | `no_visual_dcs` |
| no visual information | same-class shuffled DCS | `no_visual_dcs_shuffled` |
| cluster prototype | class label | `prototype_label` |
| cluster prototype | correct cluster DCS | `prototype_dcs` |
| cluster prototype | same-class shuffled DCS | `prototype_dcs_shuffled` |

The shuffled control is a fixed cyclic derangement within each class:
prototype `i` receives DCS caption `(i + 1) mod 10`. It preserves class,
caption style, and caption length while breaking cluster correspondence.

## Matched visual intervention

The no-visual cell is not a separate full-schedule text-to-image pipeline.
Both visual cells use the same `StableDiffusionImg2ImgPipeline`, strength,
timestep schedule, and image-level random seed:

```text
no_visual: q(z_t | z_0 = 0)
prototype: q(z_t | z_0 = cluster_prototype)
```

Resetting the generator to the same image seed gives the two cells the same
sampled epsilon. This isolates the information in the prototype without
confounding it with a different number of denoising steps.

## Primary estimands

Within each visual condition:

```text
cluster correspondence effect = accuracy(correct DCS) - accuracy(shuffled DCS)
descriptive prompt effect = accuracy(shuffled DCS) - accuracy(label)
```

The primary interaction is:

```text
(correct DCS - shuffled DCS)_prototype
-
(correct DCS - shuffled DCS)_no_visual
```

`correct DCS > shuffled DCS` is evidence that cluster-caption correspondence,
not merely descriptive language, contributes useful information.

## Run

The default source directory reuses artifacts from
`../vlcp_ablation_runs/author_checkpoint_pilot_v0`. Override
`SOURCE_RUN_ROOT`, `PROTOTYPE_PATH`, or `DCS_PATH` when needed.

```bash
cd /linxi/T2I_DD/Dataset-Distillation-via-Vision-Language-Category-Prototype

DATA_ROOT=/linxi/dataset/VLCP/ImageNette \
BASE_MODEL=/linxi/models/VLCP/stable-diffusion-v1-5 \
SOURCE_RUN_ROOT=/linxi/T2I_DD/vlcp_ablation_runs/author_checkpoint_pilot_v0 \
RUN_ID=visual_text_factorial_v0 \
GENERATION_SEEDS="0 1" \
bash experiments/visual_text_factorial/run_experiment.sh
```

Resume the exact configuration with:

```bash
source ../vlcp_factorial_runs/visual_text_factorial_v0/resume.env
bash experiments/visual_text_factorial/run_experiment.sh
```

Generation, evaluation, and summary can be controlled independently:

```bash
GENERATE=false EVALUATE=true SUMMARIZE=true \
bash experiments/visual_text_factorial/run_experiment.sh
```

Outputs are isolated under:

```text
/linxi/T2I_DD/vlcp_factorial_runs/<RUN_ID>/
  run_config.txt
  synthetic/seed_<N>/<condition>/
  evaluation/seed_<N>/<condition>.log
  summary/conditions.csv
  summary/contrasts.csv
  summary/summary.json
```

Every synthetic condition stores a manifest containing model and artifact
identities, exact prompts, prompt-source cluster indices, initialization
definition, and image seeds. Resume refuses configuration mismatches.

Audit a completed run before interpreting an unexpected result:

```bash
python experiments/visual_text_factorial/audit_run.py \
  --run-root /linxi/T2I_DD/vlcp_factorial_runs/visual_text_factorial_v0
```

The audit verifies all six manifests, paired image seeds, correct and shuffled
prompt-source indices, DCS text hashes, image counts, completed classifier
logs, and the synthetic directory reported by each evaluation log. It writes
`audit.json` in the run root.

## Prespecified shuffle randomization

One shuffled pairing is enough to reject the claim that correct pairing is
always best, but it cannot distinguish a general pairing effect from a lucky
prompt-noise assignment. The randomization extension keeps the original
shift-1 result and adds shifts 2, 4, and 7. Only the two shuffled conditions
are generated and evaluated; label and correct-DCS results are reused from the
base factorial run.

```bash
cd /linxi/T2I_DD/Dataset-Distillation-via-Vision-Language-Category-Prototype

CUDA_VISIBLE_DEVICES=0 \
DATA_ROOT=/linxi/dataset/VLCP/ImageNette \
BASE_RUN_ROOT=/linxi/T2I_DD/vlcp_factorial_runs/visual_text_factorial_v0 \
SOURCE_RUN_ROOT=/linxi/T2I_DD/vlcp_ablation_runs/author_checkpoint_pilot_v0 \
RANDOMIZATION_RUN_ID=visual_text_shuffle_randomization_v0 \
SHUFFLE_SHIFTS="2 4 7" \
GENERATION_SEEDS="0 1" \
bash experiments/visual_text_factorial/run_shuffle_randomization.sh
```

Resume with the same command plus `RESUME=true`. The primary result is:

```text
correct DCS - mean(all prespecified shuffled DCS pairings)
```

The summary deliberately averages every declared shuffle instead of selecting
the best one. Outputs are written to:

```text
/linxi/T2I_DD/vlcp_shuffle_runs/<RANDOMIZATION_RUN_ID>/summary/
  shuffle_comparisons.csv
  shuffle_summary.json
```

## Semantic coverage diagnostics

This follow-up tests whether within-class DCS shuffling expands the real data
manifold covered by the synthetic set. It reuses every generated image and
does not run diffusion or classifier training.

Two independent representations are used:

1. The frozen SD 1.5 CLIP text encoder measures how much the actual
   generation-time conditioning changes. Encoding mirrors the generator's
   untruncated 77-token chunking.
2. A separately downloaded DINOv2 model measures class-conditional image
   coverage and fidelity against the real ImageNette training split.

For each class and synthetic condition, the image diagnostic reports:

```text
coverage distance = mean real-image distance to its nearest synthetic image
fidelity distance = mean synthetic-image distance to its nearest real image
coverage@R = fraction of real images within the class-specific real-NN radius
precision@R = fraction of synthetic images within that radius
diversity = mean pairwise distance among synthetic images
```

Lower continuous coverage/fidelity distances are better. The radius `R` is
the 95th percentile of each real image's nearest-other-real DINOv2 distance.

Run with the existing base and shift `{1,2,4,7}` artifacts:

```bash
cd /linxi/T2I_DD/Dataset-Distillation-via-Vision-Language-Category-Prototype

CUDA_VISIBLE_DEVICES=0 \
DATA_ROOT=/linxi/dataset/VLCP/ImageNette \
BASE_RUN_ROOT=/linxi/T2I_DD/vlcp_factorial_runs/visual_text_factorial_v0 \
RANDOMIZATION_ROOT=/linxi/T2I_DD/vlcp_shuffle_runs/visual_text_shuffle_randomization_v0 \
BASE_MODEL=/linxi/models/VLCP/stable-diffusion-v1-5 \
DINO_MODEL=/linxi/models/DINOv2/dinov2-base \
DIAGNOSTICS_ID=semantic_coverage_v0 \
bash experiments/visual_text_factorial/run_semantic_coverage_diagnostics.sh
```

Resume interrupted feature extraction with the same command plus
`RESUME=true`. Cached features are reused only when the model and complete
image inventory match. Main outputs are:

```text
diagnostics/<DIAGNOSTICS_ID>/
  text/conditioning_pairs.csv
  text/conditioning_shift_summary.csv
  text/conditioning_class_summary.csv
  dino/dino_metrics_per_class.csv
  dino/dino_metrics_summary.csv
  dino/dino_shuffled_minus_correct.csv
  dino/dino_shuffled_minus_correct_per_class.csv
  summary/conditioning_and_coverage.csv
  summary/conditioning_and_coverage_per_class.csv
  summary/semantic_coverage_diagnostic.png
  summary/semantic_coverage_summary.json
```

## Per-class downstream utility

The original Minimax evaluations saved aggregate curves but no checkpoints or
per-class predictions. Per-class utility therefore cannot be reconstructed
from the completed logs. The downstream diagnostic reruns only the required
classifier evaluations and records each class accuracy at the epoch with the
best aggregate validation accuracy for every classifier repeat.

The default run is deliberately restricted to the primary prototype
comparison:

```text
2 generation seeds x (1 correct DCS + 4 shuffled DCS) = 10 conditions
```

It reuses all synthetic images and trains no diffusion model:

```bash
cd /linxi/T2I_DD/Dataset-Distillation-via-Vision-Language-Category-Prototype

CUDA_VISIBLE_DEVICES=0 \
DATA_ROOT=/linxi/dataset/VLCP/ImageNette \
BASE_RUN_ROOT=/linxi/T2I_DD/vlcp_factorial_runs/visual_text_factorial_v0 \
RANDOMIZATION_ROOT=/linxi/T2I_DD/vlcp_shuffle_runs/visual_text_shuffle_randomization_v0 \
DIAGNOSTICS_ID=semantic_coverage_v0 \
bash experiments/visual_text_factorial/run_downstream_class_diagnostics.sh
```

Resume with the same command plus `RESUME=true`. To additionally run the
no-visual interaction control, use:

```bash
VISUAL_MODES="prototype no_visual"
```

The analysis defines:

```text
per-class downstream gain = shuffled DCS accuracy - correct DCS accuracy
```

Classifier repeats are paired by repeat index because every condition starts
from the same classifier seed. Correlations are reported both per generation
seed and after averaging generation seeds. Main outputs are:

```text
diagnostics/<DIAGNOSTICS_ID>/downstream_per_class/
  results/seed_<N>/<visual>_correct.json
  results/seed_<N>/<visual>_shift<SHIFT>.json
  summary/downstream_dino_per_class.csv
  summary/downstream_dino_averaged_generation_seeds.csv
  summary/downstream_dino_correlations.csv
  summary/downstream_class_mean_gains.csv
  summary/downstream_shift_summary.csv
  summary/downstream_dino_correlations.png
  summary/downstream_dino_summary.json
```

## Cross-modal recombination geometry

The cyclic shift number is only a permutation identifier: KMeans cluster
indices are not ordered by visual distance. This diagnostic therefore computes
the actual DINOv2 distance between every source prototype and the prototype
whose DCS caption was assigned to it.

For a source prototype feature `v_i`, caption-source prototype feature `v_j`,
shuffled output `y_ij`, and paired correct-DCS output `y_ii`, it reports:

```text
tau_from_source =
  dot(y_ij - v_i, v_j - v_i) / ||v_j - v_i||^2

tau_from_correct =
  dot(y_ij - y_ii, v_j - v_i) / ||v_j - v_i||^2
```

`tau_from_correct` isolates the feature displacement caused by caption
reassignment because the correct and shuffled images use the same visual
prototype and generation noise. A positive projection means that shuffling
moved the output toward the visual mode from which the caption came. The
diagnostic also records direction cosine, orthogonal residual, target
similarity improvement, and the fraction with `0 <= tau_from_source <= 1`.

It decodes the existing VAE cluster centers once, reuses cached synthetic-image
DINO features when available, and performs no diffusion generation or
classifier training:

```bash
cd /linxi/T2I_DD/Dataset-Distillation-via-Vision-Language-Category-Prototype

CUDA_VISIBLE_DEVICES=0 \
BASE_RUN_ROOT=/linxi/T2I_DD/vlcp_factorial_runs/visual_text_factorial_v0 \
RANDOMIZATION_ROOT=/linxi/T2I_DD/vlcp_shuffle_runs/visual_text_shuffle_randomization_v0 \
BASE_MODEL=/linxi/models/VLCP/stable-diffusion-v1-5 \
DINO_MODEL=/linxi/models/DINOv2/dinov2-base \
DIAGNOSTICS_ID=semantic_coverage_v0 \
bash experiments/visual_text_factorial/run_cross_modal_recombination_diagnostics.sh
```

Resume with the same command plus `RESUME=true`. If the per-class downstream
CSV from the previous section exists, it is joined automatically. Main outputs
are:

```text
diagnostics/<DIAGNOSTICS_ID>/cross_modal_recombination/
  decoded_prototypes/
  recombination_per_image.csv
  recombination_per_class.csv
  recombination_per_generation_shift.csv
  recombination_per_shift.csv
  recombination_vs_downstream_per_class.csv
  recombination_summary.json
  recombination_by_permutation.png
  recombination_pair_geometry.png
  recombination_vs_downstream.png
```

The quadratic distance and `tau` fits are exploratory. They test the proposed
mixup-like mechanism but do not by themselves establish that DINO geometry
causes downstream accuracy changes.

## Real cluster-member audit

Directly decoded VAE KMeans centers can be dominated by low-frequency color
and background. This audit assigns every real ImageNette training image to its
nearest stored center in the original scaled VAE latent space and saves the
nearest real members for manual inspection.

The distance is reported as per-latent-element RMSE rather than raw squared
Euclidean distance. For every cluster the audit records:

```text
nearest, p10, p25, median, p75, p90, and maximum center RMSE
nearest / median member RMSE
nearest / p10 member RMSE
nearest member / nearest-other-center RMSE
assignment margin between the nearest and second-nearest centers
```

This distinction matters: the nearest available image need not be close to the
center. A `nearest / median` ratio near one means that even the nearest member
is not substantially more representative than a typical assigned member.

The default settings replay the original prototype extraction protocol:
`seed=0`, 512-pixel resize, batch size 10, shuffled loading, and a sampled VAE
posterior. The original fit removed 10% LOF outliers, while this audit assigns
all training images to the stored centers; this difference is recorded in the
summary and should be considered when interpreting cluster sizes.

```bash
cd /linxi/T2I_DD/Dataset-Distillation-via-Vision-Language-Category-Prototype

CUDA_VISIBLE_DEVICES=0 \
DATA_ROOT=/linxi/dataset/VLCP/ImageNette \
BASE_RUN_ROOT=/linxi/T2I_DD/vlcp_factorial_runs/visual_text_factorial_v0 \
BASE_MODEL=/linxi/models/VLCP/stable-diffusion-v1-5 \
RANDOMIZATION_ROOT=/linxi/T2I_DD/vlcp_shuffle_runs/visual_text_shuffle_randomization_v0 \
DIAGNOSTICS_ID=semantic_coverage_v0 \
bash experiments/visual_text_factorial/run_cluster_member_audit.sh
```

Resume an interrupted run with the same command plus `RESUME=true`. The VAE
pass is replayed from the start if it was interrupted before the assignments
CSV was completed; completed assignments are reused. Main outputs are:

```text
diagnostics/<DIAGNOSTICS_ID>/cluster_member_audit/
  latent_assignments.csv
  cluster_distance_summary.csv
  class_distance_summary.csv
  nearest_cluster_members.json
  cluster_center_gap.png
  montages/<synset>/cluster_<N>.png
  cluster_member_audit_summary.json
```

When decoded prototype images from the previous diagnostic are present, each
montage places the decoded center first and the nearest real images after it.

## Real-member recombination geometry

The decoded-center diagnostic is unreliable when VAE centroids decode into
blurred low-frequency images. This follow-up replaces each decoded centroid
with a semantic anchor built from the real images nearest to that centroid in
the original scaled VAE latent space:

```text
anchor_i(K) = normalize(mean(normalize(DINO(x))))
              for the K nearest real members x of cluster i
```

The default sensitivity analysis uses `K=3,5,9`. These anchor images and the
held-out validation images do not overlap: ranks 10 through 30 are classified
against the ten within-class anchors in DINO space. The diagnostic reports
10-way retrieval accuracy, own-anchor margin, and retrieval rank. A semantic
cluster-direction interpretation should be rejected when held-out retrieval is
near the 10% chance level or changes qualitatively across K.

For visual cluster `i`, caption-source cluster `j`, paired correct image
`y_ii`, and shuffled image `y_ij`, the primary movement metric is:

```text
unit_caption_pull_projection =
  dot(DINO(y_ij) - DINO(y_ii), unit(anchor_j - anchor_i))
```

The output also records caption-source similarity gain, visual-target
similarity change, off-axis displacement, cluster occupancy, and exploratory
cluster-size correlations. It reuses all existing synthetic images and does no
diffusion generation or classifier training.

Run it after the real cluster-member audit:

```bash
cd /linxi/T2I_DD/Dataset-Distillation-via-Vision-Language-Category-Prototype

CUDA_VISIBLE_DEVICES=0 \
BASE_RUN_ROOT=/linxi/T2I_DD/vlcp_factorial_runs/visual_text_factorial_v0 \
RANDOMIZATION_ROOT=/linxi/T2I_DD/vlcp_shuffle_runs/visual_text_shuffle_randomization_v0 \
DINO_MODEL=/linxi/models/DINOv2/dinov2-base \
DIAGNOSTICS_ID=semantic_coverage_v0 \
bash experiments/visual_text_factorial/run_real_member_recombination_diagnostics.sh
```

Resume with the same command plus `RESUME=true`. Main outputs are:

```text
diagnostics/<DIAGNOSTICS_ID>/real_member_recombination/
  anchor_validation_per_member.csv
  anchor_validation_per_class.csv
  anchor_validation_summary.csv
  real_anchor_recombination_per_image.csv
  real_anchor_recombination_per_class.csv
  real_anchor_recombination_per_shift.csv
  real_anchor_recombination_vs_downstream.csv
  real_anchor_class_hypothesis_summary.csv
  real_anchor_recombination_summary.json
  real_anchor_validation.png
  real_anchor_pair_geometry.png
  real_anchor_size_response.png
```

Interpret the result in order. First require valid held-out anchors. Then check
whether caption pull is positive and stable across K. Only after both checks
should cluster-size relationships be interpreted as possible cross-modal
recombination rather than generic prompt perturbation. The class-hypothesis
CSV averages generation seeds and shifts before correlating occupancy with
downstream gain, so the correlation sample size remains ten independent
classes rather than treating repeated runs as independent classes.

## Selective small-cluster shuffle

This control tests whether the class-level relationship between minimum
cluster occupancy and shuffle gain localizes to the smallest visual clusters.
It uses four settings:

```text
correct           all ten images come from the paired correct-DCS dataset
all_shuffled      all ten images come from the existing shuffled-DCS dataset
small3_shuffled   only the three least-populated visual clusters use shuffled images
random3_shuffled  three fixed non-small clusters use shuffled images
```

`random3_shuffled` samples from the seven clusters outside `small3`, so the two
selective target sets never overlap. The target selection is fixed across
generation seeds and shifts. Every hybrid image is linked from an existing
correct or shuffled image with the same visual prototype and image seed; no
diffusion generation is performed.

The runner trains classifiers only for `small3_shuffled` and
`random3_shuffled`. Existing correct and all-shuffled evaluation logs are
reused in the four-way summary:

```bash
cd /linxi/T2I_DD/Dataset-Distillation-via-Vision-Language-Category-Prototype

CUDA_VISIBLE_DEVICES=0 \
DATA_ROOT=/linxi/dataset/VLCP/ImageNette \
BASE_RUN_ROOT=/linxi/T2I_DD/vlcp_factorial_runs/visual_text_factorial_v0 \
RANDOMIZATION_ROOT=/linxi/T2I_DD/vlcp_shuffle_runs/visual_text_shuffle_randomization_v0 \
DIAGNOSTICS_ID=semantic_coverage_v0 \
bash experiments/visual_text_factorial/run_selective_shuffle_experiment.sh
```

Resume with the same command plus `RESUME=true`. With the default two
generation seeds and four shifts, the runner creates 16 new classifier jobs:
two hybrid conditions per paired seed/shift cell. Hybrid images are symbolic
links and require negligible additional dataset storage. Main outputs are:

```text
vlcp_selective_shuffle_runs/selective_small_cluster_shuffle_v0/
  target_selections.json
  shift_<S>/synthetic/seed_<G>/small3_shuffled/
  shift_<S>/synthetic/seed_<G>/random3_shuffled/
  shift_<S>/evaluation/seed_<G>/*.log
  shift_<S>/evaluation/seed_<G>/*.per_class.json
  summary/selective_conditions.csv
  summary/selective_contrasts.csv
  summary/selective_aggregate.csv
  summary/selective_per_class.csv
  summary/selective_per_class_aggregate.csv
  summary/selective_shuffle_contrasts.png
  summary/selective_summary.json
```

The primary localization contrast is `small3_shuffled - random3_shuffled`.
`small3_shuffled - correct` tests whether the targeted intervention is itself
beneficial, while `all_shuffled - small3_shuffled` tests whether improvements
require changing clusters beyond the smallest three.

### Additional generation seeds

Use the focused extension runner to test new generation seeds without training
the all-shuffled controls. It generates the paired source images required to
assemble the hybrids, then evaluates only `correct`, `small3_shuffled`, and
`random3_shuffled`:

```bash
CUDA_VISIBLE_DEVICES=0 \
DATA_ROOT=/linxi/dataset/VLCP/ImageNette \
GENERATION_SEEDS="2 3" \
bash experiments/visual_text_factorial/run_selective_shuffle_seed_extension.sh
```

Resume the same run with `RESUME=true`. The default output is
`../vlcp_selective_shuffle_runs/selective_small_cluster_shuffle_seed23_v0`.

### Multiple random-mask controls

After generation seeds 0-3 are complete, compare `small3_shuffled` against
three additional fixed random selections from the seven non-small clusters:

```bash
CUDA_VISIBLE_DEVICES=0 \
DATA_ROOT=/linxi/dataset/VLCP/ImageNette \
RANDOM_TARGET_SEEDS="20260801 20260802 20260803" \
bash experiments/visual_text_factorial/run_random_mask_controls.sh
```

This runner reuses all existing correct and shuffled images. It does not run
diffusion or retrain `small3_shuffled`. The focused default fixes the canonical
cyclic shift to 1 and contains 12 new classifier jobs (three masks and four
generation seeds). This prioritizes replication over random target masks rather
than treating arbitrary cyclic shifts as independent evidence. The summary
treats generation seed as the primary repeat and includes the original
`20260731` random mask as a fourth control. Outputs are written below
`../vlcp_selective_shuffle_runs/selective_random_mask_controls_shift1_v0/summary`.
