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
DINO_MODEL=/models/DINOv2/dinov2-base \
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
