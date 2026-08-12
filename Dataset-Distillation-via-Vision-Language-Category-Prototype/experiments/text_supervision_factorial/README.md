# Text as Condition vs. Text as Supervision

This experiment isolates two roles of text in the VLCP ImageNette pipeline.

| Training supervision | Label inference | Correct DCS inference | Shuffled DCS inference |
|---|---:|---:|---:|
| Frozen SD1.5 | yes | yes | yes |
| Label fine-tuning | yes | yes | yes |
| Dynamic unpaired-caption fine-tuning | yes | yes | yes |
| Matched-caption fine-tuning | yes | yes | yes |

All inference cells use the same VLCP prototype initialization (`strength=0.7`), guidance scale 10, 50 inference steps, generation seeds, and image noise. The only inference-side change is the prompt.

## Supervision controls

- `label_ft`: every training image receives its ImageNet class label.
- `matched_ft`: every image receives its original LLaVA caption from `nette.jsonl`.
- `unpaired_ft`: captions are remapped without replacement inside each class at every epoch. The class-level caption multiset is exactly preserved, self-pairs are forbidden, and the mapping is deterministic from the training seed and epoch.

Thus `matched_ft - unpaired_ft` estimates the value of image-caption correspondence while holding image data and class-level caption marginals fixed. This is deliberately different from a single static global shuffle.

## Causal ladder extension

The extension adds two controls and a second fine-tuning seed without rerunning completed seed-0 checkpoints:

- `empty_ft`: target-domain images with the empty CLIP text condition.
- `constant_ft`: the same images with the shared prompt `A natural photo.`.
- `label_ft`, `unpaired_ft`, and `matched_ft`: a second fine-tuning seed.

This separates target-domain image adaptation, generic conditioning style, class-language supervision, rich class-level text marginals, and instance-level correspondence. `BASE_RUN_ROOT` must point to the completed original 4x3 run so its frozen and seed-0 results can be reused.

For four A100 40GB GPUs, start the persistent scheduler with:

```bash
DATA_ROOT=/linxi/dataset/VLCP/ImageNette \
BASE_MODEL=/linxi/models/VLCP/stable-diffusion-v1-5 \
BASE_RUN_ROOT=/linxi/T2I_DD/Dataset-Distillation-via-Vision-Language-Category-Prototype/text_supervision_factorial_runs/text_supervision_factorial_2xa100_v0 \
RUN_ID=text_supervision_causal_ladder_v0 \
GPU_IDS=0,1,2,3 \
DIFFUSERS_SRC=/linxi/packages/VLCP/diffusers \
bash experiments/text_supervision_factorial/run_causal_ladder.sh
```

Each checkpoint is trained on one GPU with `batch_size=4`, `gradient_accumulation_steps=8`, and effective batch size 32. The scheduler prioritizes training, then generation, then classifier evaluation. It keeps one parent process alive, ignores SSH `SIGHUP`, dynamically fills free GPUs, resumes checkpoints, and retries failed children indefinitely by default. Use `MAX_RETRIES=N` only when deliberate fail-fast behavior is desired.

For an SSH-independent launch:

```bash
nohup env DATA_ROOT=/linxi/dataset/VLCP/ImageNette \
  BASE_MODEL=/linxi/models/VLCP/stable-diffusion-v1-5 \
  BASE_RUN_ROOT=/path/to/text_supervision_factorial_2xa100_v0 \
  RUN_ID=text_supervision_causal_ladder_v0 GPU_IDS=0,1,2,3 \
  DIFFUSERS_SRC=/linxi/packages/VLCP/diffusers \
  bash experiments/text_supervision_factorial/run_causal_ladder.sh \
  > causal_ladder_v0.log 2>&1 < /dev/null &
```

Progress is written to `scheduler_logs/`. The final causal contrasts are in `summary/causal_ladder_summary.json` and CSV files.

The causal-ladder summary also writes:

- `performance_by_supervision_and_prompt.csv`: six training regimes under Label, Correct, Shuffled, and paired descriptive-average inference.
- `prompt_effects_by_supervision.csv`: `Descriptive-Label` and `Correct-Shuffled` effects with hierarchical bootstrap intervals.
- `primary_mechanism_summary.json`: compact machine-readable versions of those two tables.
- `causal_ladder_mechanisms.png`: the performance ladder, conditioning-style effect, and correspondence effect.
- `endpoint_policy_performance.csv` and `endpoint_policy_contrast.csv`: the paired end-to-end comparison between caption-free `Empty-FT + Label` and caption-intensive `Matched-FT + Correct DCS`.
- `endpoint_policy_comparison.png`: absolute endpoint performance and the hierarchical-bootstrap accuracy premium of the caption-intensive policy. This is a joint deployment-policy contrast, not an isolated caption causal effect.

## Four-A100 run

```bash
cd /linxi/T2I_DD/Dataset-Distillation-via-Vision-Language-Category-Prototype

DATA_ROOT=/linxi/dataset/VLCP/ImageNette \
BASE_MODEL=/linxi/models/VLCP/stable-diffusion-v1-5 \
DIFFUSERS_SRC=/linxi/packages/VLCP/diffusers/src \
RUN_ID=text_supervision_factorial_v0 \
TRAIN_GPU_IDS=0,1,2,3 \
NUM_PROCESSES=4 \
TRAIN_BATCH_SIZE=4 \
GRADIENT_ACCUMULATION_STEPS=2 \
MAX_PARALLEL_EVALS=2 \
bash experiments/text_supervision_factorial/run_experiment.sh
```

The effective training batch remains 32. Each of the three fine-tunes uses all four GPUs sequentially. Generation assigns one supervision row to each GPU. Classifier evaluation defaults to two concurrent jobs because increasing this to four can exhaust host memory even when GPU memory is ample.

To resume the exact same run:

```bash
DATA_ROOT=/linxi/dataset/VLCP/ImageNette \
BASE_MODEL=/linxi/models/VLCP/stable-diffusion-v1-5 \
DIFFUSERS_SRC=/linxi/packages/VLCP/diffusers/src \
RUN_ID=text_supervision_factorial_v0 \
RESUME=true \
bash experiments/text_supervision_factorial/run_experiment.sh
```

For two A100 40GB GPUs, retain effective batch 32 with batch 4 and accumulation 4:

```bash
DATA_ROOT=/linxi/dataset/VLCP/ImageNette \
BASE_MODEL=/linxi/models/VLCP/stable-diffusion-v1-5 \
DIFFUSERS_SRC=/linxi/packages/VLCP/diffusers/src \
RUN_ID=text_supervision_factorial_2xa100_v0 \
TRAIN_GPU_IDS=0,1 \
WORKER_GPU_IDS=0,1 \
NUM_PROCESSES=2 \
TRAIN_BATCH_SIZE=4 \
GRADIENT_ACCUMULATION_STEPS=4 \
MAX_PARALLEL_EVALS=2 \
bash experiments/text_supervision_factorial/run_experiment.sh
```

For one A100, use batch 4 and accumulation 8. To distribute independent checkpoints over separate nodes, set `TRAIN_ROWS` and `TRAIN_ONLY=true`; training-only jobs do not build prototype artifacts. Copy or link the resulting `models/<row>` directories into one aggregation run before generation.

Set `TRAIN=false`, `GENERATE=false`, or `EVALUATE=false` to restart only later stages. A strict `run_config.txt` prevents mixing incompatible paths or settings.

## Loss audit

Every fine-tune writes:

- `models/<mode>/timestep_loss_intervals.jsonl`: loss in ten fixed diffusion-timestep bins every 50 optimizer steps.
- `models/<mode>/timestep_loss_epochs.csv`: per-epoch loss sums/counts for the same bins.
- `models/<mode>/caption_assignment_audit.jsonl`: assignment hash, self-pair count, and caption-multiset check for every epoch.
- `summary/timestep_loss.png`: epoch loss and final-epoch timestep profile across the three supervision modes.

The bins are defined on the training scheduler's native 1000 timesteps, so they are directly comparable across rows. Raw sums are reduced across all four processes before means are written.

## Main estimands

- `matched_ft - unpaired_ft`: value of matched captions as training supervision.
- `unpaired_ft - label_ft`: value of the class-level caption marginal without image-caption matching.
- `correct - shuffled`: value of cluster correspondence at inference.
- `(matched correct-shuffled) - (unpaired correct-shuffled)`: whether matched training makes the model specifically exploit matched cluster prompts at inference.

Results are written to `summary/summary.json`, `cells.csv`, and `contrasts.csv`.

## Generality sequence: seeds, IPC, then ImageWoof

`run_generality.sh` executes the preregistered sequence without mixing LoRA into the mechanism test:

1. Extend ImageNette IPC10 with fine-tuning seeds 2 and 3 for `Empty-FT + Label` and `Matched-FT + Correct/Shuffled`.
2. Reuse training seeds 0 and 1 at IPC20 and IPC50, comparing `Frozen + Label`, `Empty-FT + Label`, and `Matched-FT + Correct/Shuffled`.
3. Replicate on ImageWoof with Frozen, Empty-FT, Unpaired-FT, and Matched-FT. LoRA is deliberately deferred until these tests establish generality.

The default IPC sweep uses only training seeds 0 and 1 to prevent IPC50 from becoming a second full causal ladder. Set `IPC_TRAINING_SEEDS="0 1 2 3"` only if the first sweep justifies the extra generation and classifier cost.

Run the first two phases on two V100s:

```bash
cd /linxi/T2I_DD/Dataset-Distillation-via-Vision-Language-Category-Prototype

nohup env \
  NETTE_DATA_ROOT=/linxi/dataset/VLCP/ImageNette \
  NETTE_CAPTION_FILE=/linxi/dataset/VLCP/ImageNette/train/nette.jsonl \
  BASE_MODEL=/linxi/models/VLCP/stable-diffusion-v1-5 \
  BASE_RUN_ROOT=/path/to/text_supervision_factorial_2xa100_v0 \
  CAUSAL_RUN_ROOT=/path/to/text_supervision_causal_ladder_v0 \
  RUN_ID=text_supervision_generality_v0 \
  GPU_IDS=0,1 DIFFUSERS_SRC=/linxi/packages/VLCP/diffusers/src \
  PHASES="nette_seeds nette_ipc" \
  bash experiments/text_supervision_factorial/run_generality.sh \
  > text_supervision_generality_v0.log 2>&1 < /dev/null &
```

The parent process ignores SSH `SIGHUP`, runs one full-UNet job per GPU with effective batch size 32, permits only one classifier evaluation at a time to limit host-memory pressure, archives failed logs, resumes checkpoints/partial generation, and retries indefinitely by default. A strict `run_manifest.json` rejects incompatible resumes.

Prepare ImageWoof from the same ImageNet archive:

```bash
python experiments/text_supervision_factorial/prepare_imagenet_subset.py \
  --spec woof \
  --source-root /zhangchi/imagenet_512/images \
  --validation-root /linxi/dataset/imagenet/validation/val \
  --output-root /linxi/dataset/VLCP/ImageWoof \
  --link-mode symlink
```

Run LLaVA on `/linxi/dataset/VLCP/ImageWoof/llava_questions.jsonl`, then merge all answer shards:

```bash
python experiments/prior_alignment_ablation/merge_llava_answers.py \
  --questions /linxi/dataset/VLCP/ImageWoof/llava_questions.jsonl \
  --answers /path/to/answers/*.jsonl \
  --output /linxi/dataset/VLCP/ImageWoof/train/woof.jsonl
```

After captions are complete, resume the same run with only the held-out replication phase:

```bash
NETTE_DATA_ROOT=/linxi/dataset/VLCP/ImageNette \
NETTE_CAPTION_FILE=/linxi/dataset/VLCP/ImageNette/train/nette.jsonl \
WOOF_DATA_ROOT=/linxi/dataset/VLCP/ImageWoof \
WOOF_CAPTION_FILE=/linxi/dataset/VLCP/ImageWoof/train/woof.jsonl \
BASE_MODEL=/linxi/models/VLCP/stable-diffusion-v1-5 \
BASE_RUN_ROOT=/path/to/text_supervision_factorial_2xa100_v0 \
CAUSAL_RUN_ROOT=/path/to/text_supervision_causal_ladder_v0 \
RUN_ID=text_supervision_generality_woof_v0 GPU_IDS=0,1 \
DIFFUSERS_SRC=/linxi/packages/VLCP/diffusers/src PHASES=woof \
bash experiments/text_supervision_factorial/run_generality.sh
```

ImageWoof intentionally uses a separate `RUN_ID`: adding a phase to an existing run changes the preregistered manifest and is rejected. Final outputs are `summary/performance.csv`, `summary/contrasts.csv`, and `summary/generality_summary.png`.

To produce one cross-dataset report from the separately frozen manifests:

```bash
python experiments/text_supervision_factorial/summarize_generality.py \
  --evaluation-index \
    text_supervision_generality_runs/text_supervision_generality_v0/evaluation_index.json \
    text_supervision_generality_runs/text_supervision_generality_woof_v0/evaluation_index.json \
  --output-dir text_supervision_generality_runs/combined_summary
```
## Prototype-strength x prompt interaction

`run_strength_interaction.sh` tests a different estimand from a standard D4M strength sweep. It fixes the
`matched_ft` generator and CFG, then measures the paired marginal value of cluster text,
`A_correct(strength) - A_label(strength)`, across prototype initialization strengths and IPC values. The default
grid is strength `{0.7, 0.8, 0.9, 1.0}`, IPC `{10, 50}`, training seeds `{0, 1}`, and generation seeds `{0, 1}`.

The scheduler validates and reuses exact strength-0.7 cells from the original factorial, causal-ladder, and
generality runs. Missing cells, including matched-FT Label at IPC50, are generated and evaluated in the new run.
All new outputs are isolated by IPC, strength, training seed, and generation seed.

```bash
NETTE_DATA_ROOT=/linxi/dataset/VLCP/ImageNette \
BASE_MODEL=/linxi/models/VLCP/stable-diffusion-v1-5 \
BASE_RUN_ROOT=/path/to/original_text_supervision_run \
CAUSAL_RUN_ROOT=/path/to/text_supervision_causal_ladder_run \
GENERALITY_RUN_ROOT=/path/to/text_supervision_generality_run \
RUN_ID=strength_prompt_interaction_v0 \
GPU_IDS=0,1 \
bash experiments/text_supervision_factorial/run_strength_interaction.sh
```

The default `MAX_PARALLEL_EVALS=2` launches two independent single-GPU classifier evaluations so both GPUs remain
occupied during the evaluation phase. Set `MAX_PARALLEL_EVALS=1` if the host runs out of RAM or DataLoader workers
are killed; this scheduling option does not change any experimental cell or resume manifest.

The process ignores `SIGHUP`, retries failed tasks indefinitely by default, and writes scheduler state under the
run root. Re-running the same command resumes. Primary outputs are `summary/prompt_utility.csv`,
`summary/interactions_relative_to_0p7.csv`, `summary/ipc_interactions.csv`, and
`summary/strength_interaction_summary.png`. The descriptive
best-strength table is exploratory; it is not an unbiased post-selection estimate.

## Overnight A/B/C conditioning-interface matrix

`run_conditioning_interface_matrix.sh` runs the large preregistered follow-up without the abandoned `G`
condition. The only prompt conditions are Label, Correct DCS, and within-class Shuffled DCS.

| Matrix | Dataset/checkpoint | IPC | Visual interfaces | Prompt randomization |
|---|---|---|---|---|
| A | ImageNette Matched-FT, training seeds 0/1 | 10, 20, 50 | strength 0.70 to 1.00 in 0.05 increments | L/C/S1 everywhere; S2/S4/S7 at 0.7/0.8/0.9/1.0 |
| B | ImageNette Frozen and Empty-FT seeds 0/1 | 10, 50 | strength 0.7/0.8/0.9/1.0 plus true pure-noise T2I | L/C/S1 |
| C1 | ImageWoof Frozen + Empty/Constant/Label/Unpaired/Matched-FT | 10 | strength 0.7 and true pure-noise T2I | L/C/S1 |
| C2 | ImageWoof Frozen/Empty/Matched at IPC10; Matched-FT at IPC20/50 | 10, 20, 50 | strength 0.7/0.8/0.9/1.0 | L/C/S1 |
| D | ImageNette Matched-FT, training seeds 0/1 | 50 | strength 0.7/0.8/0.9/1.0 | L/C/S1; intended to reuse L/C |

With generation seeds 0/1 and classifier repeat 2, the default A/B/C manifest contains 396, 180, and 318
evaluation cells respectively. Matrix C is phase-selectable: `ladder` has 132 cells,
`curve_ipc10_20` adds 138, and `curve_ipc50` adds 48. Matrix D has 48 logical cells, but completed exact cells
are read from reuse indexes instead of regenerated. `strength=1.0` remains an img2img cell; `pure_noise` uses
`StableDiffusionPipeline` and is therefore a genuinely visual-free control.

The scheduler is one persistent parent process. It runs the required ImageWoof fine-tunes across available GPUs,
fills later phases with artifacts/generation, and by default runs up to four independent single-GPU classifier
evaluations. Set `MAX_PARALLEL_EVALS=2` or `1` on a host with limited RAM; this runtime control does not alter
the experiment manifest. Minimax keeps its historical evaluation worker configuration so resumed cells remain
comparable with earlier results. The scheduler ignores SSH `SIGHUP`, archives failed
logs, and retries indefinitely. Existing exact cells can be reused through one or more
prior `evaluation_index.json` files. `MAX_WALLTIME_HOURS` stops launching new work at the deadline, waits for active
tasks, writes `summary_partial`, and exits cleanly. It is a runtime limit and can change between resumes.

```bash
cd /linxi/T2I_DD/Dataset-Distillation-via-Vision-Language-Category-Prototype

nohup env \
  NETTE_DATA_ROOT=/linxi/dataset/VLCP/ImageNette \
  NETTE_CAPTION_FILE=/linxi/dataset/VLCP/ImageNette/train/nette.jsonl \
  WOOF_DATA_ROOT=/linxi/dataset/VLCP/ImageWoof \
  WOOF_CAPTION_FILE=/linxi/dataset/VLCP/ImageWoof/train/woof.jsonl \
  BASE_MODEL=/linxi/models/VLCP/stable-diffusion-v1-5 \
  BASE_RUN_ROOT=./text_supervision_factorial_runs/text_supervision_factorial_2xa100_v0 \
  CAUSAL_RUN_ROOT=./text_supervision_factorial_runs/text_supervision_causal_ladder_v0 \
  GENERALITY_RUN_ROOT=./text_supervision_generality_runs/text_supervision_generality_v0 \
  REUSE_INDEXES="/path/to/strength_prompt_interaction_v0/evaluation_index.json /path/to/text_supervision_generality_v0/evaluation_index.json" \
  RUN_ID=conditioning_interface_abc_v1 GPU_IDS=0,1,2,3 \
  DIFFUSERS_SRC=/linxi/packages/VLCP/diffusers/src \
  bash experiments/text_supervision_factorial/run_conditioning_interface_matrix.sh \
  > conditioning_interface_abc_v1.log 2>&1 < /dev/null &
```

For the targeted follow-up, use a fresh run ID. The scheduler prioritizes the missing ImageNette IPC50 shuffled
control, then the ImageWoof causal ladder, IPC10/20 curve, and IPC50 curve. A 13-hour invocation stops cleanly;
repeat the identical command and run ID on subsequent nights to continue the same frozen manifest:

```bash
nohup env \
  MATRICES="D C" WOOF_PHASES="ladder curve_ipc10_20 curve_ipc50" MAX_WALLTIME_HOURS=13 \
  RUN_ID=conditioning_interface_generality_v0 GPU_IDS=0,1,2,3 \
  REUSE_INDEXES="./strength_prompt_interaction_runs/strength_prompt_interaction_v0/evaluation_index.json ./conditioning_interface_matrix_runs/conditioning_interface_abc_v0/evaluation_index.json" \
  bash experiments/text_supervision_factorial/run_conditioning_interface_matrix.sh \
  > conditioning_interface_generality_v0.log 2>&1 < /dev/null &
```
Do not change `MATRICES`, `WOOF_PHASES`, seeds, or paths when resuming. `MAX_WALLTIME_HOURS` may be changed because
it controls scheduling only and is deliberately excluded from the scientific manifest.
When Matrix D is selected, startup requires all 32 existing Label/Correct IPC50 cells to be found in the reuse
catalog. This prevents an incorrect index path from silently rerunning them. `ALLOW_D_REGENERATION=true` is an
explicit escape hatch, not the recommended default.

`NETTE_CAPTION_FILE` defaults to `$NETTE_DATA_ROOT/train/nette.jsonl`. When Matrix C is enabled,
`WOOF_DATA_ROOT` defaults to the sibling `ImageWoof` directory and `WOOF_CAPTION_FILE` prefers
`$WOOF_DATA_ROOT/train/woof.jsonl` (with root-level `woof.jsonl` and legacy `train/metadata.jsonl` fallbacks).
The declared ImageNette/model/historical-run paths are also built-in defaults, while every value remains
overridable through the environment. Compatible `evaluation_index.json` files under the three historical
run roots are discovered automatically; use `REUSE_INDEXES` only to append indexes from separate runs such as
the earlier strength sweep.

Resume with the identical command and `RUN_ID`. `run_manifest.json` rejects changed paths or settings. To run a
subset without changing code, set `MATRICES="A B"` and use a different `RUN_ID`.

The summary uses fixed shift `S1` for comparisons across every strength. At the four preregistered robustness
strengths it also averages shuffle shifts inside each training-seed x generation-seed x classifier-repeat cell
before applying the hierarchical bootstrap. Its estimands are:

- `Correct - Label`: utility of the matched descriptive prompt;
- `Shuffled-S1 - Label`: primary utility of the class-level descriptive prompt marginal;
- `Correct - Shuffled-S1`: primary value of cluster-level correspondence;
- the corresponding `mean(Shuffled)` contrasts: randomization-robustness estimates at strengths with four shifts.

Outputs are `summary/performance.csv`, `summary/contrasts.csv`, `summary/shuffle_shift_effects.csv`, and
`summary/conditioning_interface_matrix_summary.json`. The summarizer also performs preregistered paired
difference-in-differences tests. It decomposes prompt utility into
`descriptive_marginal = mean(Correct, Shuffled-S1) - Label` and
`correspondence = Correct - Shuffled-S1`, then reports:

- strength interactions relative to `strength=0.7`;
- checkpoint x prompt interactions, including Matched-FT versus Frozen/Empty-FT;
- ImageWoof minus ImageNette dataset interactions at matched IPC/strength/seeds;
- dataset x strength interactions relative to `strength=0.7`.

The estimates and hierarchical bootstrap intervals are written to `summary/formal_interactions.csv`; the
underlying repeat-paired values are retained in `summary/formal_interaction_cells.csv`, and the overview plot is
`summary/conditioning_interface_formal_interactions.png`. Shuffle shifts are randomization realizations, not
independent experimental units.
## Schedule-matched visual-content control

The schedule-matched follow-up separates prototype content from the shorter
img2img denoising schedule. `schedule_matched_noise` uses the same img2img
timesteps, strength, prompt, and diffusion RNG as `prototype`, but replaces the
prototype latent with an independently seeded standard-normal latent. Because
the scheduler mixes two independent standard normals, its noised latent keeps a
standard-normal marginal while carrying no prototype content.

The four-GPU runner evaluates ImageNette and ImageWoof at IPC50 for Label-FT
and Matched-FT. It generates schedule-matched controls at strengths 0.7 and
0.9, completes the pure-noise endpoint, reuses existing Matched-FT cells and
prototype references, and then computes DINO cluster retention from
real-image-only VAE assignments:

```bash
GPU_IDS=0,1,2,3 \
INTERFACE_RUN_ROOT=./conditioning_interface_matrix_runs/conditioning_interface_abc_v0 \
WOOF_MODEL_ROOT=./conditioning_interface_matrix_runs/conditioning_interface_generality_v0 \
bash experiments/text_supervision_factorial/run_schedule_matched_followup.sh
```

`INTERFACE_RUN_ROOT` supplies Woof artifacts and historical indexes, while
`WOOF_MODEL_ROOT` supplies `models/woof/train_seed_*/{label_ft,matched_ft}`.
They are intentionally separate because the completed Woof checkpoints live
under `conditioning_interface_generality_v0`, not the earlier `abc_v0` run.

After completing the original Matched-FT run, the default covariance run ID
`prototype_checkpoint_covariance_v0` automatically reuses
`schedule_matched_followup_v0/evaluation_index.json` and only schedules the
missing Label-FT cells. Set `SUPERVISIONS=label_ft` to make that incremental
intent explicit. The summary now derives matrix panels dynamically, so E/F/R
runs no longer produce an empty legacy A/B/C plot. It also reports direct
`Matched-FT - Label-FT` prompt interactions and same-matrix Woof-Nette
interactions.

The runner is resume-safe. Set `MAX_WALLTIME_HOURS=13` to stop launching new
tasks after 13 hours while allowing active jobs to finish. Set
`RUN_RETENTION=false` to postpone the final VAE-assignment and DINO pass.

Primary outputs are:

- `evaluation_index.json`: reusable synthetic/evaluation cell inventory;
- `summary/`: downstream prompt effects;
- `visual_retention/visual_retention_per_cell.csv`: generated cluster Top-1,
  source-centroid cosine, and target margin;
- `visual_retention/retention_vs_prompt_utility.csv`: measured retention joined
  to descriptive marginal utility and correspondence value;
- `visual_retention/retention_utility_correlations.csv`: preregistered Spearman
  associations with bootstrap intervals.

`pure_noise` remains a full txt2img endpoint. It is deliberately distinct from
`schedule_matched_noise`, which is the causal control for prototype content at
an unchanged shortened denoising schedule.
