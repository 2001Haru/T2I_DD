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
  NETTE_CAPTION_FILE=/linxi/dataset/VLCP/ImageNette/train/metadata.jsonl \
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
  --output /linxi/dataset/VLCP/ImageWoof/train/metadata.jsonl
```

After captions are complete, resume the same run with only the held-out replication phase:

```bash
NETTE_DATA_ROOT=/linxi/dataset/VLCP/ImageNette \
NETTE_CAPTION_FILE=/linxi/dataset/VLCP/ImageNette/train/metadata.jsonl \
WOOF_DATA_ROOT=/linxi/dataset/VLCP/ImageWoof \
WOOF_CAPTION_FILE=/linxi/dataset/VLCP/ImageWoof/train/metadata.jsonl \
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
