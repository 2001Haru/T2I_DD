#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXPERIMENT_DIR="$REPO_ROOT/experiments/text_supervision_factorial"

: "${DATA_ROOT:?Set DATA_ROOT to ImageNette containing train/ and val/}"
: "${BASE_MODEL:?Set BASE_MODEL to the local SD1.5 Diffusers pipeline}"
: "${BASE_RUN_ROOT:?Set BASE_RUN_ROOT to the completed original 4x3 run root}"

CAPTION_FILE="${CAPTION_FILE:-}"
if [[ -z "$CAPTION_FILE" ]]; then
  for candidate in "$DATA_ROOT/train/metadata.jsonl" "$DATA_ROOT/train/nette.jsonl" "$DATA_ROOT/nette.jsonl"; do
    if [[ -f "$candidate" ]]; then CAPTION_FILE="$candidate"; break; fi
  done
fi
: "${CAPTION_FILE:?No ImageNette caption JSONL found; set CAPTION_FILE}"

RUN_ID="${RUN_ID:-text_supervision_causal_ladder_v0}"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/text_supervision_factorial_runs/$RUN_ID}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
GENERATION_SEEDS="${GENERATION_SEEDS:-0 1}"
CLASSIFIER_REPEATS="${CLASSIFIER_REPEATS:-2}"
CLASSIFIER_SEED="${CLASSIFIER_SEED:-0}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
NUM_WORKERS_PER_TRAIN="${NUM_WORKERS_PER_TRAIN:-4}"
MIXED_PRECISION="${MIXED_PRECISION:-fp16}"
MAX_PARALLEL_EVALS="${MAX_PARALLEL_EVALS:-2}"
RETRY_DELAY_SECONDS="${RETRY_DELAY_SECONDS:-120}"
MAX_RETRIES="${MAX_RETRIES:-0}"
CONSTANT_PROMPT="${CONSTANT_PROMPT:-A natural photo.}"
DIFFUSERS_SRC="${DIFFUSERS_SRC:-}"

read -r -a generation_seed_args <<< "$GENERATION_SEEDS"
args=(
  --data-root "$DATA_ROOT"
  --base-model "$BASE_MODEL"
  --caption-file "$CAPTION_FILE"
  --base-run-root "$BASE_RUN_ROOT"
  --run-root "$RUN_ROOT"
  --gpus "$GPU_IDS"
  --generation-seeds "${generation_seed_args[@]}"
  --classifier-repeats "$CLASSIFIER_REPEATS"
  --classifier-seed "$CLASSIFIER_SEED"
  --train-batch-size "$TRAIN_BATCH_SIZE"
  --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS"
  --num-workers "$NUM_WORKERS_PER_TRAIN"
  --mixed-precision "$MIXED_PRECISION"
  --constant-prompt "$CONSTANT_PROMPT"
  --max-parallel-evals "$MAX_PARALLEL_EVALS"
  --retry-delay-seconds "$RETRY_DELAY_SECONDS"
  --max-retries "$MAX_RETRIES"
)
[[ -n "$DIFFUSERS_SRC" ]] && args+=(--diffusers-src "$DIFFUSERS_SRC")
[[ -n "${PROTOTYPE_PATH:-}" ]] && args+=(--prototype "$PROTOTYPE_PATH")
[[ -n "${DCS_PATH:-}" ]] && args+=(--dcs "$DCS_PATH")

echo "Persistent causal-ladder scheduler: $RUN_ROOT"
echo "GPUs: $GPU_IDS; failed tasks retry forever when MAX_RETRIES=0"
exec python "$EXPERIMENT_DIR/run_causal_ladder.py" "${args[@]}"
