#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$EXPERIMENT_DIR/../.." && pwd)"

NETTE_DATA_ROOT="${NETTE_DATA_ROOT:-/linxi/dataset/VLCP/ImageNette}"
NETTE_CAPTION_FILE="${NETTE_CAPTION_FILE:-$NETTE_DATA_ROOT/train/nette.jsonl}"
BASE_MODEL="${BASE_MODEL:-/linxi/models/VLCP/stable-diffusion-v1-5}"
BASE_RUN_ROOT="${BASE_RUN_ROOT:-$REPO_ROOT/text_supervision_factorial_runs/text_supervision_factorial_2xa100_v0}"
CAUSAL_RUN_ROOT="${CAUSAL_RUN_ROOT:-$REPO_ROOT/text_supervision_factorial_runs/text_supervision_causal_ladder_v0}"
GENERALITY_RUN_ROOT="${GENERALITY_RUN_ROOT:-$REPO_ROOT/text_supervision_generality_runs/text_supervision_generality_v0}"

RUN_ID="${RUN_ID:-conditioning_interface_abc_v0}"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/conditioning_interface_matrix_runs/$RUN_ID}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
MATRICES="${MATRICES:-A B C}"
WOOF_PHASES="${WOOF_PHASES:-ladder curve_ipc10_20 curve_ipc50}"
TRAINING_SEEDS="${TRAINING_SEEDS:-0 1}"
GENERATION_SEEDS="${GENERATION_SEEDS:-0 1}"
CLASSIFIER_REPEATS="${CLASSIFIER_REPEATS:-2}"
MAX_PARALLEL_EVALS="${MAX_PARALLEL_EVALS:-4}"
NUM_WORKERS="${NUM_WORKERS:-2}"
DIFFUSERS_SRC="${DIFFUSERS_SRC:-}"
MAX_WALLTIME_HOURS="${MAX_WALLTIME_HOURS:-0}"

# Reuse compatible completed cells by default. Additional strength-run indexes
# can still be appended explicitly through REUSE_INDEXES.
default_reuse_indexes=()
for candidate in \
  "$BASE_RUN_ROOT/evaluation_index.json" \
  "$CAUSAL_RUN_ROOT/evaluation_index.json" \
  "$GENERALITY_RUN_ROOT/evaluation_index.json"; do
  [[ -f "$candidate" ]] && default_reuse_indexes+=("$candidate")
done
explicit_reuse_indexes="${REUSE_INDEXES:-}"
REUSE_INDEXES="${default_reuse_indexes[*]}${explicit_reuse_indexes:+ $explicit_reuse_indexes}"

read -r -a matrix_args <<< "$MATRICES"
read -r -a training_seed_args <<< "$TRAINING_SEEDS"
read -r -a generation_seed_args <<< "$GENERATION_SEEDS"
read -r -a woof_phase_args <<< "$WOOF_PHASES"
read -r -a reuse_index_args <<< "$REUSE_INDEXES"

if [[ " $MATRICES " == *" C "* ]]; then
  WOOF_DATA_ROOT="${WOOF_DATA_ROOT:-$(dirname "$NETTE_DATA_ROOT")/ImageWoof}"
  if [[ -z "${WOOF_CAPTION_FILE:-}" ]]; then
    for candidate in "$WOOF_DATA_ROOT/train/woof.jsonl" "$WOOF_DATA_ROOT/woof.jsonl" "$WOOF_DATA_ROOT/train/metadata.jsonl"; do
      if [[ -f "$candidate" ]]; then
        WOOF_CAPTION_FILE="$candidate"
        break
      fi
    done
    WOOF_CAPTION_FILE="${WOOF_CAPTION_FILE:-$WOOF_DATA_ROOT/train/woof.jsonl}"
  fi
fi

args=(
  --nette-data-root "$NETTE_DATA_ROOT" --nette-caption-file "$NETTE_CAPTION_FILE"
  --base-model "$BASE_MODEL" --base-run-root "$BASE_RUN_ROOT"
  --causal-run-root "$CAUSAL_RUN_ROOT" --generality-run-root "$GENERALITY_RUN_ROOT"
  --run-root "$RUN_ROOT" --gpus "$GPU_IDS" --matrices "${matrix_args[@]}"
  --woof-phases "${woof_phase_args[@]}" --max-walltime-hours "$MAX_WALLTIME_HOURS"
  --training-seeds "${training_seed_args[@]}" --generation-seeds "${generation_seed_args[@]}"
  --classifier-repeats "$CLASSIFIER_REPEATS" --max-parallel-evals "$MAX_PARALLEL_EVALS"
  --train-batch-size 8 --gradient-accumulation-steps 4 --num-workers "$NUM_WORKERS"
  --mixed-precision fp16
)
if [[ " $MATRICES " == *" C "* ]]; then
  args+=(--woof-data-root "$WOOF_DATA_ROOT" --woof-caption-file "$WOOF_CAPTION_FILE")
fi
[[ -n "$DIFFUSERS_SRC" ]] && args+=(--diffusers-src "$DIFFUSERS_SRC")
[[ "${ALLOW_D_REGENERATION:-false}" == "true" ]] && args+=(--allow-d-regeneration)
for index_path in "${reuse_index_args[@]}"; do
  args+=(--reuse-index "$index_path")
done

echo "Persistent A/B/C conditioning-interface matrix: $RUN_ROOT"
echo "Matrices: $MATRICES; GPUs: $GPU_IDS; G condition: removed"
exec python "$EXPERIMENT_DIR/run_conditioning_interface_matrix.py" "${args[@]}"
