#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$EXPERIMENT_DIR/../.." && pwd)"

: "${NETTE_DATA_ROOT:=/linxi/dataset/VLCP/ImageNette}"
: "${WOOF_DATA_ROOT:=/linxi/dataset/VLCP/ImageWoof}"
: "${BASE_MODEL:=/linxi/models/VLCP/stable-diffusion-v1-5}"
: "${GENERALITY_RUN_ROOT:=$REPO_ROOT/text_supervision_generality_runs/text_supervision_generality_v0}"
: "${CONDITIONING_RUN_ROOT:=$REPO_ROOT/conditioning_interface_matrix_runs/conditioning_interface_abc_v0}"
: "${RUN_ROOT:=$REPO_ROOT/sparse_prompt_search_runs/sparse_m4_generality_v0}"
: "${GPUS:=0,1}"
: "${DIFFUSERS_SRC:=/linxi/packages/VLCP/diffusers}"
: "${OLD_NETTE_SPARSE_INDEX:=$REPO_ROOT/sparse_prompt_search_runs/random_sparse_marginal_v0/evaluation_index.json}"
: "${SPARSE_CONTROL_INDEX:=$REPO_ROOT/sparse_prompt_search_runs/random_sparse_marginal_controls_v1/evaluation_index.json}"
: "${INTERFACE_INDEX:=$REPO_ROOT/conditioning_interface_matrix_runs/conditioning_interface_abc_v0/evaluation_index.json}"
: "${COVARIANCE_INDEX:=$REPO_ROOT/schedule_matched_followup_runs/prototype_checkpoint_covariance_v0/evaluation_index.json}"

NETTE_CAPTION_FILE="${NETTE_CAPTION_FILE:-$NETTE_DATA_ROOT/train/nette.jsonl}"
if [[ -z "${WOOF_CAPTION_FILE:-}" ]]; then
  for candidate in "$WOOF_DATA_ROOT/train/woof.jsonl" "$WOOF_DATA_ROOT/woof.jsonl"; do
    if [[ -f "$candidate" ]]; then
      WOOF_CAPTION_FILE="$candidate"
      break
    fi
  done
  WOOF_CAPTION_FILE="${WOOF_CAPTION_FILE:-$WOOF_DATA_ROOT/train/woof.jsonl}"
fi
NETTE_PROTOTYPE="${NETTE_PROTOTYPE:-$GENERALITY_RUN_ROOT/artifacts/nette/ipc50/nette-ipc50-0.7-30-kmexpand1.json}"
NETTE_DCS="${NETTE_DCS:-$GENERALITY_RUN_ROOT/artifacts/nette/ipc50/dcs.json}"
WOOF_PROTOTYPE="${WOOF_PROTOTYPE:-$CONDITIONING_RUN_ROOT/artifacts/woof/ipc50/woof-ipc50-0.7-30-kmexpand1.json}"
WOOF_DCS="${WOOF_DCS:-$CONDITIONING_RUN_ROOT/artifacts/woof/ipc50/dcs.json}"

SUMMARY_ARGS=()
if [[ -f "$OLD_NETTE_SPARSE_INDEX" ]]; then
  SUMMARY_ARGS+=(--old-nette-sparse-index "$OLD_NETTE_SPARSE_INDEX")
fi
for index in "$SPARSE_CONTROL_INDEX" "$INTERFACE_INDEX" "$COVARIANCE_INDEX"; do
  if [[ -f "$index" ]]; then
    SUMMARY_ARGS+=(--control-index "$index")
  fi
done

python "$EXPERIMENT_DIR/run_sparse_generality.py" \
  --nette-data-root "$NETTE_DATA_ROOT" --nette-caption-file "$NETTE_CAPTION_FILE" \
  --woof-data-root "$WOOF_DATA_ROOT" --woof-caption-file "$WOOF_CAPTION_FILE" \
  --base-model "$BASE_MODEL" \
  --nette-prototype "$NETTE_PROTOTYPE" --nette-dcs "$NETTE_DCS" \
  --woof-prototype "$WOOF_PROTOTYPE" --woof-dcs "$WOOF_DCS" \
  --run-root "$RUN_ROOT" --gpus "$GPUS" \
  --nette-training-seeds 1 2 --woof-training-seeds 0 1 \
  --generation-seeds 0 1 --ipc 50 --strength 0.8 \
  --classifier-repeats 2 --diffusers-src "$DIFFUSERS_SRC" \
  "${SUMMARY_ARGS[@]}"
