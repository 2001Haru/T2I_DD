#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$EXPERIMENT_DIR/../.." && pwd)"

NETTE_DATA_ROOT="${NETTE_DATA_ROOT:-/linxi/dataset/VLCP/ImageNette}"
BASE_MODEL="${BASE_MODEL:-/linxi/models/VLCP/stable-diffusion-v1-5}"
DIFFUSERS_SRC="${DIFFUSERS_SRC:-/linxi/packages/VLCP/diffusers}"
GENERALITY_RUN_ROOT="${GENERALITY_RUN_ROOT:-$REPO_ROOT/text_supervision_generality_runs/text_supervision_generality_v0}"
BASE_RUN_ROOT="${BASE_RUN_ROOT:-$REPO_ROOT/text_supervision_factorial_runs/text_supervision_factorial_2xa100_v0}"
CAUSAL_RUN_ROOT="${CAUSAL_RUN_ROOT:-$REPO_ROOT/text_supervision_factorial_runs/text_supervision_causal_ladder_v0}"
OLD_SPARSE_ROOT="${OLD_SPARSE_ROOT:-$REPO_ROOT/sparse_prompt_search_runs/random_sparse_marginal_v0}"
SPARSE_GENERALITY_ROOT="${SPARSE_GENERALITY_ROOT:-$REPO_ROOT/sparse_prompt_search_runs/sparse_m4_generality_v0}"
MATRIX_R_ROOT="${MATRIX_R_ROOT:-$REPO_ROOT/schedule_matched_followup_runs/prototype_checkpoint_covariance_v0}"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/sparse_prompt_search_runs/sparse_interface_transfer_v0}"
GPU_IDS="${GPU_IDS:-0,1}"
CLASSIFIER_REPEATS="${CLASSIFIER_REPEATS:-2}"

PROTOTYPE="${PROTOTYPE:-$GENERALITY_RUN_ROOT/artifacts/nette/ipc50/nette-ipc50-0.7-30-kmexpand1.json}"
DCS="${DCS:-$GENERALITY_RUN_ROOT/artifacts/nette/ipc50/dcs.json}"
PROMPT_BANK="${PROMPT_BANK:-$OLD_SPARSE_ROOT/caption_banks/bank_seed_0/m_4.json}"
SPARSE_SEED0_MODEL="${SPARSE_SEED0_MODEL:-$OLD_SPARSE_ROOT/models/bank_seed_0/m_4/sparse_ft}"
SPARSE_SEED1_MODEL="${SPARSE_SEED1_MODEL:-$SPARSE_GENERALITY_ROOT/nette/train_seed_1/models/bank_seed_0/m_4/sparse_ft}"
MATCHED_SEED0_MODEL="${MATCHED_SEED0_MODEL:-$BASE_RUN_ROOT/models/matched_ft}"
MATCHED_SEED1_MODEL="${MATCHED_SEED1_MODEL:-$CAUSAL_RUN_ROOT/models/train_seed_1/matched_ft}"

reuse=()
for candidate in \
  "$OLD_SPARSE_ROOT/evaluation_index.json" \
  "$SPARSE_GENERALITY_ROOT/nette/train_seed_1/evaluation_index.json" \
  "$MATRIX_R_ROOT/evaluation_index.json"; do
  [[ -f "$candidate" ]] && reuse+=(--reuse-index "$candidate")
done
bank_copies=()
for candidate in \
  "$SPARSE_GENERALITY_ROOT/nette/train_seed_1/caption_banks/bank_seed_0/m_4.json"; do
  [[ -f "$candidate" ]] && bank_copies+=(--bank-copy "$candidate")
done

export PYTHONPATH="$DIFFUSERS_SRC${PYTHONPATH:+:$PYTHONPATH}"
python "$EXPERIMENT_DIR/run_sparse_interface_transfer.py" \
  --data-root "$NETTE_DATA_ROOT" --base-model "$BASE_MODEL" \
  --prototype "$PROTOTYPE" --dcs "$DCS" --prompt-bank "$PROMPT_BANK" \
  --sparse-seed0-model "$SPARSE_SEED0_MODEL" --sparse-seed1-model "$SPARSE_SEED1_MODEL" \
  --matched-seed0-model "$MATCHED_SEED0_MODEL" --matched-seed1-model "$MATCHED_SEED1_MODEL" \
  --run-root "$RUN_ROOT" --gpus "$GPU_IDS" --training-seeds 0 1 --generation-seeds 0 1 \
  --ipc 50 --strength 0.8 --classifier-repeats "$CLASSIFIER_REPEATS" --diffusers-src "$DIFFUSERS_SRC" \
  "${bank_copies[@]}" "${reuse[@]}"
