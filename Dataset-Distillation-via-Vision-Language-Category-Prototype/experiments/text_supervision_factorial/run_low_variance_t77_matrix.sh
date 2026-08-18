#!/usr/bin/env bash
set -euo pipefail

NETTE_DATA_ROOT="${NETTE_DATA_ROOT:-/linxi/dataset/VLCP/ImageNette}"
BASE_MODEL="${BASE_MODEL:-/linxi/models/VLCP/stable-diffusion-v1-5}"
GENERALITY_RUN_ROOT="${GENERALITY_RUN_ROOT:-./text_supervision_generality_runs/text_supervision_generality_v0}"
REFIT_RUN_ROOT="${REFIT_RUN_ROOT:-./sparse_prompt_search_runs/sparse_t77_refit_seed01_v0}"
HIGH_RUN_ROOT="${HIGH_RUN_ROOT:-./sparse_prompt_search_runs/random_sparse_marginal_high_budget_t77_v0}"
BASE_FT_ROOT="${BASE_FT_ROOT:-./text_supervision_factorial_runs/text_supervision_factorial_2xa100_v0}"
RUN_ROOT="${RUN_ROOT:-./low_variance_t77_runs/low_variance_t77_g6_c3_v0}"
GPU_IDS="${GPU_IDS:-0,1}"
DIFFUSERS_SRC="${DIFFUSERS_SRC:-/linxi/packages/VLCP/diffusers/src}"
EXTRA_ARGS=()
if [[ "${AUDIT_ONLY:-false}" == "true" ]]; then
  EXTRA_ARGS+=(--audit-only)
fi

PROTOTYPE="${PROTOTYPE:-${GENERALITY_RUN_ROOT}/artifacts/nette/ipc50/nette-ipc50-0.7-30-kmexpand1.json}"
DCS="${DCS:-${GENERALITY_RUN_ROOT}/artifacts/nette/ipc50/dcs.json}"
MATCHED_MODEL="${MATCHED_MODEL:-${REFIT_RUN_ROOT}/fixed/models/train_seed_0/matched_ft}"
UNPAIRED_MODEL="${UNPAIRED_MODEL:-${REFIT_RUN_ROOT}/fixed/models/train_seed_0/unpaired_ft}"
BANK_M4_MODEL="${BANK_M4_MODEL:-${REFIT_RUN_ROOT}/sparse/models/bank_seed_0/m_4/sparse_ft}"
BANK_M4_JSON="${BANK_M4_JSON:-${REFIT_RUN_ROOT}/sparse/caption_banks/bank_seed_0/m_4.json}"
BANK_M64_MODEL="${BANK_M64_MODEL:-${HIGH_RUN_ROOT}/models/bank_seed_0/m_64/sparse_ft}"
BANK_M64_JSON="${BANK_M64_JSON:-${HIGH_RUN_ROOT}/caption_banks/bank_seed_0/m_64.json}"
LABEL_MODEL="${LABEL_MODEL:-${BASE_FT_ROOT}/models/label_ft}"

python experiments/text_supervision_factorial/run_low_variance_t77_matrix.py \
  --data-root "${NETTE_DATA_ROOT}" \
  --base-model "${BASE_MODEL}" \
  --prototype "${PROTOTYPE}" \
  --dcs "${DCS}" \
  --matched-model "${MATCHED_MODEL}" \
  --unpaired-model "${UNPAIRED_MODEL}" \
  --bank-m4-model "${BANK_M4_MODEL}" \
  --bank-m4-json "${BANK_M4_JSON}" \
  --bank-m64-model "${BANK_M64_MODEL}" \
  --bank-m64-json "${BANK_M64_JSON}" \
  --label-model "${LABEL_MODEL}" \
  --run-root "${RUN_ROOT}" \
  --gpus "${GPU_IDS}" \
  --generation-seeds 0 1 2 3 4 5 \
  --classifier-repeats 3 \
  --classifier-seed 0 \
  --tail-k 10 \
  --ipc 50 \
  --strength 0.8 \
  --diffusers-src "${DIFFUSERS_SRC}" \
  "${EXTRA_ARGS[@]}"
