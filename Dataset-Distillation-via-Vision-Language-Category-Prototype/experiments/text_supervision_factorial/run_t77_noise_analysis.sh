#!/usr/bin/env bash
set -euo pipefail

LOW_RUN="${LOW_RUN:-./sparse_prompt_search_runs/sparse_t77_refit_seed01_v0}"
BOUNDARY_RUN="${BOUNDARY_RUN:-./sparse_prompt_search_runs/random_sparse_marginal_boundary_t77_v0}"
HIGH_RUN="${HIGH_RUN:-./sparse_prompt_search_runs/random_sparse_marginal_high_budget_t77_v0}"
OUTPUT_DIR="${OUTPUT_DIR:-./sparse_prompt_search_runs/t77_noise_analysis_v0}"

python experiments/text_supervision_factorial/analyze_t77_noise.py \
  --sparse-index \
    "${LOW_RUN}/sparse_evaluation_index.json" \
    "${BOUNDARY_RUN}/evaluation_index.json" \
    "${HIGH_RUN}/evaluation_index.json" \
  --fixed-index "${LOW_RUN}/fixed_evaluation_index.json" \
  --output-dir "${OUTPUT_DIR}" \
  --bootstrap-samples 10000
