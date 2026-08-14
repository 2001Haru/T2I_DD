#!/usr/bin/env bash
set -euo pipefail

BASE_MODEL="${BASE_MODEL:-/linxi/models/VLCP/stable-diffusion-v1-5}"
NETTE_DATA_ROOT="${NETTE_DATA_ROOT:-/linxi/dataset/VLCP/ImageNette}"
WOOF_DATA_ROOT="${WOOF_DATA_ROOT:-/linxi/dataset/VLCP/ImageWoof}"
OUTPUT_DIR="${OUTPUT_DIR:-caption_interface_audit_runs/caption_interface_audit_v0}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-64}"

first_file() {
  local candidate
  for candidate in "$@"; do
    if [[ -f "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

NETTE_CAPTIONS="${NETTE_CAPTIONS:-$(first_file \
  "$NETTE_DATA_ROOT/train/nette.jsonl" \
  "$NETTE_DATA_ROOT/nette.jsonl")}" || {
  echo "Set NETTE_CAPTIONS to the ImageNette caption JSONL." >&2
  exit 1
}
WOOF_CAPTIONS="${WOOF_CAPTIONS:-$(first_file \
  "$WOOF_DATA_ROOT/train/woof.jsonl" \
  "$WOOF_DATA_ROOT/woof.jsonl" \
  "$(dirname "$NETTE_DATA_ROOT")/ImageWoof/train/woof.jsonl" \
  "$(dirname "$NETTE_DATA_ROOT")/ImageWoof/woof.jsonl")}" || {
  echo "Set WOOF_CAPTIONS to the ImageWoof caption JSONL." >&2
  exit 1
}

NETTE_DCS="${NETTE_DCS:-$(first_file \
  text_supervision_generality_runs/text_supervision_generality_v0/artifacts/nette/ipc50/dcs.json \
  conditioning_interface_matrix_runs/conditioning_interface_generality_v0/artifacts/nette/ipc50/dcs.json)}" || true
WOOF_DCS="${WOOF_DCS:-$(first_file \
  conditioning_interface_matrix_runs/conditioning_interface_generality_v0/artifacts/woof/ipc50/dcs.json \
  text_supervision_generality_runs/text_supervision_generality_v0/artifacts/woof/ipc50/dcs.json)}" || true

NETTE_BANK="${NETTE_BANK:-$(first_file \
  sparse_prompt_search_runs/random_sparse_marginal_v0/caption_banks/bank_seed_0/m_4.json \
  sparse_prompt_search_runs/sparse_m4_generality_v0/nette/train_seed_1/caption_banks/bank_seed_0/m_4.json)}" || true
WOOF_BANK="${WOOF_BANK:-$(first_file \
  sparse_prompt_search_runs/sparse_m4_generality_v0/woof/train_seed_1/caption_banks/bank_seed_0/m_4.json \
  sparse_prompt_search_runs/sparse_m4_generality_v0/woof/train_seed_2/caption_banks/bank_seed_0/m_4.json)}" || true

NETTE_ASSIGNMENTS="${NETTE_ASSIGNMENTS:-$(first_file \
  schedule_matched_followup_runs/prototype_checkpoint_covariance_v0/retention_inputs/nette/latent_assignments.csv \
  schedule_matched_followup_runs/schedule_matched_checkpoint_v1/retention_inputs/nette/latent_assignments.csv \
  schedule_matched_followup_runs/schedule_matched_followup_v0/retention_inputs/nette/latent_assignments.csv)}" || true
WOOF_ASSIGNMENTS="${WOOF_ASSIGNMENTS:-$(first_file \
  schedule_matched_followup_runs/prototype_checkpoint_covariance_v0/retention_inputs/woof/latent_assignments.csv \
  schedule_matched_followup_runs/schedule_matched_checkpoint_v1/retention_inputs/woof/latent_assignments.csv \
  schedule_matched_followup_runs/schedule_matched_followup_v0/retention_inputs/woof/latent_assignments.csv)}" || true

args=(
  --base-model "$BASE_MODEL"
  --dataset "nette=$NETTE_CAPTIONS"
  --dataset "woof=$WOOF_CAPTIONS"
  --output-dir "$OUTPUT_DIR"
  --device "$DEVICE"
  --batch-size "$BATCH_SIZE"
  --local-files-only
)

if [[ -n "$NETTE_DCS" ]]; then args+=(--dcs "nette=$NETTE_DCS"); fi
if [[ -n "$WOOF_DCS" ]]; then args+=(--dcs "woof=$WOOF_DCS"); fi
if [[ -n "$NETTE_BANK" ]]; then args+=(--bank "nette=$NETTE_BANK"); fi
if [[ -n "$WOOF_BANK" ]]; then args+=(--bank "woof=$WOOF_BANK"); fi
if [[ -n "$NETTE_ASSIGNMENTS" ]]; then args+=(--assignment "nette=$NETTE_ASSIGNMENTS"); fi
if [[ -n "$WOOF_ASSIGNMENTS" ]]; then args+=(--assignment "woof=$WOOF_ASSIGNMENTS"); fi
if [[ "${SKIP_PROBE:-false}" == "true" ]]; then args+=(--skip-probe); fi
if [[ "${FORCE_FEATURES:-false}" == "true" ]]; then args+=(--force-features); fi

echo "Caption-interface audit inputs:"
printf '  base model: %s\n  Nette captions: %s\n  Woof captions: %s\n' \
  "$BASE_MODEL" "$NETTE_CAPTIONS" "$WOOF_CAPTIONS"
printf '  Nette DCS: %s\n  Woof DCS: %s\n  Nette bank: %s\n  Woof bank: %s\n' \
  "${NETTE_DCS:-<not found>}" "${WOOF_DCS:-<not found>}" \
  "${NETTE_BANK:-<not found>}" "${WOOF_BANK:-<not found>}"
printf '  Nette assignments: %s\n  Woof assignments: %s\n' \
  "${NETTE_ASSIGNMENTS:-<not found>}" "${WOOF_ASSIGNMENTS:-<not found>}"

python experiments/text_supervision_factorial/audit_caption_interface.py "${args[@]}"
