#!/usr/bin/env bash
set -euo pipefail
shopt -s inherit_errexit

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXPERIMENT_DIR="$REPO_ROOT/experiments/visual_text_factorial"

BASE_RUN_ROOT="${BASE_RUN_ROOT:-$REPO_ROOT/../vlcp_factorial_runs/visual_text_factorial_v0}"
RANDOMIZATION_RUN_ID="${RANDOMIZATION_RUN_ID:-visual_text_shuffle_randomization_v0}"
RANDOMIZATION_ROOT="${RANDOMIZATION_ROOT:-$REPO_ROOT/../vlcp_shuffle_runs/$RANDOMIZATION_RUN_ID}"
DINO_MODEL="${DINO_MODEL:-/models/DINOv2/dinov2-base}"
SHUFFLE_SHIFTS="${SHUFFLE_SHIFTS:-2 4 7}"
DIAGNOSTICS_ID="${DIAGNOSTICS_ID:-semantic_coverage_v0}"
DIAGNOSTICS_ROOT="${DIAGNOSTICS_ROOT:-$RANDOMIZATION_ROOT/diagnostics/$DIAGNOSTICS_ID}"
CLUSTER_AUDIT_ROOT="${CLUSTER_AUDIT_ROOT:-$DIAGNOSTICS_ROOT/cluster_member_audit}"
CLUSTER_ASSIGNMENTS="${CLUSTER_ASSIGNMENTS:-$CLUSTER_AUDIT_ROOT/latent_assignments.csv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$DIAGNOSTICS_ROOT/real_member_recombination}"
FEATURE_CACHE_DIR="${FEATURE_CACHE_DIR:-$DIAGNOSTICS_ROOT/dino/feature_cache}"
DOWNSTREAM_CSV="${DOWNSTREAM_CSV:-$DIAGNOSTICS_ROOT/downstream_per_class/summary/downstream_dino_per_class.csv}"
ANCHOR_KS="${ANCHOR_KS:-3 5 9}"
HELDOUT_START="${HELDOUT_START:-10}"
HELDOUT_END="${HELDOUT_END:-30}"
BATCH_SIZE="${BATCH_SIZE:-64}"
DEVICE="${DEVICE:-cuda}"
RESUME="${RESUME:-false}"

for required in \
  "$BASE_RUN_ROOT" \
  "$RANDOMIZATION_ROOT" \
  "$DINO_MODEL" \
  "$CLUSTER_ASSIGNMENTS"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required path: $required" >&2
    exit 1
  fi
done

shuffle_run_args=()
for shift in $SHUFFLE_SHIFTS; do
  shift_root="$RANDOMIZATION_ROOT/shift_$shift"
  if [[ ! -d "$shift_root/synthetic" ]]; then
    echo "Missing shuffled run for shift $shift: $shift_root" >&2
    exit 1
  fi
  shuffle_run_args+=(--shuffle-run "$shift=$shift_root")
done

anchor_k_args=()
for anchor_k in $ANCHOR_KS; do
  anchor_k_args+=(--anchor-k "$anchor_k")
done

downstream_args=()
if [[ -f "$DOWNSTREAM_CSV" ]]; then
  downstream_args=(--downstream-csv "$DOWNSTREAM_CSV")
else
  echo "Downstream per-class CSV not found; geometry will run without downstream joins: $DOWNSTREAM_CSV"
fi

mkdir -p "$OUTPUT_ROOT"
CONFIG_FILE="$OUTPUT_ROOT/real_member_recombination_config.txt"
CONFIG_CONTENT="BASE_RUN_ROOT=$(realpath "$BASE_RUN_ROOT")
RANDOMIZATION_ROOT=$(realpath "$RANDOMIZATION_ROOT")
DINO_MODEL=$(realpath "$DINO_MODEL")
CLUSTER_ASSIGNMENTS=$(realpath "$CLUSTER_ASSIGNMENTS")
SHUFFLE_SHIFTS=$SHUFFLE_SHIFTS
ANCHOR_KS=$ANCHOR_KS
HELDOUT_START=$HELDOUT_START
HELDOUT_END=$HELDOUT_END
FEATURE_CACHE_DIR=$(realpath -m "$FEATURE_CACHE_DIR")
DOWNSTREAM_CSV=$(realpath -m "$DOWNSTREAM_CSV")
BATCH_SIZE=$BATCH_SIZE
DEVICE=$DEVICE"
if [[ -f "$CONFIG_FILE" ]]; then
  if [[ "$(<"$CONFIG_FILE")" != "$CONFIG_CONTENT" ]]; then
    echo "Resume configuration differs from $CONFIG_FILE" >&2
    exit 1
  fi
  if [[ "$RESUME" != "true" ]]; then
    echo "Real-member diagnostics already exist; set RESUME=true: $OUTPUT_ROOT" >&2
    exit 1
  fi
else
  printf '%s\n' "$CONFIG_CONTENT" > "$CONFIG_FILE.tmp"
  mv "$CONFIG_FILE.tmp" "$CONFIG_FILE"
fi

python - <<'PY'
import importlib

for package in ("matplotlib", "numpy", "PIL", "torch", "transformers"):
    importlib.import_module(package)
print("Real-member recombination diagnostic dependencies are available.")
PY

resume_args=()
if [[ "$RESUME" == "true" ]]; then
  resume_args=(--resume)
fi

python "$EXPERIMENT_DIR/diagnose_real_member_recombination.py" \
  --base-run-root "$BASE_RUN_ROOT" \
  "${shuffle_run_args[@]}" \
  --cluster-assignments "$CLUSTER_ASSIGNMENTS" \
  --dino-model "$DINO_MODEL" \
  --output-dir "$OUTPUT_ROOT" \
  --feature-cache-dir "$FEATURE_CACHE_DIR" \
  "${downstream_args[@]}" \
  "${anchor_k_args[@]}" \
  --heldout-start "$HELDOUT_START" \
  --heldout-end "$HELDOUT_END" \
  --device "$DEVICE" \
  --batch-size "$BATCH_SIZE" \
  "${resume_args[@]}"

echo "Real-member recombination diagnostics complete: $OUTPUT_ROOT"
