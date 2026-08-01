#!/usr/bin/env bash
set -euo pipefail
shopt -s inherit_errexit

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXPERIMENT_DIR="$REPO_ROOT/experiments/visual_text_factorial"
EVALUATION_DIR="$REPO_ROOT/04_evaluation/Minimax"

: "${DATA_ROOT:?Set DATA_ROOT to the prepared ImageNette root containing train/ and val/}"

SOURCE_RUN_ROOT="${SOURCE_RUN_ROOT:-$REPO_ROOT/../vlcp_ablation_runs/author_checkpoint_pilot_v0}"
BASE_MODEL="${BASE_MODEL:-/linxi/models/VLCP/stable-diffusion-v1-5}"
PROTOTYPE_PATH="${PROTOTYPE_PATH:-$SOURCE_RUN_ROOT/prototypes/prior_alignment-ipc10-0.7-30-kmexpand1.json}"
DCS_PATH="${DCS_PATH:-$SOURCE_RUN_ROOT/prototypes/dcs.json}"
REFERENCE_RANDOMIZATION_ROOT="${REFERENCE_RANDOMIZATION_ROOT:-$REPO_ROOT/../vlcp_shuffle_runs/visual_text_shuffle_randomization_v0}"
CLUSTER_SUMMARY="${CLUSTER_SUMMARY:-$REFERENCE_RANDOMIZATION_ROOT/diagnostics/semantic_coverage_v0/cluster_member_audit/cluster_distance_summary.csv}"
EXTENSION_ID="${EXTENSION_ID:-selective_small_cluster_shuffle_seed23_v0}"
SOURCE_BASE_ROOT="${SOURCE_BASE_ROOT:-$REPO_ROOT/../vlcp_factorial_runs/${EXTENSION_ID}_sources_shift1}"
SOURCE_SHUFFLE_ROOT="${SOURCE_SHUFFLE_ROOT:-$REPO_ROOT/../vlcp_shuffle_runs/${EXTENSION_ID}_sources}"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/../vlcp_selective_shuffle_runs/$EXTENSION_ID}"
GENERATION_SEEDS="${GENERATION_SEEDS:-2 3}"
SHUFFLE_SHIFTS="${SHUFFLE_SHIFTS:-1 2 4 7}"
SELECTED_COUNT="${SELECTED_COUNT:-3}"
RANDOM_TARGET_SEED="${RANDOM_TARGET_SEED:-20260731}"
LINK_MODE="${LINK_MODE:-symlink}"
CLASSIFIER_SEED="${CLASSIFIER_SEED:-0}"
RESUME="${RESUME:-false}"

for required in "$DATA_ROOT" "$SOURCE_RUN_ROOT" "$PROTOTYPE_PATH" "$DCS_PATH" "$CLUSTER_SUMMARY"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required path: $required" >&2
    exit 1
  fi
done

mkdir -p "$RUN_ROOT" "$SOURCE_SHUFFLE_ROOT"
CONFIG_FILE="$RUN_ROOT/seed_extension_config.txt"
CONFIG_CONTENT="DATA_ROOT=$(realpath "$DATA_ROOT")
SOURCE_RUN_ROOT=$(realpath "$SOURCE_RUN_ROOT")
BASE_MODEL=$BASE_MODEL
PROTOTYPE_PATH=$(realpath "$PROTOTYPE_PATH")
DCS_PATH=$(realpath "$DCS_PATH")
CLUSTER_SUMMARY=$(realpath "$CLUSTER_SUMMARY")
SOURCE_BASE_ROOT=$(realpath -m "$SOURCE_BASE_ROOT")
SOURCE_SHUFFLE_ROOT=$(realpath -m "$SOURCE_SHUFFLE_ROOT")
GENERATION_SEEDS=$GENERATION_SEEDS
SHUFFLE_SHIFTS=$SHUFFLE_SHIFTS
SELECTED_COUNT=$SELECTED_COUNT
RANDOM_TARGET_SEED=$RANDOM_TARGET_SEED
LINK_MODE=$LINK_MODE
CLASSIFIER_SEED=$CLASSIFIER_SEED"
if [[ -f "$CONFIG_FILE" ]]; then
  if [[ "$(<"$CONFIG_FILE")" != "$CONFIG_CONTENT" ]]; then
    echo "Resume configuration differs from $CONFIG_FILE" >&2
    exit 1
  fi
  if [[ "$RESUME" != "true" ]]; then
    echo "Seed-extension run already exists; set RESUME=true: $RUN_ROOT" >&2
    exit 1
  fi
else
  printf '%s\n' "$CONFIG_CONTENT" > "$CONFIG_FILE.tmp"
  mv "$CONFIG_FILE.tmp" "$CONFIG_FILE"
fi

echo "==> Generating correct and shift-1 source images for seeds: $GENERATION_SEEDS"
DATA_ROOT="$DATA_ROOT" \
SOURCE_RUN_ROOT="$SOURCE_RUN_ROOT" \
BASE_MODEL="$BASE_MODEL" \
PROTOTYPE_PATH="$PROTOTYPE_PATH" \
DCS_PATH="$DCS_PATH" \
RUN_ID="${EXTENSION_ID}_sources_shift1" \
RUN_ROOT="$SOURCE_BASE_ROOT" \
GENERATION_SEEDS="$GENERATION_SEEDS" \
CONDITIONS="prototype_dcs prototype_dcs_shuffled" \
SHUFFLE_SHIFT=1 \
RESUME="$RESUME" \
EVALUATE=false \
SUMMARIZE=false \
bash "$EXPERIMENT_DIR/run_experiment.sh"

shuffle_run_args=()
for shift in $SHUFFLE_SHIFTS; do
  if (( shift < 1 || shift > 9 )); then
    echo "SHUFFLE_SHIFTS must be integers in [1, 9], got $shift" >&2
    exit 1
  fi
  if (( shift == 1 )); then
    continue
  fi
  shift_root="$SOURCE_SHUFFLE_ROOT/shift_$shift"
  echo "==> Generating shift-$shift shuffled source images"
  DATA_ROOT="$DATA_ROOT" \
  SOURCE_RUN_ROOT="$SOURCE_RUN_ROOT" \
  BASE_MODEL="$BASE_MODEL" \
  PROTOTYPE_PATH="$PROTOTYPE_PATH" \
  DCS_PATH="$DCS_PATH" \
  RUN_ID="${EXTENSION_ID}_sources_shift${shift}" \
  RUN_ROOT="$shift_root" \
  GENERATION_SEEDS="$GENERATION_SEEDS" \
  CONDITIONS="prototype_dcs_shuffled" \
  SHUFFLE_SHIFT="$shift" \
  RESUME="$RESUME" \
  EVALUATE=false \
  SUMMARIZE=false \
  bash "$EXPERIMENT_DIR/run_experiment.sh"
  shuffle_run_args+=(--shuffle-run "$shift=$shift_root")
done

resume_args=()
if [[ "$RESUME" == "true" ]]; then
  resume_args=(--resume)
fi
read -r -a generation_seed_args <<< "$GENERATION_SEEDS"
python "$EXPERIMENT_DIR/build_selective_shuffle.py" \
  --base-run-root "$SOURCE_BASE_ROOT" \
  "${shuffle_run_args[@]}" \
  --cluster-summary "$CLUSTER_SUMMARY" \
  --output-root "$RUN_ROOT" \
  --generation-seeds "${generation_seed_args[@]}" \
  --selected-count "$SELECTED_COUNT" \
  --random-target-seed "$RANDOM_TARGET_SEED" \
  --link-mode "$LINK_MODE" \
  "${resume_args[@]}"

evaluate_condition() {
  local synthetic_dir="$1"
  local log_path="$2"
  local per_class_path="$3"
  local tag="$4"
  mkdir -p "$(dirname "$log_path")"
  if grep -q "Best, last acc" "$log_path" 2>/dev/null && [[ -f "$per_class_path" ]]; then
    echo "==> Reusing completed classifier: $tag"
    return
  fi
  if [[ -s "$log_path" ]]; then
    if [[ "$RESUME" == "true" ]]; then
      mv "$log_path" "$log_path.interrupted_$(date -u +%Y%m%dT%H%M%SZ)"
    else
      echo "Incomplete evaluation log exists: $log_path" >&2
      exit 1
    fi
  fi
  echo "==> Evaluating $tag"
  (
    cd "$EVALUATION_DIR"
    python train.py \
      -d imagenet \
      --imagenet_dir "$synthetic_dir" "$DATA_ROOT" \
      -n resnet_ap \
      --nclass 10 \
      --norm_type instance \
      --ipc 10 \
      --tag "$tag" \
      --slct_type random \
      --repeat 3 \
      --spec nette \
      --seed "$CLASSIFIER_SEED" \
      --per_class_output "$per_class_path"
  ) 2>&1 | tee "$log_path"
}

for generation_seed in $GENERATION_SEEDS; do
  reference_dir="$RUN_ROOT/reference_evaluation/seed_$generation_seed"
  evaluate_condition \
    "$SOURCE_BASE_ROOT/synthetic/seed_$generation_seed/prototype_dcs" \
    "$reference_dir/correct.log" \
    "$reference_dir/correct.per_class.json" \
    "selective_${EXTENSION_ID}_g${generation_seed}_correct"
  for shift in $SHUFFLE_SHIFTS; do
    for condition in small3_shuffled random3_shuffled; do
      evaluation_dir="$RUN_ROOT/shift_$shift/evaluation/seed_$generation_seed"
      evaluate_condition \
        "$RUN_ROOT/shift_$shift/synthetic/seed_$generation_seed/$condition" \
        "$evaluation_dir/$condition.log" \
        "$evaluation_dir/$condition.per_class.json" \
        "selective_${EXTENSION_ID}_s${shift}_g${generation_seed}_${condition}"
    done
  done
done

python "$EXPERIMENT_DIR/summarize_selective_shuffle.py" \
  --base-run-root "$SOURCE_BASE_ROOT" \
  "${shuffle_run_args[@]}" \
  --hybrid-run-root "$RUN_ROOT" \
  --correct-evaluation-root "$RUN_ROOT/reference_evaluation" \
  --conditions correct small3_shuffled random3_shuffled \
  --generation-seeds "${generation_seed_args[@]}" \
  --output-dir "$RUN_ROOT/summary"

echo "Selective seed-extension experiment complete: $RUN_ROOT"
