#!/usr/bin/env bash
set -euo pipefail
shopt -s inherit_errexit

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXPERIMENT_DIR="$REPO_ROOT/experiments/visual_text_factorial"
EVALUATION_DIR="$REPO_ROOT/04_evaluation/Minimax"

: "${DATA_ROOT:?Set DATA_ROOT to the prepared ImageNette root containing train/ and val/}"

BASE_SOURCE_ROOT="${BASE_SOURCE_ROOT:-$REPO_ROOT/../vlcp_factorial_runs/visual_text_factorial_v0}"
BASE_SHUFFLE_ROOT="${BASE_SHUFFLE_ROOT:-$REPO_ROOT/../vlcp_shuffle_runs/visual_text_shuffle_randomization_v0}"
EXTENSION_SOURCE_ROOT="${EXTENSION_SOURCE_ROOT:-$REPO_ROOT/../vlcp_factorial_runs/selective_small_cluster_shuffle_seed23_v0_sources_shift1}"
EXTENSION_SHUFFLE_ROOT="${EXTENSION_SHUFFLE_ROOT:-$REPO_ROOT/../vlcp_shuffle_runs/selective_small_cluster_shuffle_seed23_v0_sources}"
SMALL_BASE_RUN_ROOT="${SMALL_BASE_RUN_ROOT:-$REPO_ROOT/../vlcp_selective_shuffle_runs/selective_small_cluster_shuffle_v0}"
SMALL_EXTENSION_RUN_ROOT="${SMALL_EXTENSION_RUN_ROOT:-$REPO_ROOT/../vlcp_selective_shuffle_runs/selective_small_cluster_shuffle_seed23_v0}"
CLUSTER_SUMMARY="${CLUSTER_SUMMARY:-$BASE_SHUFFLE_ROOT/diagnostics/semantic_coverage_v0/cluster_member_audit/cluster_distance_summary.csv}"
CONTROL_RUN_ID="${CONTROL_RUN_ID:-selective_random_mask_controls_v0}"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/../vlcp_selective_shuffle_runs/$CONTROL_RUN_ID}"
RANDOM_TARGET_SEEDS="${RANDOM_TARGET_SEEDS:-20260801 20260802 20260803}"
BASE_GENERATION_SEEDS="${BASE_GENERATION_SEEDS:-0 1}"
EXTENSION_GENERATION_SEEDS="${EXTENSION_GENERATION_SEEDS:-2 3}"
SHUFFLE_SHIFTS="${SHUFFLE_SHIFTS:-1 2 4 7}"
EXISTING_RANDOM_TARGET_SEED="${EXISTING_RANDOM_TARGET_SEED:-20260731}"
SELECTED_COUNT="${SELECTED_COUNT:-3}"
LINK_MODE="${LINK_MODE:-symlink}"
CLASSIFIER_SEED="${CLASSIFIER_SEED:-0}"
RESUME="${RESUME:-false}"

for required in \
  "$DATA_ROOT" \
  "$BASE_SOURCE_ROOT" \
  "$BASE_SHUFFLE_ROOT" \
  "$EXTENSION_SOURCE_ROOT" \
  "$EXTENSION_SHUFFLE_ROOT" \
  "$SMALL_BASE_RUN_ROOT" \
  "$SMALL_EXTENSION_RUN_ROOT" \
  "$CLUSTER_SUMMARY"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required path: $required" >&2
    exit 1
  fi
done

mkdir -p "$RUN_ROOT"
CONFIG_FILE="$RUN_ROOT/random_mask_config.txt"
CONFIG_CONTENT="DATA_ROOT=$(realpath "$DATA_ROOT")
BASE_SOURCE_ROOT=$(realpath "$BASE_SOURCE_ROOT")
BASE_SHUFFLE_ROOT=$(realpath "$BASE_SHUFFLE_ROOT")
EXTENSION_SOURCE_ROOT=$(realpath "$EXTENSION_SOURCE_ROOT")
EXTENSION_SHUFFLE_ROOT=$(realpath "$EXTENSION_SHUFFLE_ROOT")
SMALL_BASE_RUN_ROOT=$(realpath "$SMALL_BASE_RUN_ROOT")
SMALL_EXTENSION_RUN_ROOT=$(realpath "$SMALL_EXTENSION_RUN_ROOT")
CLUSTER_SUMMARY=$(realpath "$CLUSTER_SUMMARY")
RANDOM_TARGET_SEEDS=$RANDOM_TARGET_SEEDS
BASE_GENERATION_SEEDS=$BASE_GENERATION_SEEDS
EXTENSION_GENERATION_SEEDS=$EXTENSION_GENERATION_SEEDS
SHUFFLE_SHIFTS=$SHUFFLE_SHIFTS
EXISTING_RANDOM_TARGET_SEED=$EXISTING_RANDOM_TARGET_SEED
SELECTED_COUNT=$SELECTED_COUNT
LINK_MODE=$LINK_MODE
CLASSIFIER_SEED=$CLASSIFIER_SEED"
if [[ -f "$CONFIG_FILE" ]]; then
  if [[ "$(<"$CONFIG_FILE")" != "$CONFIG_CONTENT" ]]; then
    echo "Resume configuration differs from $CONFIG_FILE" >&2
    exit 1
  fi
  if [[ "$RESUME" != "true" ]]; then
    echo "Random-mask control run already exists; set RESUME=true: $RUN_ROOT" >&2
    exit 1
  fi
else
  printf '%s\n' "$CONFIG_CONTENT" > "$CONFIG_FILE.tmp"
  mv "$CONFIG_FILE.tmp" "$CONFIG_FILE"
fi

base_shuffle_args=()
extension_shuffle_args=()
for shift in $SHUFFLE_SHIFTS; do
  if (( shift < 1 || shift > 9 )); then
    echo "SHUFFLE_SHIFTS must be integers in [1, 9], got $shift" >&2
    exit 1
  fi
  if (( shift > 1 )); then
    base_shuffle_args+=(--shuffle-run "$shift=$BASE_SHUFFLE_ROOT/shift_$shift")
    extension_shuffle_args+=(--shuffle-run "$shift=$EXTENSION_SHUFFLE_ROOT/shift_$shift")
  fi
done

resume_args=()
if [[ "$RESUME" == "true" ]]; then
  resume_args=(--resume)
fi
read -r -a base_generation_seed_args <<< "$BASE_GENERATION_SEEDS"
read -r -a extension_generation_seed_args <<< "$EXTENSION_GENERATION_SEEDS"
read -r -a random_target_seed_args <<< "$RANDOM_TARGET_SEEDS"

for random_target_seed in "${random_target_seed_args[@]}"; do
  if [[ "$random_target_seed" == "$EXISTING_RANDOM_TARGET_SEED" ]]; then
    echo "New random target seed duplicates the existing mask: $random_target_seed" >&2
    exit 1
  fi
  mask_root="$RUN_ROOT/mask_$random_target_seed"
  echo "==> Building random mask $random_target_seed for generation seeds $BASE_GENERATION_SEEDS"
  python "$EXPERIMENT_DIR/build_selective_shuffle.py" \
    --base-run-root "$BASE_SOURCE_ROOT" \
    "${base_shuffle_args[@]}" \
    --cluster-summary "$CLUSTER_SUMMARY" \
    --output-root "$mask_root" \
    --generation-seeds "${base_generation_seed_args[@]}" \
    --conditions random3_shuffled \
    --selected-count "$SELECTED_COUNT" \
    --random-target-seed "$random_target_seed" \
    --link-mode "$LINK_MODE" \
    "${resume_args[@]}"

  echo "==> Extending random mask $random_target_seed to generation seeds $EXTENSION_GENERATION_SEEDS"
  python "$EXPERIMENT_DIR/build_selective_shuffle.py" \
    --base-run-root "$EXTENSION_SOURCE_ROOT" \
    "${extension_shuffle_args[@]}" \
    --cluster-summary "$CLUSTER_SUMMARY" \
    --output-root "$mask_root" \
    --generation-seeds "${extension_generation_seed_args[@]}" \
    --conditions random3_shuffled \
    --selected-count "$SELECTED_COUNT" \
    --random-target-seed "$random_target_seed" \
    --link-mode "$LINK_MODE" \
    "${resume_args[@]}"
done

evaluate_random_mask() {
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

for random_target_seed in "${random_target_seed_args[@]}"; do
  for shift in $SHUFFLE_SHIFTS; do
    for generation_seed in $BASE_GENERATION_SEEDS $EXTENSION_GENERATION_SEEDS; do
      evaluation_dir="$RUN_ROOT/mask_$random_target_seed/shift_$shift/evaluation/seed_$generation_seed"
      evaluate_random_mask \
        "$RUN_ROOT/mask_$random_target_seed/shift_$shift/synthetic/seed_$generation_seed/random3_shuffled" \
        "$evaluation_dir/random3_shuffled.log" \
        "$evaluation_dir/random3_shuffled.per_class.json" \
        "random_mask_${CONTROL_RUN_ID}_m${random_target_seed}_s${shift}_g${generation_seed}"
    done
  done
done

python "$EXPERIMENT_DIR/summarize_random_mask_controls.py" \
  --small-base-run-root "$SMALL_BASE_RUN_ROOT" \
  --small-extension-run-root "$SMALL_EXTENSION_RUN_ROOT" \
  --control-run-root "$RUN_ROOT" \
  --existing-mask-seed "$EXISTING_RANDOM_TARGET_SEED" \
  --new-mask-seeds "${random_target_seed_args[@]}" \
  --base-generation-seeds "${base_generation_seed_args[@]}" \
  --extension-generation-seeds "${extension_generation_seed_args[@]}" \
  --shuffle-shifts $SHUFFLE_SHIFTS \
  --output-dir "$RUN_ROOT/summary"

echo "Random-mask controls complete: $RUN_ROOT"
