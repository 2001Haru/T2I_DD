#!/usr/bin/env bash
set -euo pipefail
shopt -s inherit_errexit

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXPERIMENT_DIR="$REPO_ROOT/experiments/visual_text_factorial"
EVALUATION_DIR="$REPO_ROOT/04_evaluation/Minimax"

: "${DATA_ROOT:?Set DATA_ROOT to the prepared ImageNette root containing train/ and val/}"

BASE_RUN_ROOT="${BASE_RUN_ROOT:-$REPO_ROOT/../vlcp_factorial_runs/visual_text_factorial_v0}"
RANDOMIZATION_RUN_ID="${RANDOMIZATION_RUN_ID:-visual_text_shuffle_randomization_v0}"
RANDOMIZATION_ROOT="${RANDOMIZATION_ROOT:-$REPO_ROOT/../vlcp_shuffle_runs/$RANDOMIZATION_RUN_ID}"
DIAGNOSTICS_ID="${DIAGNOSTICS_ID:-semantic_coverage_v0}"
CLUSTER_SUMMARY="${CLUSTER_SUMMARY:-$RANDOMIZATION_ROOT/diagnostics/$DIAGNOSTICS_ID/cluster_member_audit/cluster_distance_summary.csv}"
SOURCE_PER_CLASS_ROOT="${SOURCE_PER_CLASS_ROOT:-$RANDOMIZATION_ROOT/diagnostics/$DIAGNOSTICS_ID/downstream_per_class}"
SELECTIVE_RUN_ID="${SELECTIVE_RUN_ID:-selective_small_cluster_shuffle_v0}"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/../vlcp_selective_shuffle_runs/$SELECTIVE_RUN_ID}"
SHUFFLE_SHIFTS="${SHUFFLE_SHIFTS:-1 2 4 7}"
GENERATION_SEEDS="${GENERATION_SEEDS:-0 1}"
SELECTED_COUNT="${SELECTED_COUNT:-3}"
RANDOM_TARGET_SEED="${RANDOM_TARGET_SEED:-20260731}"
LINK_MODE="${LINK_MODE:-symlink}"
CLASSIFIER_SEED="${CLASSIFIER_SEED:-0}"
RESUME="${RESUME:-false}"

for required in \
  "$DATA_ROOT" \
  "$BASE_RUN_ROOT" \
  "$RANDOMIZATION_ROOT" \
  "$CLUSTER_SUMMARY"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required path: $required" >&2
    exit 1
  fi
done

mkdir -p "$RUN_ROOT"
CONFIG_FILE="$RUN_ROOT/selective_run_config.txt"
CONFIG_CONTENT="DATA_ROOT=$(realpath "$DATA_ROOT")
BASE_RUN_ROOT=$(realpath "$BASE_RUN_ROOT")
RANDOMIZATION_ROOT=$(realpath "$RANDOMIZATION_ROOT")
CLUSTER_SUMMARY=$(realpath "$CLUSTER_SUMMARY")
SOURCE_PER_CLASS_ROOT=$(realpath -m "$SOURCE_PER_CLASS_ROOT")
SHUFFLE_SHIFTS=$SHUFFLE_SHIFTS
GENERATION_SEEDS=$GENERATION_SEEDS
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
    echo "Selective-shuffle run already exists; set RESUME=true: $RUN_ROOT" >&2
    exit 1
  fi
else
  printf '%s\n' "$CONFIG_CONTENT" > "$CONFIG_FILE.tmp"
  mv "$CONFIG_FILE.tmp" "$CONFIG_FILE"
fi

shuffle_run_args=()
for shift in $SHUFFLE_SHIFTS; do
  if (( shift < 1 || shift > 9 )); then
    echo "SHUFFLE_SHIFTS must be integers in [1, 9], got $shift" >&2
    exit 1
  fi
  if (( shift > 1 )); then
    shift_root="$RANDOMIZATION_ROOT/shift_$shift"
    if [[ ! -d "$shift_root/synthetic" ]]; then
      echo "Missing shuffled run for shift $shift: $shift_root" >&2
      exit 1
    fi
    shuffle_run_args+=(--shuffle-run "$shift=$shift_root")
  fi
done

per_class_source_args=()
if [[ -d "$SOURCE_PER_CLASS_ROOT/results" ]]; then
  per_class_source_args=(--source-per-class-root "$SOURCE_PER_CLASS_ROOT")
else
  echo "Existing correct/all-shuffled per-class results not found; aggregate accuracy summary will still be produced: $SOURCE_PER_CLASS_ROOT"
fi

resume_args=()
if [[ "$RESUME" == "true" ]]; then
  resume_args=(--resume)
fi

read -r -a generation_seed_args <<< "$GENERATION_SEEDS"
python "$EXPERIMENT_DIR/build_selective_shuffle.py" \
  --base-run-root "$BASE_RUN_ROOT" \
  "${shuffle_run_args[@]}" \
  --cluster-summary "$CLUSTER_SUMMARY" \
  --output-root "$RUN_ROOT" \
  --generation-seeds "${generation_seed_args[@]}" \
  --selected-count "$SELECTED_COUNT" \
  --random-target-seed "$RANDOM_TARGET_SEED" \
  --link-mode "$LINK_MODE" \
  "${resume_args[@]}"

for shift in $SHUFFLE_SHIFTS; do
  for generation_seed in $GENERATION_SEEDS; do
    seed_eval_dir="$RUN_ROOT/shift_$shift/evaluation/seed_$generation_seed"
    mkdir -p "$seed_eval_dir"
    for condition in small3_shuffled random3_shuffled; do
      synthetic_dir="$RUN_ROOT/shift_$shift/synthetic/seed_$generation_seed/$condition"
      log_path="$seed_eval_dir/$condition.log"
      per_class_path="$seed_eval_dir/$condition.per_class.json"
      if [[ ! -f "$synthetic_dir/complete.json" ]]; then
        echo "Incomplete hybrid condition: $synthetic_dir" >&2
        exit 1
      fi
      if grep -q "Best, last acc" "$log_path" 2>/dev/null && [[ -f "$per_class_path" ]]; then
        echo "==> Reusing completed selective evaluation shift=$shift seed=$generation_seed condition=$condition"
        continue
      fi
      if [[ -s "$log_path" ]]; then
        if [[ "$RESUME" == "true" ]]; then
          interrupted_log="$log_path.interrupted_$(date -u +%Y%m%dT%H%M%SZ)"
          mv "$log_path" "$interrupted_log"
          echo "==> Archived interrupted evaluation log: $interrupted_log"
        else
          echo "Incomplete evaluation log exists: $log_path" >&2
          echo "Rerun with RESUME=true to archive and restart it." >&2
          exit 1
        fi
      fi
      echo "==> Evaluating shift=$shift seed=$generation_seed condition=$condition"
      (
        cd "$EVALUATION_DIR"
        python train.py \
          -d imagenet \
          --imagenet_dir "$synthetic_dir" "$DATA_ROOT" \
          -n resnet_ap \
          --nclass 10 \
          --norm_type instance \
          --ipc 10 \
          --tag "selective_${SELECTIVE_RUN_ID}_s${shift}_g${generation_seed}_${condition}" \
          --slct_type random \
          --repeat 3 \
          --spec nette \
          --seed "$CLASSIFIER_SEED" \
          --per_class_output "$per_class_path"
      ) 2>&1 | tee "$log_path"
    done
  done
done

python "$EXPERIMENT_DIR/summarize_selective_shuffle.py" \
  --base-run-root "$BASE_RUN_ROOT" \
  "${shuffle_run_args[@]}" \
  --hybrid-run-root "$RUN_ROOT" \
  "${per_class_source_args[@]}" \
  --generation-seeds "${generation_seed_args[@]}" \
  --output-dir "$RUN_ROOT/summary"

echo "Selective small-cluster shuffle experiment complete: $RUN_ROOT"
