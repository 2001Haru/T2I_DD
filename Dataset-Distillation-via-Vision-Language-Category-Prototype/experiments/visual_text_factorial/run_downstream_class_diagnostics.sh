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
DIAGNOSTICS_ROOT="${DIAGNOSTICS_ROOT:-$RANDOMIZATION_ROOT/diagnostics/$DIAGNOSTICS_ID}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$DIAGNOSTICS_ROOT/downstream_per_class}"
GENERATION_SEEDS="${GENERATION_SEEDS:-0 1}"
SHUFFLE_SHIFTS="${SHUFFLE_SHIFTS:-1 2 4 7}"
VISUAL_MODES="${VISUAL_MODES:-prototype}"
CLASSIFIER_SEED="${CLASSIFIER_SEED:-0}"
RESUME="${RESUME:-false}"

DIAGNOSTIC_CSV="$DIAGNOSTICS_ROOT/summary/conditioning_and_coverage_per_class.csv"
if [[ ! -f "$DIAGNOSTIC_CSV" ]]; then
  echo "Missing semantic coverage diagnostics: $DIAGNOSTIC_CSV" >&2
  exit 1
fi

mkdir -p "$OUTPUT_ROOT/results" "$OUTPUT_ROOT/logs"
CONFIG_FILE="$OUTPUT_ROOT/downstream_config.txt"
CONFIG_CONTENT="DATA_ROOT=$(realpath "$DATA_ROOT")
BASE_RUN_ROOT=$(realpath "$BASE_RUN_ROOT")
RANDOMIZATION_ROOT=$(realpath "$RANDOMIZATION_ROOT")
DIAGNOSTIC_CSV=$(realpath "$DIAGNOSTIC_CSV")
GENERATION_SEEDS=$GENERATION_SEEDS
SHUFFLE_SHIFTS=$SHUFFLE_SHIFTS
VISUAL_MODES=$VISUAL_MODES
CLASSIFIER_SEED=$CLASSIFIER_SEED"
if [[ -f "$CONFIG_FILE" ]]; then
  if [[ "$(<"$CONFIG_FILE")" != "$CONFIG_CONTENT" ]]; then
    echo "Resume configuration differs from $CONFIG_FILE" >&2
    exit 1
  fi
  if [[ "$RESUME" != "true" ]]; then
    echo "Downstream diagnostic run exists; set RESUME=true: $OUTPUT_ROOT" >&2
    exit 1
  fi
else
  printf '%s\n' "$CONFIG_CONTENT" > "$CONFIG_FILE.tmp"
  mv "$CONFIG_FILE.tmp" "$CONFIG_FILE"
fi

run_condition() {
  local generation_seed="$1"
  local visual_mode="$2"
  local condition_key="$3"
  local synthetic_dir="$4"
  local result_dir="$OUTPUT_ROOT/results/seed_$generation_seed"
  local log_dir="$OUTPUT_ROOT/logs/seed_$generation_seed"
  local result_path="$result_dir/${visual_mode}_${condition_key}.json"
  local log_path="$log_dir/${visual_mode}_${condition_key}.log"

  mkdir -p "$result_dir" "$log_dir"
  if [[ ! -f "$synthetic_dir/complete.json" ]]; then
    echo "Incomplete synthetic condition: $synthetic_dir" >&2
    exit 1
  fi
  if [[ -f "$result_path" ]]; then
    echo "==> Reusing per-class result seed=$generation_seed $visual_mode/$condition_key"
    return
  fi
  if [[ -s "$log_path" ]]; then
    if [[ "$RESUME" == "true" ]]; then
      interrupted="$log_path.interrupted_$(date -u +%Y%m%dT%H%M%SZ)"
      mv "$log_path" "$interrupted"
      echo "==> Archived interrupted log: $interrupted"
    else
      echo "Incomplete per-class log exists: $log_path" >&2
      exit 1
    fi
  fi

  echo "==> Per-class evaluation seed=$generation_seed $visual_mode/$condition_key"
  (
    cd "$EVALUATION_DIR"
    python train.py \
      -d imagenet \
      --imagenet_dir "$synthetic_dir" "$DATA_ROOT" \
      -n resnet_ap \
      --nclass 10 \
      --norm_type instance \
      --ipc 10 \
      --tag "semantic_class_${DIAGNOSTICS_ID}_g${generation_seed}_${visual_mode}_${condition_key}" \
      --slct_type random \
      --repeat 3 \
      --spec nette \
      --seed "$CLASSIFIER_SEED" \
      --per_class_output "$result_path"
  ) 2>&1 | tee "$log_path"
}

for generation_seed in $GENERATION_SEEDS; do
  for visual_mode in $VISUAL_MODES; do
    run_condition \
      "$generation_seed" \
      "$visual_mode" \
      "correct" \
      "$BASE_RUN_ROOT/synthetic/seed_$generation_seed/${visual_mode}_dcs"
    for shift in $SHUFFLE_SHIFTS; do
      if [[ "$shift" == "1" ]]; then
        shift_root="$BASE_RUN_ROOT"
      else
        shift_root="$RANDOMIZATION_ROOT/shift_$shift"
      fi
      run_condition \
        "$generation_seed" \
        "$visual_mode" \
        "shift$shift" \
        "$shift_root/synthetic/seed_$generation_seed/${visual_mode}_dcs_shuffled"
    done
  done
done

read -r -a generation_seed_args <<< "$GENERATION_SEEDS"
read -r -a shuffle_shift_args <<< "$SHUFFLE_SHIFTS"
read -r -a visual_mode_args <<< "$VISUAL_MODES"
python "$EXPERIMENT_DIR/analyze_downstream_per_class.py" \
  --results-root "$OUTPUT_ROOT/results" \
  --diagnostic-csv "$DIAGNOSTIC_CSV" \
  --output-dir "$OUTPUT_ROOT/summary" \
  --visual-modes "${visual_mode_args[@]}" \
  --generation-seeds "${generation_seed_args[@]}" \
  --shuffle-shifts "${shuffle_shift_args[@]}"

echo "Downstream per-class diagnostics complete: $OUTPUT_ROOT"
