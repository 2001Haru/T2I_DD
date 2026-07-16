#!/usr/bin/env bash
set -euo pipefail

FINAL_CONTROL_RUN_ID="${FINAL_CONTROL_RUN_ID:-final_prompt_controls_v0}"
PCS_RUN_ID="${PCS_RUN_ID:-pcs_v0}"

ACCURACY_CSV="./trained_results/final_prompt_controls/${FINAL_CONTROL_RUN_ID}/summary/per_class_comparison.csv"
RUN_DIR="./results/pcs_diagnostics/${PCS_RUN_ID}"
OUTPUT_DIR="${RUN_DIR}/analysis_log"

for required in \
    "$ACCURACY_CSV" \
    "${RUN_DIR}/imageA/pcs_per_class.csv" \
    "${RUN_DIR}/imageA/pcs_raw.csv" \
    "${RUN_DIR}/imageB/pcs_per_class.csv" \
    "${RUN_DIR}/imageB/pcs_raw.csv"; do
    if [[ ! -f "$required" ]]; then
        echo "Required PCS artifact was not found: ${required}" >&2
        exit 1
    fi
done

if [[ -e "$OUTPUT_DIR" ]]; then
    echo "Refusing to overwrite logarithmic PCS analysis: ${OUTPUT_DIR}" >&2
    exit 1
fi

python analyze_prior_compatibility.py \
    --accuracy_csv "$ACCURACY_CSV" \
    --pcs imageA "${RUN_DIR}/imageA/pcs_per_class.csv" \
    --pcs imageB "${RUN_DIR}/imageB/pcs_per_class.csv" \
    --output_dir "$OUTPUT_DIR"

echo "Logarithmic PCS analysis completed: ${OUTPUT_DIR}"
