#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DATASET="${DATASET:-formality}"
MODEL="${MODEL:-/data/nydeng/model_download/Qwen_Qwen2.5-3B-Instruct}"
LOSS="lossv1"
TEACHER="teacherv1"
MODE="infer"
DATE_TAG="${DATE_TAG:-$(date +%Y%m%d_%H%M%S)}"
MODEL_TAG="${MODEL//\//_}"
LOG_MODEL_TAG="${LOG_MODEL_TAG:-Qwen2.5-3B-Instruct}"
PROCESSED_DIR="${PROCESSED_DIR:-data/processed/${DATASET}}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/${MODE}/${DATE_TAG}_${MODE}_${LOSS}_${DATASET}_${TEACHER}_${MODEL_TAG}}"
LOG_DIR="${LOG_DIR:-logs}"
LOG_FILE="${LOG_DIR}/${DATE_TAG}_${MODE}_${DATASET}_${LOG_MODEL_TAG}.log"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

echo "Writing infer log to ${LOG_FILE}"

{
  echo "mode=${MODE}"
  echo "model=${MODEL}"
  echo "output_dir=${OUTPUT_DIR}"

  PYTHONPATH=src python -m preprocess \
    --dataset_name "$DATASET" \
    --output_dir "$PROCESSED_DIR" \
    --tolerance 10

  PYTHONPATH=src python -m infer \
    --model_path "$MODEL" \
    --dataset_name "$DATASET" \
    --loss_version "$LOSS" \
    --teacher_version "$TEACHER" \
    --test_files "${PROCESSED_DIR}/test.jsonl" \
    --output_dir "$OUTPUT_DIR" \
    --max_length 1024 \
    --max_new_tokens 512 \
    --tolerance 10 \
    --ece_bins 10
} > "$LOG_FILE" 2>&1 &
