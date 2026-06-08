#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DATASET="${DATASET:-formality}"
MODEL="${MODEL:-/data/nydeng/model_download/Qwen_Qwen2.5-3B-Instruct}"
STUDENT_DEVICE="${STUDENT_DEVICE:-auto}"
TEACHER_DEVICE="${TEACHER_DEVICE:-auto}"
LOSS="lossv1"
TEACHER="teacherv1"
DATE_TAG="${DATE_TAG:-$(date +%Y%m%d_%H%M%S)}"
MODEL_TAG="${MODEL##*/}"
PROCESSED_DIR="${PROCESSED_DIR:-data/processed/${DATASET}}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/train/${DATE_TAG}_${LOSS}_${DATASET}_${TEACHER}_${MODEL_TAG}}"
LOG_DIR="${LOG_DIR:-logs}"
LOG_FILE="${LOG_DIR}/${DATE_TAG}_train_${LOSS}_${DATASET}_${TEACHER}_${MODEL_TAG}.log"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

echo "Writing train log to ${LOG_FILE}"

{
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

  PYTHONPATH=src python -m preprocess \
    --dataset_name "$DATASET" \
    --output_dir "$PROCESSED_DIR" \
    --tolerance 10

  PYTHONPATH=src python -m train_opd \
    --model_name "$MODEL" \
    --dataset_name "$DATASET" \
    --loss_version "$LOSS" \
    --teacher_version "$TEACHER" \
    --train_files "${PROCESSED_DIR}/train.jsonl" \
    --test_files "${PROCESSED_DIR}/test.jsonl" \
    --output_dir "$OUTPUT_DIR" \
    --max_length 1024 \
    --max_new_tokens 512 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 5e-6 \
    --lambda_kl 1.0 \
    --tolerance 10 \
    --ece_bins 10 \
    --device "$STUDENT_DEVICE" \
    --teacher_device "$TEACHER_DEVICE"
} > "$LOG_FILE" 2>&1 &
