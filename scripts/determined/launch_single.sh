#!/bin/bash
# Single-GPU training launcher for Determined AI
# This script handles single-GPU setup without accelerate

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

echo "=== Determined AI Single-GPU Training Launcher ==="
echo "Repo directory: $REPO_DIR"

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Training script
TRAIN_SCRIPT="$REPO_DIR/scripts/sft_european.py"

# Read hyperparameters from Determined environment
MODEL="${model_name_or_path:-Qwen/Qwen2.5-0.5B}"
OUTPUT_DIR="${output_dir:-$REPO_DIR/outputs/det-run}"
BATCH_SIZE="${per_device_train_batch_size:-1}"
GRAD_ACCUM="${gradient_accumulation_steps:-4}"
LR="${learning_rate:-2e-6}"
EPOCHS="${num_train_epochs:-1}"
WARMUP="${warmup_ratio:-0.03}"
MAX_SEQ_LEN="${max_seq_length:-4096}"
LOG_STEPS="${logging_steps:-10}"
SAVE_STEPS="${save_steps:-500}"
REPORT_TO="${report_to:-wandb}"
SEED="${seed:-42}"

# Data source
PRETOKENIZED="${pretokenized_path:-}"
DATASET="${dataset_path:-}"
MAX_SAMPLES="${max_samples:-}"

# Create output directory with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
FINAL_OUTPUT_DIR="${OUTPUT_DIR}/${TIMESTAMP}"
mkdir -p "$FINAL_OUTPUT_DIR"

echo ""
echo "=== Training Configuration ==="
echo "Model: $MODEL"
echo "Output: $FINAL_OUTPUT_DIR"
echo "Batch size: $BATCH_SIZE"
echo "Gradient accumulation: $GRAD_ACCUM"
echo "Learning rate: $LR"
echo ""

# Build command
CMD="python $TRAIN_SCRIPT"
CMD="$CMD --model_name_or_path $MODEL"
CMD="$CMD --output_dir $FINAL_OUTPUT_DIR"
CMD="$CMD --per_device_train_batch_size $BATCH_SIZE"
CMD="$CMD --gradient_accumulation_steps $GRAD_ACCUM"
CMD="$CMD --learning_rate $LR"
CMD="$CMD --num_train_epochs $EPOCHS"
CMD="$CMD --warmup_ratio $WARMUP"
CMD="$CMD --max_seq_length $MAX_SEQ_LEN"
CMD="$CMD --logging_steps $LOG_STEPS"
CMD="$CMD --save_steps $SAVE_STEPS"
CMD="$CMD --report_to $REPORT_TO"
CMD="$CMD --seed $SEED"
CMD="$CMD --gradient_checkpointing"
CMD="$CMD --bf16"

# Add data source
if [ -n "$PRETOKENIZED" ] && [ "$PRETOKENIZED" != "null" ]; then
    CMD="$CMD --pretokenized_path $PRETOKENIZED"
elif [ -n "$DATASET" ] && [ "$DATASET" != "null" ]; then
    CMD="$CMD --dataset_path $DATASET"
    if [ -n "$MAX_SAMPLES" ] && [ "$MAX_SAMPLES" != "null" ]; then
        CMD="$CMD --max_samples $MAX_SAMPLES"
    fi
else
    echo "ERROR: Either pretokenized_path or dataset_path must be set"
    exit 1
fi

# Optional flags
if [ "${use_liger_kernel:-false}" = "true" ]; then
    CMD="$CMD --use_liger_kernel"
fi

if [ "${packing:-false}" = "true" ]; then
    CMD="$CMD --packing"
fi

echo "=== Executing Command ==="
echo "$CMD"
echo ""

# Execute training
exec $CMD
