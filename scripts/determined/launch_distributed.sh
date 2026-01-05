#!/bin/bash
# Distributed training launcher for Determined AI
# This script handles the distributed setup using accelerate + FSDP
#
# Hyperparameters are read from the experiment config and passed
# as environment variables by Determined AI.

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

echo "=== Determined AI Distributed Training Launcher ==="
echo "Repo directory: $REPO_DIR"

# Determine number of GPUs from Determined environment
# DET_SLOT_IDS contains comma-separated GPU IDs, or use slots_per_trial
if [ -n "${DET_SLOT_IDS:-}" ]; then
    NUM_GPUS=$(echo "$DET_SLOT_IDS" | tr ',' '\n' | wc -l)
elif [ -n "${DET_SLOTS_PER_TRIAL:-}" ]; then
    NUM_GPUS=$DET_SLOTS_PER_TRIAL
else
    # Fallback: count available GPUs
    NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l || echo 1)
fi

echo "Number of GPUs: $NUM_GPUS"
echo "Determined experiment ID: ${DET_EXPERIMENT_ID:-N/A}"
echo "Determined trial ID: ${DET_TRIAL_ID:-N/A}"

# Generate accelerate config dynamically based on GPU count
ACCELERATE_CONFIG="/tmp/accelerate_fsdp_det.yaml"

cat > "$ACCELERATE_CONFIG" << EOF
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: FSDP
downcast_bf16: 'no'
enable_cpu_affinity: false
fsdp_config:
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_backward_prefetch: BACKWARD_PRE
  fsdp_cpu_ram_efficient_loading: true
  fsdp_forward_prefetch: false
  fsdp_offload_params: false
  fsdp_sharding_strategy: FULL_SHARD
  fsdp_state_dict_type: SHARDED_STATE_DICT
  fsdp_sync_module_states: true
  fsdp_use_orig_params: true
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: $NUM_GPUS
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
EOF

echo "Generated accelerate config at $ACCELERATE_CONFIG"
cat "$ACCELERATE_CONFIG"

# Build training command arguments
TRAIN_SCRIPT="$REPO_DIR/scripts/sft_european.py"

# Read hyperparameters from Determined environment (set via experiment config)
MODEL="${model_name_or_path:-swiss-ai/Apertus-8B-Instruct-2509}"
OUTPUT_DIR="${output_dir:-$REPO_DIR/outputs/det-run}"
BATCH_SIZE="${per_device_train_batch_size:-2}"
GRAD_ACCUM="${gradient_accumulation_steps:-2}"
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
echo "Batch size per device: $BATCH_SIZE"
echo "Gradient accumulation: $GRAD_ACCUM"
echo "Learning rate: $LR"
echo "Epochs: $EPOCHS"
echo ""

# Build command
CMD="accelerate launch --config_file $ACCELERATE_CONFIG $TRAIN_SCRIPT"
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
