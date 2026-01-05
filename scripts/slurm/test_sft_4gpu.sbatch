#!/bin/bash
#SBATCH --job-name=test-sft-eu-4g
#SBATCH --output=/mnt/vast/home/ct1002/logs/slurm/test_sft_eu_4g_%j.out
#SBATCH --error=/mnt/vast/home/ct1002/logs/slurm/test_sft_eu_4g_%j.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --gres=gpu:4
#SBATCH --mem=200G
#SBATCH --time=4:00:00

# Test run for European SFT training with TRL
# Uses 4 GPUs with FSDP for validation

set -euo pipefail

# Paths
TRL_EU_DIR="/mnt/vast/home/ct1002/repos/trl-eu"
DATA_PATH="${DATA_PATH:-/mnt/vast/home/ct1002/data/eu_reasoning/pretokenized/tokenized/dolci_7b_sft/de}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/vast/home/ct1002/outputs/test_sft_eu_4g_$(date +%Y%m%d_%H%M%S)}"
MODEL="${MODEL:-/mnt/vast/home/ct1002/models/DeepSeek-R1-Distill-Qwen-1.5B}"

echo "=============================================="
echo "European SFT Test Run (4 GPU)"
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Model: ${MODEL}"
echo "Data: ${DATA_PATH}"
echo "Output: ${OUTPUT_DIR}"
echo "GPUs: 4"
echo "=============================================="

# Create output directory
mkdir -p "${OUTPUT_DIR}"
mkdir -p /mnt/vast/home/ct1002/logs/slurm

# Load conda/env
source /mnt/vast/home/ct1002/.bashrc
conda activate base

# Set up environment
export HF_HOME=/mnt/vast/home/ct1002/.cache/huggingface
export TRANSFORMERS_CACHE=/mnt/vast/home/ct1002/.cache/huggingface/transformers
export HF_DATASETS_CACHE=/mnt/vast/home/ct1002/.cache/huggingface/datasets

# Add TRL to path
export PYTHONPATH="${TRL_EU_DIR}/trl:${PYTHONPATH:-}"

# WandB setup
export WANDB_PROJECT="eu-sft"
export WANDB_RUN_NAME="test-sft-4g-${SLURM_JOB_ID}"

cd "${TRL_EU_DIR}"

echo ""
echo "Starting training with accelerate + FSDP (4 GPUs)..."
echo ""

# Create 4-GPU accelerate config on the fly
cat > /tmp/accelerate_fsdp_4gpu_${SLURM_JOB_ID}.yaml << EOF
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
num_processes: 4
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
EOF

# Run training with FSDP
accelerate launch \
    --config_file /tmp/accelerate_fsdp_4gpu_${SLURM_JOB_ID}.yaml \
    scripts/sft_european.py \
    --model_name_or_path "${MODEL}" \
    --pretokenized_path "${DATA_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --learning_rate 2e-6 \
    --num_train_epochs 1 \
    --warmup_ratio 0.03 \
    --logging_steps 10 \
    --save_steps 500 \
    --gradient_checkpointing \
    --bf16 \
    --report_to wandb \
    --seed 42

# Cleanup
rm -f /tmp/accelerate_fsdp_4gpu_${SLURM_JOB_ID}.yaml

echo ""
echo "=============================================="
echo "Training complete!"
echo "Output saved to: ${OUTPUT_DIR}"
echo "=============================================="
