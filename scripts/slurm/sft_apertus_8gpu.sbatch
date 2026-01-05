#!/bin/bash
#SBATCH --job-name=sft-apertus-eu
#SBATCH --output=/mnt/vast/home/ct1002/logs/slurm/sft_apertus_eu_%j.out
#SBATCH --error=/mnt/vast/home/ct1002/logs/slurm/sft_apertus_eu_%j.err
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=112
#SBATCH --gres=gpu:8
#SBATCH --mem=0
#SBATCH --time=24:00:00

# Production European SFT training with TRL + FSDP
# Model: swiss-ai/Apertus-8B-Instruct-2509
# Uses 8 GPUs with FSDP sharding

set -euo pipefail

# Paths
TRL_EU_DIR="/mnt/vast/home/ct1002/repos/trl-eu"
DATA_PATH="${DATA_PATH:-/mnt/vast/home/ct1002/data/eu_reasoning/pretokenized/tokenized/dolci_7b_sft/de}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/vast/home/ct1002/outputs/sft_apertus_eu_$(date +%Y%m%d_%H%M%S)}"
MODEL="${MODEL:-swiss-ai/Apertus-8B-Instruct-2509}"
LANG="${LANG:-de}"

echo "=============================================="
echo "European SFT Production Run"
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Model: ${MODEL}"
echo "Language: ${LANG}"
echo "Data: ${DATA_PATH}"
echo "Output: ${OUTPUT_DIR}"
echo "GPUs: 8"
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
export WANDB_RUN_NAME="sft-apertus-${LANG}-${SLURM_JOB_ID}"

cd "${TRL_EU_DIR}"

echo ""
echo "Starting training with accelerate + FSDP..."
echo ""

# Run training with FSDP
accelerate launch \
    --config_file configs/accelerate_fsdp_8gpu.yaml \
    scripts/sft_european.py \
    --model_name_or_path "${MODEL}" \
    --pretokenized_path "${DATA_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --per_device_train_batch_size 1 \
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

echo ""
echo "=============================================="
echo "Training complete!"
echo "Output saved to: ${OUTPUT_DIR}"
echo "=============================================="
