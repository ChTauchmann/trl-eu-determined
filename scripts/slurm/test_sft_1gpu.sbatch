#!/bin/bash
#SBATCH --job-name=test-sft-eu
#SBATCH --output=/mnt/vast/home/ct1002/logs/slurm/test_sft_eu_%j.out
#SBATCH --error=/mnt/vast/home/ct1002/logs/slurm/test_sft_eu_%j.err
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=2:00:00

# Test run for European SFT training with TRL
# Uses 1 GPU for quick validation
# Runs inside Apptainer container for dependency compatibility

set -eo pipefail

# Paths
CONTAINER="/mnt/vast/home/ct1002/containers/star-grpo-revise-fixed.sif"
TRL_EU_DIR="/mnt/vast/home/ct1002/repos/trl-eu"
DATA_PATH="/mnt/vast/home/ct1002/data/eu_reasoning/pretokenized/tokenized/dolci_7b_sft/de"
OUTPUT_DIR="/mnt/vast/home/ct1002/outputs/test_sft_eu_$(date +%Y%m%d_%H%M%S)"
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B}"

echo "=============================================="
echo "European SFT Test Run (Container)"
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Container: ${CONTAINER}"
echo "Model: ${MODEL}"
echo "Data: ${DATA_PATH}"
echo "Output: ${OUTPUT_DIR}"
echo "=============================================="

# Create output directory
mkdir -p "${OUTPUT_DIR}"
mkdir -p /mnt/vast/home/ct1002/logs/slurm

echo ""
echo "Starting training inside container..."
echo ""

# Run training inside container
apptainer exec --nv \
    --bind /mnt/vast/home/ct1002:/mnt/vast/home/ct1002 \
    "${CONTAINER}" \
    bash -c "
        export PYTHONNOUSERSITE=1
        export CUDA_VISIBLE_DEVICES=0
        export HF_HOME=/mnt/vast/home/ct1002/.cache/huggingface
        export HF_DATASETS_CACHE=/mnt/vast/home/ct1002/.cache/huggingface/datasets
        export WANDB_PROJECT=eu-sft
        export WANDB_RUN_NAME=test-sft-${SLURM_JOB_ID}

        python ${TRL_EU_DIR}/scripts/sft_european.py \
            --model_name_or_path ${MODEL} \
            --pretokenized_path ${DATA_PATH} \
            --output_dir ${OUTPUT_DIR} \
            --per_device_train_batch_size 1 \
            --gradient_accumulation_steps 4 \
            --learning_rate 2e-6 \
            --num_train_epochs 1 \
            --warmup_ratio 0.03 \
            --logging_steps 10 \
            --save_steps 500 \
            --gradient_checkpointing \
            --bf16 \
            --report_to wandb \
            --seed 42
    "

echo ""
echo "=============================================="
echo "Training complete!"
echo "Output saved to: ${OUTPUT_DIR}"
echo "=============================================="
