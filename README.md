# TRL European Language Models

Training European language models using HuggingFace TRL for SFT and beyond.

## Overview

This repository extends HuggingFace TRL for training European multilingual models. It supports:
- Pre-tokenized packed datasets (from open-instruct tokenization pipeline)
- Multi-GPU training with FSDP
- Memory optimizations (gradient checkpointing, Liger kernel)
- **Determined AI** for experiment management and distributed training
- SLURM for traditional cluster job scheduling

## Target Model

- **Base Model**: swiss-ai/Apertus-8B-Instruct-2509
- **Languages**: German (de), Spanish (es), French (fr), Italian (it)
- **Dataset**: Translated Dolci-Think-SFT

---

## Determined AI (Recommended)

### Quick Start

```bash
# 1-GPU test run
det experiment create configs/determined/experiment_1gpu.yaml .

# 4-GPU test run with FSDP
det experiment create configs/determined/experiment_4gpu.yaml .

# 8-GPU production training
det experiment create configs/determined/experiment_8gpu.yaml .
```

### Configuration

Edit the experiment YAML files in `configs/determined/` to customize:

```yaml
hyperparameters:
  # Model
  model_name_or_path: swiss-ai/Apertus-8B-Instruct-2509

  # Data - set one of these
  pretokenized_path: /path/to/pretokenized/de
  dataset_path: /path/to/translated/de.jsonl
  max_samples: null  # Set to limit samples for testing

  # Training
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 2
  learning_rate: 2.0e-6
  num_train_epochs: 1
```

### Available Configurations

| Config | GPUs | Model | Use Case |
|--------|------|-------|----------|
| `experiment_1gpu.yaml` | 1 | Qwen2.5-0.5B | Quick testing |
| `experiment_4gpu.yaml` | 4 | Qwen2.5-0.5B | Distributed testing |
| `experiment_8gpu.yaml` | 8 | Apertus-8B | Production training |

### Monitor Experiments

```bash
# List experiments
det experiment list

# View logs
det experiment logs <experiment_id>

# View experiment in web UI
# Navigate to: https://<det-master>:8080/det/experiments/<id>
```

---

## SLURM (Legacy)

### Test Run (1 GPU)
```bash
# Uses Qwen2.5-0.5B for quick validation
sbatch scripts/slurm/test_sft_1gpu.sbatch
```

### Production Run (8 GPU)
```bash
# Full Apertus-8B training with FSDP
export DATA_PATH=/path/to/pretokenized/de
export LANG=de
sbatch scripts/slurm/sft_apertus_8gpu.sbatch
```

---

## Manual Run

### Direct Python Execution
```bash
# With pre-tokenized data
python scripts/sft_european.py \
    --model_name_or_path swiss-ai/Apertus-8B-Instruct-2509 \
    --pretokenized_path /path/to/pretokenized/de \
    --output_dir ./outputs/apertus-8b-de

# With raw translated JSONL
python scripts/sft_european.py \
    --model_name_or_path Qwen/Qwen2.5-0.5B \
    --dataset_path /path/to/translated/de.jsonl \
    --max_samples 1000 \
    --output_dir ./outputs/test
```

### Multi-GPU with Accelerate
```bash
accelerate launch \
    --config_file configs/accelerate_fsdp_8gpu.yaml \
    scripts/sft_european.py \
    --model_name_or_path swiss-ai/Apertus-8B-Instruct-2509 \
    --pretokenized_path /path/to/pretokenized/de \
    --output_dir ./outputs/apertus-8b-de
```

---

## Data Format

### Pre-tokenized (Recommended)
Arrow format with columns:
- `input_ids`: Packed token sequences
- `labels`: Training labels (-100 for non-trainable tokens)
- `attention_mask`: Attention mask

### Raw Translated JSONL
```json
{
    "source_text": "Original English text",
    "generated_text": "Translated text",
    "source_problem": "The problem/prompt"
}
```

---

## Directory Structure

```
trl-eu/
├── trl/                           # HuggingFace TRL library (submodule)
├── scripts/
│   ├── sft_european.py            # Main training script
│   ├── determined/                # Determined AI launchers
│   │   ├── train_determined.py    # Python launcher with Core API
│   │   ├── launch_distributed.sh  # Bash launcher for multi-GPU
│   │   └── launch_single.sh       # Bash launcher for single GPU
│   └── slurm/
│       ├── test_sft_1gpu.sbatch   # SLURM test script
│       ├── test_sft_4gpu.sbatch   # SLURM 4-GPU test
│       └── sft_apertus_8gpu.sbatch # SLURM production script
└── configs/
    ├── accelerate_fsdp.yaml       # Single GPU FSDP config
    ├── accelerate_fsdp_8gpu.yaml  # Multi-GPU FSDP config
    └── determined/                # Determined experiment configs
        ├── experiment_1gpu.yaml   # 1-GPU test experiment
        ├── experiment_4gpu.yaml   # 4-GPU test experiment
        └── experiment_8gpu.yaml   # 8-GPU production experiment
```

---

## Dependencies

Key packages:
- transformers
- accelerate
- trl
- torch (with CUDA)
- flash-attn
- wandb (for logging)
- determined (for Determined AI integration)

### Environment Variables

```bash
# HuggingFace cache
export HF_HOME=/path/to/.cache/huggingface
export TRANSFORMERS_CACHE=/path/to/.cache/huggingface/transformers
export HF_DATASETS_CACHE=/path/to/.cache/huggingface/datasets

# WandB
export WANDB_PROJECT=eu-sft
```
