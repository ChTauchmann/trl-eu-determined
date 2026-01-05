# TRL European Language Models

Training European language models using HuggingFace TRL for SFT and beyond.

## Overview

This repository extends HuggingFace TRL for training European multilingual models. It supports:
- Pre-tokenized packed datasets (from open-instruct tokenization pipeline)
- Multi-GPU training with FSDP
- Memory optimizations (gradient checkpointing, Liger kernel)

## Target Model

- **Base Model**: swiss-ai/Apertus-8B-Instruct-2509
- **Languages**: German (de), Spanish (es), French (fr), Italian (it)
- **Dataset**: Translated Dolci-Think-SFT

## Quick Start

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

### Manual Run
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

## Directory Structure
```
trl-eu/
├── trl/                    # HuggingFace TRL library
├── scripts/
│   ├── sft_european.py     # Main training script
│   └── slurm/
│       ├── test_sft_1gpu.sbatch    # Test script
│       └── sft_apertus_8gpu.sbatch # Production script
└── configs/
    ├── accelerate_fsdp.yaml        # Single GPU config
    └── accelerate_fsdp_8gpu.yaml   # Multi-GPU config
```

## Dependencies

Uses the included TRL library (submodule). Key packages:
- transformers
- accelerate
- trl
- torch (with CUDA)
- flash-attn
- wandb (for logging)
