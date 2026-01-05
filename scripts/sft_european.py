#!/usr/bin/env python3
"""
European Multilingual SFT Training with TRL

Train European language models on translated Dolci-Think-SFT datasets.
Supports both pre-tokenized (packed) and raw data formats.

Usage:
    # With pre-tokenized data (recommended for large datasets)
    python scripts/sft_european.py \
        --model_name_or_path swiss-ai/Apertus-8B-Instruct-2509 \
        --pretokenized_path /path/to/pretokenized/packed/dolci_7b_sft/de \
        --output_dir ./outputs/apertus-8b-eu-de

    # With raw translated data (TRL handles tokenization)
    python scripts/sft_european.py \
        --model_name_or_path swiss-ai/Apertus-8B-Instruct-2509 \
        --dataset_path /path/to/translated/dolci_7b_sft/de.jsonl \
        --output_dir ./outputs/apertus-8b-eu-de

    # Test run with small subset
    python scripts/sft_european.py \
        --model_name_or_path Qwen/Qwen2.5-0.5B \
        --dataset_path /path/to/translated/dolci_7b_sft/de.jsonl \
        --max_samples 1000 \
        --output_dir ./outputs/test-run
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from datasets import Dataset, load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

# Note: Using TRL from conda environment (open-r1), not the local submodule
# TRL_PATH = Path(__file__).parent.parent / "trl"
# if TRL_PATH.exists():
#     sys.path.insert(0, str(TRL_PATH))

from trl import SFTConfig, SFTTrainer


def load_pretokenized_dataset(path: str) -> Dataset:
    """Load pre-tokenized dataset from disk (Arrow format)."""
    print(f"Loading pre-tokenized dataset from {path}")
    dataset = load_from_disk(path)
    print(f"Loaded {len(dataset)} examples")
    print(f"Columns: {dataset.column_names}")

    # Verify expected columns
    required_cols = {"input_ids", "labels", "attention_mask"}
    if not required_cols.issubset(set(dataset.column_names)):
        raise ValueError(f"Pre-tokenized dataset missing columns. Expected {required_cols}, got {dataset.column_names}")

    return dataset


def load_raw_dataset(path: str, max_samples: int = None) -> Dataset:
    """Load raw translated JSONL dataset."""
    print(f"Loading raw dataset from {path}")
    dataset = load_dataset("json", data_files=path, split="train")

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    print(f"Loaded {len(dataset)} examples")
    print(f"Columns: {dataset.column_names}")

    return dataset


def convert_translation_to_messages(example: dict) -> dict:
    """
    Convert translation output format to messages format for SFT.

    Input format (from translation):
        - source_text: original English text
        - generated_text: translated text
        - source_problem: the problem/prompt

    Output format:
        - messages: [{"role": "user", "content": ...}, {"role": "assistant", "content": ...}]
    """
    source_text = example.get("source_text", [])
    generated_text = example.get("generated_text", [])
    source_problem = example.get("source_problem", "")

    # Handle list format
    if isinstance(source_text, list):
        source_text = source_text[0] if source_text else ""
    if isinstance(generated_text, list):
        generated_text = generated_text[0] if generated_text else ""

    # Build messages
    user_content = source_problem if source_problem else source_text
    assistant_content = generated_text

    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content}
        ]
    }


def main():
    parser = argparse.ArgumentParser(description="European SFT Training with TRL")

    # Model
    parser.add_argument("--model_name_or_path", type=str, required=True,
                        help="Model name or path (e.g., swiss-ai/Apertus-8B-Instruct-2509)")
    parser.add_argument("--trust_remote_code", action="store_true", default=True,
                        help="Trust remote code for model loading")

    # Data - one of these required
    parser.add_argument("--pretokenized_path", type=str,
                        help="Path to pre-tokenized dataset (Arrow format)")
    parser.add_argument("--dataset_path", type=str,
                        help="Path to raw translated JSONL dataset")
    parser.add_argument("--max_samples", type=int,
                        help="Max samples for testing (only for raw datasets)")

    # Training
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for model and checkpoints")
    parser.add_argument("--max_seq_length", type=int, default=4096,
                        help="Max sequence length (ignored for pre-tokenized)")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1,
                        help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-6,
                        help="Learning rate")
    parser.add_argument("--num_train_epochs", type=int, default=1,
                        help="Number of training epochs")
    parser.add_argument("--warmup_ratio", type=float, default=0.03,
                        help="Warmup ratio")
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="Logging steps")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="Save checkpoint steps")

    # Memory optimization
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True,
                        help="Use gradient checkpointing")
    parser.add_argument("--use_liger_kernel", action="store_true",
                        help="Use Liger kernel for memory efficiency")
    parser.add_argument("--packing", action="store_true",
                        help="Enable packing (only for raw datasets)")

    # Misc
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--bf16", action="store_true", default=True, help="Use bf16")
    parser.add_argument("--report_to", type=str, default="wandb", help="Reporting backend")

    args = parser.parse_args()

    # Validate data args
    if not args.pretokenized_path and not args.dataset_path:
        parser.error("Either --pretokenized_path or --dataset_path required")

    use_pretokenized = args.pretokenized_path is not None

    # Load dataset
    if use_pretokenized:
        dataset = load_pretokenized_dataset(args.pretokenized_path)
    else:
        dataset = load_raw_dataset(args.dataset_path, args.max_samples)
        # Convert to messages format
        print("Converting to messages format...")
        dataset = dataset.map(
            convert_translation_to_messages,
            remove_columns=[c for c in dataset.column_names if c != "messages"],
            desc="Converting format"
        )

    # Load tokenizer
    print(f"Loading tokenizer from {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    print(f"Loading model from {args.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        trust_remote_code=args.trust_remote_code,
        attn_implementation="flash_attention_2",
    )

    # Training config - build kwargs dynamically
    config_kwargs = {
        "output_dir": args.output_dir,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_train_epochs,
        "warmup_ratio": args.warmup_ratio,
        "lr_scheduler_type": "cosine",
        "weight_decay": 0.01,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "save_total_limit": 3,
        "gradient_checkpointing": args.gradient_checkpointing,
        "bf16": args.bf16,
        "seed": args.seed,
        "report_to": args.report_to,
    }

    # Add SFT-specific args only when not using pre-tokenized data
    if not use_pretokenized:
        config_kwargs["max_seq_length"] = args.max_seq_length
        config_kwargs["packing"] = args.packing
    else:
        # For pre-tokenized data, skip dataset preparation
        config_kwargs["dataset_kwargs"] = {"skip_prepare_dataset": True}

    # Memory optimization
    if args.use_liger_kernel:
        config_kwargs["use_liger_kernel"] = True

    training_args = SFTConfig(**config_kwargs)

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Save
    print(f"Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("Training complete!")


if __name__ == "__main__":
    main()
