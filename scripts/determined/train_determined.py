#!/usr/bin/env python3
"""
Determined AI Training Wrapper for European SFT

This script provides a clean integration with Determined AI's Core API,
handling distributed training, checkpointing, and metrics reporting.

Usage:
    # Submit experiment
    det experiment create configs/determined/experiment_8gpu.yaml .

    # Or run directly for testing
    python scripts/determined/train_determined.py
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime

# Attempt to import determined - graceful fallback for local testing
try:
    import determined as det
    from determined import core
    DETERMINED_AVAILABLE = True
except ImportError:
    DETERMINED_AVAILABLE = False
    print("Warning: Determined not available, running in standalone mode")


def get_hyperparameters():
    """Get hyperparameters from Determined or environment."""
    if DETERMINED_AVAILABLE:
        try:
            info = det.get_cluster_info()
            if info and info.trial:
                return info.trial.hparams
        except Exception as e:
            print(f"Could not get Determined hyperparameters: {e}")

    # Fallback to environment variables or defaults
    return {
        "model_name_or_path": os.environ.get("model_name_or_path", "Qwen/Qwen2.5-0.5B"),
        "pretokenized_path": os.environ.get("pretokenized_path"),
        "dataset_path": os.environ.get("dataset_path"),
        "max_samples": os.environ.get("max_samples"),
        "output_dir": os.environ.get("output_dir", "./outputs"),
        "per_device_train_batch_size": int(os.environ.get("per_device_train_batch_size", "1")),
        "gradient_accumulation_steps": int(os.environ.get("gradient_accumulation_steps", "4")),
        "learning_rate": float(os.environ.get("learning_rate", "2e-6")),
        "num_train_epochs": int(os.environ.get("num_train_epochs", "1")),
        "warmup_ratio": float(os.environ.get("warmup_ratio", "0.03")),
        "max_seq_length": int(os.environ.get("max_seq_length", "4096")),
        "logging_steps": int(os.environ.get("logging_steps", "10")),
        "save_steps": int(os.environ.get("save_steps", "500")),
        "report_to": os.environ.get("report_to", "wandb"),
        "gradient_checkpointing": os.environ.get("gradient_checkpointing", "true").lower() == "true",
        "use_liger_kernel": os.environ.get("use_liger_kernel", "false").lower() == "true",
        "packing": os.environ.get("packing", "false").lower() == "true",
        "bf16": os.environ.get("bf16", "true").lower() == "true",
        "seed": int(os.environ.get("seed", "42")),
    }


def get_num_gpus():
    """Determine number of GPUs available."""
    # Check Determined environment
    if os.environ.get("DET_SLOT_IDS"):
        return len(os.environ["DET_SLOT_IDS"].split(","))
    if os.environ.get("DET_SLOTS_PER_TRIAL"):
        return int(os.environ["DET_SLOTS_PER_TRIAL"])

    # Fallback: check CUDA_VISIBLE_DEVICES
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        return len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))

    # Last resort: try nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True, text=True, timeout=10
        )
        return len(result.stdout.strip().split("\n"))
    except Exception:
        return 1


def generate_accelerate_config(num_gpus: int, config_path: str = "/tmp/accelerate_fsdp.yaml"):
    """Generate accelerate FSDP config for the given number of GPUs."""
    config = f"""compute_environment: LOCAL_MACHINE
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
num_processes: {num_gpus}
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
"""
    with open(config_path, "w") as f:
        f.write(config)
    return config_path


def build_training_command(hparams: dict, num_gpus: int, repo_dir: Path):
    """Build the training command with all arguments."""
    train_script = repo_dir / "scripts" / "sft_european.py"

    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(hparams["output_dir"]) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    # Base command
    if num_gpus > 1:
        accelerate_config = generate_accelerate_config(num_gpus)
        cmd = [
            "accelerate", "launch",
            "--config_file", accelerate_config,
            str(train_script)
        ]
    else:
        cmd = ["python", str(train_script)]

    # Add arguments
    cmd.extend([
        "--model_name_or_path", str(hparams["model_name_or_path"]),
        "--output_dir", str(output_dir),
        "--per_device_train_batch_size", str(hparams["per_device_train_batch_size"]),
        "--gradient_accumulation_steps", str(hparams["gradient_accumulation_steps"]),
        "--learning_rate", str(hparams["learning_rate"]),
        "--num_train_epochs", str(hparams["num_train_epochs"]),
        "--warmup_ratio", str(hparams["warmup_ratio"]),
        "--max_seq_length", str(hparams["max_seq_length"]),
        "--logging_steps", str(hparams["logging_steps"]),
        "--save_steps", str(hparams["save_steps"]),
        "--report_to", str(hparams["report_to"]),
        "--seed", str(hparams["seed"]),
    ])

    # Data source
    pretokenized = hparams.get("pretokenized_path")
    dataset = hparams.get("dataset_path")

    if pretokenized and pretokenized not in ("null", "None", ""):
        cmd.extend(["--pretokenized_path", str(pretokenized)])
    elif dataset and dataset not in ("null", "None", ""):
        cmd.extend(["--dataset_path", str(dataset)])
        max_samples = hparams.get("max_samples")
        if max_samples and max_samples not in ("null", "None", ""):
            cmd.extend(["--max_samples", str(max_samples)])
    else:
        raise ValueError("Either pretokenized_path or dataset_path must be set")

    # Boolean flags
    if hparams.get("gradient_checkpointing", True):
        cmd.append("--gradient_checkpointing")
    if hparams.get("bf16", True):
        cmd.append("--bf16")
    if hparams.get("use_liger_kernel", False):
        cmd.append("--use_liger_kernel")
    if hparams.get("packing", False):
        cmd.append("--packing")

    return cmd


def main():
    """Main entry point for Determined training."""
    # Find repo directory
    script_dir = Path(__file__).parent
    repo_dir = script_dir.parent.parent

    print("=" * 60)
    print("Determined AI European SFT Training")
    print("=" * 60)

    # Get configuration
    hparams = get_hyperparameters()
    num_gpus = get_num_gpus()

    print(f"\nConfiguration:")
    print(f"  Repository: {repo_dir}")
    print(f"  Number of GPUs: {num_gpus}")
    print(f"  Model: {hparams.get('model_name_or_path')}")
    print(f"  Batch size: {hparams.get('per_device_train_batch_size')}")
    print(f"  Learning rate: {hparams.get('learning_rate')}")

    if DETERMINED_AVAILABLE:
        print(f"\nDetermined Environment:")
        print(f"  Experiment ID: {os.environ.get('DET_EXPERIMENT_ID', 'N/A')}")
        print(f"  Trial ID: {os.environ.get('DET_TRIAL_ID', 'N/A')}")

    # Build and execute training command
    cmd = build_training_command(hparams, num_gpus, repo_dir)

    print(f"\nExecuting command:")
    print(f"  {' '.join(cmd)}")
    print("=" * 60)
    print()

    # Execute
    result = subprocess.run(cmd, cwd=str(repo_dir))
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
