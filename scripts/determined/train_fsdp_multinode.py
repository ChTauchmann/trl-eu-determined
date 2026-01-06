#!/usr/bin/env python3
"""
Multi-node FSDP Training with Native PyTorch on Determined AI

This script uses Determined only for resource allocation and coordination.
Actual distributed training is handled by PyTorch FSDP via HuggingFace Accelerate.

Usage:
    # Determined will provide DET_CHIEF_IP, DET_SLOT_IDS, etc.
    python scripts/determined/train_fsdp_multinode.py
"""

import os
import subprocess
import sys
import socket
import time
from pathlib import Path


def get_determined_info():
    """Extract distributed training info from Determined environment."""
    # Determined provides these environment variables
    chief_ip = os.environ.get("DET_CHIEF_IP", "localhost")

    # DET_SLOT_IDS is a comma-separated list of GPU indices for this container
    slot_ids = os.environ.get("DET_SLOT_IDS", "0")
    num_local_gpus = len(slot_ids.split(","))

    # Calculate rank from container info
    # DET_CONTAINER_RANK_<n> tells us our position
    container_rank = 0
    for key, val in os.environ.items():
        if key.startswith("DET_CONTAINER_RANK_"):
            container_rank = int(val)
            break

    # Alternatively, use simpler approach with DET_RANK if available
    rank = int(os.environ.get("DET_RANK", os.environ.get("RANK", container_rank)))
    world_size = int(os.environ.get("DET_WORLD_SIZE", os.environ.get("WORLD_SIZE", "1")))

    # Number of nodes
    num_nodes = world_size // num_local_gpus if num_local_gpus > 0 else 1
    node_rank = rank // num_local_gpus if num_local_gpus > 0 else 0

    return {
        "chief_ip": chief_ip,
        "num_local_gpus": num_local_gpus,
        "rank": rank,
        "world_size": world_size,
        "num_nodes": num_nodes,
        "node_rank": node_rank,
        "slot_ids": slot_ids,
    }


def get_free_port():
    """Find a free port on this machine."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def create_fsdp_config(num_processes: int, num_machines: int, machine_rank: int, main_ip: str) -> str:
    """Create accelerate FSDP config dynamically."""
    config = f"""compute_environment: LOCAL_MACHINE
debug: false
distributed_type: FSDP
downcast_bf16: 'no'
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
machine_rank: {machine_rank}
main_process_ip: {main_ip}
main_process_port: 29500
main_training_function: main
mixed_precision: bf16
num_machines: {num_machines}
num_processes: {num_processes}
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
"""
    return config


def main():
    """Launch multi-node FSDP training."""
    info = get_determined_info()

    print(f"[train_fsdp_multinode] Determined info: {info}", flush=True)

    # Get training script path
    script_dir = Path(__file__).parent.parent.parent
    train_script = script_dir / "scripts" / "sft_european.py"

    # Training hyperparameters - can be overridden by environment
    model_name = os.environ.get("MODEL_NAME", "swiss-ai/Apertus-8B-Instruct-2509")
    dataset_path = os.environ.get("DATASET_PATH", "/tmp/test_sft_data.jsonl")
    output_dir = os.environ.get("OUTPUT_DIR", str(script_dir / "outputs" / "fsdp-multinode"))
    max_samples = os.environ.get("MAX_SAMPLES", "")
    num_epochs = os.environ.get("NUM_EPOCHS", "1")
    batch_size = os.environ.get("BATCH_SIZE", "1")
    grad_accum = os.environ.get("GRAD_ACCUM", "2")

    # Total GPUs = num_nodes * gpus_per_node
    total_gpus = info["num_nodes"] * info["num_local_gpus"]

    # Create config
    config_content = create_fsdp_config(
        num_processes=info["num_local_gpus"],  # GPUs on this node
        num_machines=info["num_nodes"],
        machine_rank=info["node_rank"],
        main_ip=info["chief_ip"],
    )

    # Write config to temp file
    config_path = Path("/tmp/fsdp_config.yaml")
    config_path.write_text(config_content)
    print(f"[train_fsdp_multinode] FSDP config written to {config_path}", flush=True)
    print(config_content, flush=True)

    # Build training command
    cmd = [
        "accelerate", "launch",
        "--config_file", str(config_path),
        str(train_script),
        "--model_name_or_path", model_name,
        "--dataset_path", dataset_path,
        "--output_dir", output_dir,
        "--num_train_epochs", num_epochs,
        "--per_device_train_batch_size", batch_size,
        "--gradient_accumulation_steps", grad_accum,
        "--logging_steps", "1",
        "--save_steps", "500",
        "--report_to", os.environ.get("REPORT_TO", "wandb"),
    ]

    if max_samples:
        cmd.extend(["--max_samples", max_samples])

    print(f"[train_fsdp_multinode] Launching: {' '.join(cmd)}", flush=True)

    # Set up environment for NCCL
    env = os.environ.copy()
    env.update({
        "NCCL_DEBUG": "INFO",
        "NCCL_IB_DISABLE": "1",  # Disable InfiniBand if not available
        "NCCL_SOCKET_IFNAME": "eth0",
        "MASTER_ADDR": info["chief_ip"],
        "MASTER_PORT": "29500",
    })

    # Run training
    result = subprocess.run(cmd, env=env)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
