#!/usr/bin/env python
"""
Progressive On-Policy Distillation with vLLM

This script implements the progressive distillation algorithm:
- N rounds of training
- Each round: T_i = α * SFT + (1-α) * R_i
- R_i = S_{i-1} (reference model is previous stage's student)
- Uses DistillationTrainer + vLLM for maximum speed

Architecture:
1. Teacher server (separate GPUs): SFT + ref models, serves mixed logits
2. Student training (main GPUs): DistillationTrainer with vLLM generation

Usage:
    # Start teacher server first (on separate GPUs, e.g., GPU 0-1):
    CUDA_VISIBLE_DEVICES=0,1 python progressive_teacher_server.py \\
        --sft-model /path/to/sft_model \\
        --initial-ref-model /path/to/base_model \\
        --mixing-ratio 0.5 \\
        --port 8000 \\
        --tensor-parallel-size 2

    # Then run training (on remaining GPUs, e.g., GPU 2-7):
    CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 torchrun --nproc_per_node=6 progressive_distillation.py \\
        --student_model_path /path/to/base_model \\
        --teacher_server_url http://localhost:8000 \\
        --data_path /path/to/data.json \\
        --num_stages 10 \\
        --epochs_per_stage 1 \\
        --mixing_ratio 0.5 \\
        --lmbda 1.0 \\
        --use_vllm
"""

import argparse
import glob
import json
import os
import time
from datetime import datetime

import requests
import torch
from datasets import Dataset
from transformers import AutoTokenizer

try:
    from trl.experimental.distillation import DistillationConfig, DistillationTrainer
except ImportError as exc:
    raise ImportError(
        "DistillationTrainer not found. Install recent TRL: pip install -U trl[vllm]"
    ) from exc


def load_data(data_path, data_num=None, prompt_only=False):
    """
    Load LLaMA-Factory style JSON into TRL conversational format.

    Args:
        data_path: Path to JSON file with {instruction, input, output} format
        data_num: Limit number of samples (for debugging)
        prompt_only: If True, only keep user prompts (for fully on-policy lmbda=1.0)
    """
    with open(data_path, "r", encoding="utf-8") as f:
        all_data = json.load(f)

    if data_num is not None:
        all_data = all_data[:data_num]

    messages = []
    for d in all_data:
        prompt = f"{d.get('instruction', '')}{d.get('input', '')}"
        turns = [{"role": "user", "content": prompt}]

        # For fully on-policy (lmbda=1.0), we don't need gold completions
        if not prompt_only:
            turns.append({"role": "assistant", "content": d.get("output", "")})

        messages.append(turns)

    return Dataset.from_dict({"messages": messages})


def get_latest_checkpoint(output_dir):
    """Get the most recent checkpoint from a training output directory."""
    checkpoint_dirs = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    if not checkpoint_dirs:
        return None
    checkpoint_dirs.sort(key=lambda x: int(x.rsplit("-", 1)[-1]))
    return checkpoint_dirs[-1]


def update_teacher_reference_model(teacher_url: str, new_model_path: str, timeout: int = 300):
    """
    Update the reference model on the teacher server.

    This is called between stages to set R_i = S_{i-1}.
    """
    print(f"\n{'='*80}")
    print(f"Updating teacher reference model to: {new_model_path}")
    print(f"{'='*80}")

    update_url = f"{teacher_url}/update_reference"

    try:
        response = requests.post(
            update_url,
            json={"model_path": new_model_path},
            timeout=timeout
        )
        response.raise_for_status()
        result = response.json()

        print(f"✓ Reference model updated successfully")
        print(f"  Old model: {result.get('old_model', 'N/A')}")
        print(f"  New model: {result.get('new_model', 'N/A')}")
        print(f"{'='*80}\n")

        return True

    except requests.exceptions.RequestException as e:
        print(f"✗ Failed to update reference model: {e}")
        return False


def check_teacher_server(teacher_url: str, timeout: int = 10):
    """Check if teacher server is ready."""
    try:
        response = requests.get(f"{teacher_url}/status", timeout=timeout)
        response.raise_for_status()
        status = response.json()
        print(f"\nTeacher server status:")
        print(f"  Status: {status.get('status')}")
        print(f"  SFT model: {status.get('sft_model')}")
        print(f"  Ref model: {status.get('ref_model')}")
        print(f"  Mixing ratio: {status.get('mixing_ratio')}\n")
        return status.get('status') == 'ready'
    except Exception as e:
        print(f"Teacher server not ready: {e}")
        return False


def str2bool(x):
    if isinstance(x, bool):
        return x
    return str(x).lower() in {"1", "true", "yes", "y"}


def get_args():
    parser = argparse.ArgumentParser(description="Progressive On-Policy Distillation")

    # Model paths
    parser.add_argument("--student_model_path", type=str, required=True,
                        help="Path to initial student model (base model for stage 0)")
    parser.add_argument("--teacher_server_url", type=str, required=True,
                        help="URL of the progressive teacher server (e.g., http://localhost:8000)")

    # Data
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to training data (LLaMA-Factory JSON format)")
    parser.add_argument("--data_name", type=str, default="train",
                        help="Dataset name for logging")
    parser.add_argument("--eval_data_num", type=int, default=200,
                        help="Number of samples for evaluation")
    parser.add_argument("--prompt_only", action="store_true",
                        help="Only use prompts (for fully on-policy lmbda=1.0)")

    # Progressive distillation
    parser.add_argument("--num_stages", type=int, default=10,
                        help="Total number of progressive stages")
    parser.add_argument("--epochs_per_stage", type=float, default=1,
                        help="Training epochs per stage")
    parser.add_argument("--mixing_ratio", type=float, default=0.5,
                        help="Mixing ratio α for teacher (must match server config)")

    # Training hyperparameters
    parser.add_argument("--per_device_train_batch_size", type=int, default=1,
                        help="Batch size per GPU")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--dataloader_num_workers", type=int, default=8,
                        help="Number of dataloader workers")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Learning rate")
    parser.add_argument("--warmup_ratio", type=float, default=0.0,
                        help="Warmup ratio")
    parser.add_argument("--weight_decay", type=float, default=0.0,
                        help="Weight decay")

    # Sequence lengths
    parser.add_argument("--max_length", type=int, default=2048,
                        help="Maximum total sequence length")
    parser.add_argument("--max_prompt_length", type=int, default=1536,
                        help="Maximum prompt length")
    parser.add_argument("--max_completion_length", type=int, default=512,
                        help="Maximum completion length")

    # Generation parameters
    parser.add_argument("--temperature", type=float, default=0.9,
                        help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.95,
                        help="Nucleus sampling top-p")
    parser.add_argument("--top_k", type=int, default=0,
                        help="Top-k sampling (0 = disabled)")
    parser.add_argument("--num_generations", type=int, default=1,
                        help="Number of generations per prompt")
    parser.add_argument("--generation_batch_size", type=int, default=None,
                        help="Generation batch size (auto if None)")

    # Distillation loss
    parser.add_argument("--lmbda", type=float, default=1.0,
                        help="On-policy mixing (0=off-policy, 1=fully on-policy)")
    parser.add_argument("--beta", type=float, default=1.0,
                        help="JSD beta (0=forward KL, 0.5=JSD, 1=reverse KL)")
    parser.add_argument("--loss_top_k", type=int, default=1,
                        help="Top-k tokens for loss computation")
    parser.add_argument("--loss_add_tail", type=str2bool, default=True,
                        help="Add tail bucket in loss")

    # Performance
    parser.add_argument("--bf16", type=str2bool, default=True,
                        help="Use bfloat16")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Enable gradient checkpointing")
    parser.add_argument("--use_flash_attention", action="store_true",
                        help="Use Flash Attention 2")
    parser.add_argument("--use_liger_kernel", action="store_true",
                        help="Use Liger kernel for fused JSD")

    # vLLM for student generation
    parser.add_argument("--use_vllm", action="store_true",
                        help="Use vLLM for student on-policy generation")
    parser.add_argument("--vllm_mode", type=str, default="colocate",
                        choices=["server", "colocate"],
                        help="vLLM mode (server requires separate vllm-serve)")
    parser.add_argument("--vllm_server_base_url", type=str, default="http://127.0.0.1:8001",
                        help="Student vLLM server URL (server mode only)")
    parser.add_argument("--vllm_server_timeout", type=float, default=600.0,
                        help="vLLM server timeout")
    parser.add_argument("--vllm_sync_frequency", type=int, default=1,
                        help="Frequency to sync weights to vLLM")
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.3,
                        help="GPU memory utilization for vLLM (colocate mode)")
    parser.add_argument("--vllm_tensor_parallel_size", type=int, default=1,
                        help="Tensor parallel size for vLLM")
    parser.add_argument("--vllm_max_model_length", type=int, default=None,
                        help="Max model length for vLLM")
    parser.add_argument("--vllm_enable_sleep_mode", action="store_true",
                        help="Enable vLLM sleep mode")

    # Checkpointing
    parser.add_argument("--save_strategy", type=str, default="epoch",
                        choices=["no", "steps", "epoch"],
                        help="Checkpoint save strategy")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="Save checkpoint every N steps")
    parser.add_argument("--save_total_limit", type=int, default=2,
                        help="Maximum number of checkpoints to keep")
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="Log every N steps")

    # Output
    parser.add_argument("--model_name", type=str, default="Qwen3-4B",
                        help="Model name for logging")
    parser.add_argument("--exp_name", type=str, default="progressive_distill",
                        help="Experiment name")
    parser.add_argument("--output_base_dir", type=str, default="./outputs",
                        help="Base output directory")

    return parser.parse_args()


def save_stage_args(args, output_dir, stage):
    """Save stage configuration."""
    args_dict = vars(args).copy()
    args_dict["current_stage"] = stage
    args_dict["timestamp"] = datetime.now().isoformat()

    with open(os.path.join(output_dir, "args.json"), "w", encoding="utf-8") as f:
        json.dump(args_dict, f, ensure_ascii=False, indent=2)


def build_model_init_kwargs(args):
    """Build model initialization kwargs."""
    kwargs = {
        "torch_dtype": torch.bfloat16 if args.bf16 else torch.float16,
        "device_map": None,  # Disable auto device mapping for DDP
    }
    if args.use_flash_attention:
        kwargs["attn_implementation"] = "flash_attention_2"
    return kwargs


def run_single_stage(args, stage: int, student_model_path: str):
    """
    Run a single stage of progressive distillation.

    Args:
        args: Command-line arguments
        stage: Current stage number (1-indexed)
        student_model_path: Path to student model for this stage

    Returns:
        Path to the trained model checkpoint (for next stage)
    """
    print(f"\n{'#'*80}")
    print(f"STAGE {stage}/{args.num_stages}")
    print(f"{'#'*80}")
    print(f"Student model: {student_model_path}")
    print(f"Teacher server: {args.teacher_server_url}")
    print(f"Mixing ratio α: {args.mixing_ratio}")
    print(f"Lambda (on-policy): {args.lmbda}")
    print(f"Beta (KL): {args.beta}")
    print(f"{'#'*80}\n")

    # Build run name and output directory
    run_name = (
        f"ProgressiveDistill+{args.model_name}+{args.data_name}+"
        f"bs{args.per_device_train_batch_size}_ga{args.gradient_accumulation_steps}+"
        f"t{args.temperature}_beta{args.beta}_lambda{args.lmbda}+"
        f"mix{args.mixing_ratio}_lr{args.lr}+"
        f"{args.exp_name}_stage{stage}"
    )
    output_dir = os.path.join(args.output_base_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)

    # Save configuration
    save_stage_args(args, output_dir, stage)

    # Load data
    prompt_only = args.prompt_only or args.lmbda >= 1.0
    print(f"Loading training data (prompt_only={prompt_only})...")
    train_dataset = load_data(args.data_path, prompt_only=prompt_only)
    eval_dataset = load_data(args.data_path, data_num=args.eval_data_num, prompt_only=prompt_only)
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Eval samples: {len(eval_dataset)}\n")

    # Load tokenizer
    print(f"Loading tokenizer from {student_model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(student_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("  Tokenizer loaded\n")

    # Build model init kwargs
    model_init_kwargs = build_model_init_kwargs(args)
    teacher_model_init_kwargs = build_model_init_kwargs(args)

    # Create DistillationConfig
    config = DistillationConfig(
        # Output
        output_dir=output_dir,
        run_name=run_name,
        report_to="none",

        # Training
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs_per_stage,
        dataloader_num_workers=args.dataloader_num_workers,

        # Model
        model_init_kwargs=model_init_kwargs,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,

        # Teacher (external server)
        use_teacher_server=True,
        teacher_model_server_url=args.teacher_server_url,
        teacher_model_init_kwargs=teacher_model_init_kwargs,

        # Distillation
        temperature=args.temperature,
        lmbda=args.lmbda,
        beta=args.beta,
        loss_top_k=args.loss_top_k,
        loss_add_tail=args.loss_add_tail,
        use_liger_kernel=False,  # Not compatible with external teacher

        # Generation
        top_p=args.top_p,
        top_k=args.top_k,
        num_generations=args.num_generations,
        generation_batch_size=args.generation_batch_size,

        # vLLM for student
        use_vllm=args.use_vllm,
        vllm_mode=args.vllm_mode,
        vllm_server_base_url=args.vllm_server_base_url if args.vllm_mode == "server" else None,
        vllm_server_timeout=args.vllm_server_timeout,
        vllm_sync_frequency=args.vllm_sync_frequency,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
        vllm_max_model_length=args.vllm_max_model_length,
        vllm_enable_sleep_mode=args.vllm_enable_sleep_mode,

        # Checkpointing and logging
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        logging_steps=args.logging_steps,
    )

    # Create trainer
    print("Creating DistillationTrainer...")
    trainer = DistillationTrainer(
        model=student_model_path,
        teacher_model=None,  # Using external teacher server
        args=config,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    print("  Trainer created\n")

    # Train
    print(f"Starting training for stage {stage}...")
    print(f"  Epochs: {args.epochs_per_stage}")
    print(f"  Steps per epoch: ~{len(train_dataset) // (args.per_device_train_batch_size * args.gradient_accumulation_steps)}")
    print()

    start_time = time.time()
    trainer.train()
    elapsed = time.time() - start_time

    print(f"\n{'='*80}")
    print(f"Stage {stage} training completed in {elapsed:.1f}s ({elapsed/60:.1f}m)")
    print(f"{'='*80}\n")

    # Save final model
    stage_final_dir = os.path.join(output_dir, "stage-final")
    print(f"Saving final model to {stage_final_dir}...")
    trainer.save_model(stage_final_dir)
    tokenizer.save_pretrained(stage_final_dir)
    print("  Model saved\n")

    # Cleanup
    del trainer
    torch.cuda.empty_cache()

    return stage_final_dir


def main():
    args = get_args()

    print(f"\n{'#'*80}")
    print(f"PROGRESSIVE ON-POLICY DISTILLATION")
    print(f"{'#'*80}")
    print(f"Started at: {datetime.now().isoformat(timespec='seconds')}")
    print(f"Total stages: {args.num_stages}")
    print(f"Epochs per stage: {args.epochs_per_stage}")
    print(f"Mixing ratio α: {args.mixing_ratio}")
    print(f"On-policy lambda: {args.lmbda}")
    print(f"Student vLLM: {args.use_vllm} (mode: {args.vllm_mode if args.use_vllm else 'N/A'})")
    print(f"Teacher server: {args.teacher_server_url}")
    print(f"{'#'*80}\n")

    # Check teacher server
    print("Checking teacher server...")
    if not check_teacher_server(args.teacher_server_url):
        print("\n✗ Teacher server is not ready!")
        print("Please start the teacher server first:")
        print(f"  python progressive_teacher_server.py \\")
        print(f"    --sft-model <sft_model_path> \\")
        print(f"    --initial-ref-model <base_model_path> \\")
        print(f"    --mixing-ratio {args.mixing_ratio} \\")
        print(f"    --port {args.teacher_server_url.split(':')[-1]}")
        return

    print("✓ Teacher server is ready\n")

    # Progressive training loop
    current_student_path = args.student_model_path

    for stage in range(1, args.num_stages + 1):
        # Run this stage
        trained_model_path = run_single_stage(args, stage, current_student_path)

        # Update teacher's reference model for next stage
        if stage < args.num_stages:
            print(f"\nPreparing for stage {stage + 1}...")
            success = update_teacher_reference_model(
                args.teacher_server_url,
                trained_model_path
            )

            if not success:
                print(f"✗ Failed to update teacher reference model. Stopping.")
                break

            # Next stage uses this stage's trained model as student
            current_student_path = trained_model_path

            # Brief pause to let server settle
            print("Waiting 5 seconds before next stage...")
            time.sleep(5)

    print(f"\n{'#'*80}")
    print(f"ALL {args.num_stages} STAGES COMPLETED!")
    print(f"{'#'*80}")
    print(f"Final model: {current_student_path}")
    print(f"Completed at: {datetime.now().isoformat(timespec='seconds')}")
    print(f"{'#'*80}\n")


if __name__ == "__main__":
    main()
