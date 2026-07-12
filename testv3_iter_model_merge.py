from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl.experimental.gkd import GKDConfig, GKDTrainer
import json
import torch
from datetime import datetime
import os
import argparse
import subprocess
import glob as glob_module
import sys


MODEL_WEIGHT_PATTERNS = (
    "model.safetensors",
    "model-*.safetensors",
    "pytorch_model.bin",
    "pytorch_model-*.bin",
)


def has_hf_model_weights(model_dir):
    """Return True only if the directory can be loaded as a HF model checkpoint."""
    if not os.path.isdir(model_dir):
        return False
    return any(glob_module.glob(os.path.join(model_dir, pattern)) for pattern in MODEL_WEIGHT_PATTERNS)


def get_latest_checkpoint(output_dir):
    checkpoint_dirs = glob_module.glob(os.path.join(output_dir, "checkpoint-*"))
    if not checkpoint_dirs:
        return None
    checkpoint_dirs.sort(key=lambda x: int(x.rsplit("-", 1)[-1]))
    return checkpoint_dirs[-1]


def get_world_size():
    return int(os.environ.get("WORLD_SIZE", "1"))


def convert_deepspeed_to_fp32(checkpoint_dir, output_path):
    """
    将DeepSpeed ZeRO-3 checkpoint转换为完整的FP32模型

    Args:
        checkpoint_dir: DeepSpeed checkpoint目录（包含zero_to_fp32.py的目录）
        output_path: 输出的完整模型路径
    """
    zero_to_fp32_script = os.path.join(checkpoint_dir, "zero_to_fp32.py")

    if not os.path.exists(zero_to_fp32_script):
        print(f"Warning: zero_to_fp32.py not found in {checkpoint_dir}")
        return False

    # 查找最新的global_step目录
    global_step_dirs = glob_module.glob(os.path.join(checkpoint_dir, "global_step*"))
    if not global_step_dirs:
        print(f"Warning: No global_step directory found in {checkpoint_dir}")
        return False

    # 按step编号排序，取最新的
    global_step_dirs.sort(key=lambda x: int(x.split("global_step")[-1].rstrip("/")))
    latest_step_dir = global_step_dirs[-1]

    print(f"Converting DeepSpeed checkpoint from {latest_step_dir}...")
    print(f"Output path: {output_path}")

    try:
        # 运行zero_to_fp32.py脚本
        cmd = [sys.executable, zero_to_fp32_script, latest_step_dir, output_path]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(result.stdout)
        print(f"Successfully converted checkpoint to {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error converting checkpoint: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False


def merge_model_parameters(sft_model, ref_model, mixing_ratio):
    """
    将两个模型的参数按照mixing_ratio进行加权合并
    merged_param = mixing_ratio * sft_param + (1 - mixing_ratio) * ref_param

    Args:
        sft_model: SFT模型
        ref_model: 参考模型
        mixing_ratio: SFT模型的权重（0-1之间）

    Returns:
        merged_model: 参数已合并的新模型（基于sft_model）
    """
    print(f"Merging model parameters with ratio {mixing_ratio}...")

    # 获取两个模型的state dict
    sft_state_dict = sft_model.state_dict()
    ref_state_dict = ref_model.state_dict()

    # 创建合并后的state dict
    merged_state_dict = {}

    for param_name in sft_state_dict.keys():
        if param_name in ref_state_dict:
            # 加权合并参数
            merged_state_dict[param_name] = (
                mixing_ratio * sft_state_dict[param_name] +
                (1.0 - mixing_ratio) * ref_state_dict[param_name]
            )
        else:
            # 如果ref_model中没有对应参数，直接使用sft_model的参数
            merged_state_dict[param_name] = sft_state_dict[param_name]

    # 将合并后的参数加载到sft_model中
    sft_model.load_state_dict(merged_state_dict)

    print("Model parameters merged successfully!")
    return sft_model


def load_data(data_path, data_num=None):
    data_dict = {}
    with open(data_path, 'r') as f:
        all_data = json.load(f)
        messages = []
        for d in all_data: 
            messages.append([{"role": "user", "content": d['instruction'] + d['input']}, {"role": "assistant", "content": d['output']}])
        
        if data_num is not None:
            messages = messages[:data_num]
        
        data_dict['messages'] = messages
        
    return Dataset.from_dict(data_dict)


def load_model_and_tokenizer(teacher_model_path, student_model_path, ref_model_path, mixing_ratio):
    """加载模型和tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(teacher_model_path)

    # The model to optimise (student)
    model = AutoModelForCausalLM.from_pretrained(
        student_model_path,
        torch_dtype=torch.bfloat16
    )

    # The SFT model (teacher的一部分)
    sft_model = AutoModelForCausalLM.from_pretrained(
        teacher_model_path,
        torch_dtype=torch.bfloat16
    )

    # The reference model (teacher的另一部分)
    ref_model = AutoModelForCausalLM.from_pretrained(
        ref_model_path,
        torch_dtype=torch.bfloat16
    )

    # 合并两个模型的参数，得到teacher模型
    teacher_model = merge_model_parameters(sft_model, ref_model, mixing_ratio)

    # 释放ref_model的显存
    del ref_model
    torch.cuda.empty_cache()

    return model, teacher_model, tokenizer


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='Qwen2.5-3B-Instruct', help='Model name')
    parser.add_argument('--data_name', type=str, default='medcalc_train', help='Dataset name')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--teacher_model_path', type=str, required=True, help='Path to the teacher (SFT) model')
    parser.add_argument('--ref_model_path', type=str, required=True, help='Path to the initial reference model')
    parser.add_argument('--student_model_path', type=str, required=True, help='Path to the initial student model')
    parser.add_argument('--mixing_ratio', type=float, default=0.2, help='Weight for SFT model logits')
    parser.add_argument('--save_steps', type=int, default=500, help='Save steps')
    parser.add_argument('--per_device_train_batch_size', type=int, default=3, help='Per device train batch size')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--epochs_per_stage', type=float, default=5, help='Number of training epochs per stage')
    parser.add_argument('--num_stages', type=int, default=10, help='Total number of iterative stages')
    parser.add_argument('--max_new_tokens', type=int, default=4096, help='Maximum number of new tokens to generate')
    parser.add_argument('--max_length', type=int, default=4096, help='Maximum sequence length for training')
    parser.add_argument('--temperature', type=float, default=0)
    parser.add_argument('--lmbda', type=float, default=0)
    parser.add_argument('--beta', type=float, default=0)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--exp_name', type=str, default='iter', help='Experiment name prefix')
    parser.add_argument('--output_base_dir', type=str, default='./outputs', help='Base output directory')
    parser.add_argument('--deepspeed', type=str, default=None, help='Path to deepspeed config file')
    args = parser.parse_args()
    return args


def save_args(args, output_dir, stage):
    """保存当前阶段的参数配置"""
    args_dict = vars(args).copy()
    args_dict['current_stage'] = stage
    with open(os.path.join(output_dir, 'args.json'), 'w') as f:
        json.dump(args_dict, f, ensure_ascii=False, indent=2)


def convert_checkpoint_to_hf(checkpoint_dir, hf_model_dir, base_model_path, tokenizer):
    """Convert a DeepSpeed ZeRO checkpoint to a loadable Hugging Face model dir."""
    fp32_path = os.path.join(os.path.dirname(hf_model_dir), "final_model_fp32.bin")
    if not convert_deepspeed_to_fp32(checkpoint_dir, fp32_path):
        return False

    print("Loading converted FP32 weights and saving HuggingFace format...")
    converted_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
    )
    state_dict = torch.load(fp32_path, map_location="cpu")
    converted_model.load_state_dict(state_dict)
    converted_model.save_pretrained(hf_model_dir)
    tokenizer.save_pretrained(hf_model_dir)

    del converted_model, state_dict
    torch.cuda.empty_cache()
    return has_hf_model_weights(hf_model_dir)


def save_stage_model(trainer, tokenizer, output_dir, base_model_path):
    """Save a stage result and return a path that AutoModel can load next stage."""
    final_model_dir = os.path.join(output_dir, "final_model")
    final_model_hf_dir = os.path.join(output_dir, "final_model_hf")

    os.makedirs(final_model_dir, exist_ok=True)
    if trainer.is_world_process_zero():
        print(f"\nSaving final model to {final_model_dir}...")

    # With DeepSpeed ZeRO-3, every rank must enter save_model because saving may
    # gather partitioned parameters through distributed collectives.
    trainer.save_model(final_model_dir)
    trainer.accelerator.wait_for_everyone()

    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(final_model_dir)

        if has_hf_model_weights(final_model_dir):
            print(f"Saved loadable HuggingFace model to: {final_model_dir}")
        else:
            print(
                "final_model does not contain HF weight files. "
                "Trying DeepSpeed ZeRO -> FP32 -> HF conversion..."
            )
            converted = convert_checkpoint_to_hf(
                checkpoint_dir=final_model_dir,
                hf_model_dir=final_model_hf_dir,
                base_model_path=base_model_path,
                tokenizer=tokenizer,
            )

            if not converted:
                latest_checkpoint = get_latest_checkpoint(output_dir)
                if latest_checkpoint is not None:
                    print(f"Trying conversion from latest Trainer checkpoint: {latest_checkpoint}")
                    converted = convert_checkpoint_to_hf(
                        checkpoint_dir=latest_checkpoint,
                        hf_model_dir=final_model_hf_dir,
                        base_model_path=base_model_path,
                        tokenizer=tokenizer,
                    )

            if not converted:
                raise RuntimeError(
                    "Could not produce a loadable HuggingFace model directory for the next stage. "
                    "DeepSpeed ZeRO-3 did not write model weights and zero_to_fp32 conversion failed. "
                    "Check that the checkpoint contains zero_to_fp32.py and global_step*."
                )

    trainer.accelerator.wait_for_everyone()

    if has_hf_model_weights(final_model_dir):
        return final_model_dir
    if has_hf_model_weights(final_model_hf_dir):
        return final_model_hf_dir

    raise RuntimeError(
        f"No loadable model weights found in {final_model_dir} or {final_model_hf_dir} after saving."
    )


def build_deepspeed_config(args):
    world_size = get_world_size()
    train_micro_batch_size_per_gpu = args.per_device_train_batch_size
    gradient_accumulation_steps = args.gradient_accumulation_steps
    train_batch_size = train_micro_batch_size_per_gpu * gradient_accumulation_steps * world_size

    return {
        "zero_optimization": {
            "stage": 3,  # ZeRO-3 最省显存
            "offload_optimizer": {
                "device": "cpu",  # 将优化器状态卸载到 CPU
                "pin_memory": True,
            },
            "offload_param": {
                "device": "cpu",  # 将参数卸载到 CPU
                "pin_memory": True,
            },
            "stage3_prefetch_bucket_size": 5e7,
            "stage3_param_persistence_threshold": 1e5,
            "stage3_max_live_parameters": 1e8,
            "stage3_max_reuse_distance": 1e8,
            "stage3_gather_16bit_weights_on_model_save": True,
        },
        "train_micro_batch_size_per_gpu": train_micro_batch_size_per_gpu,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "train_batch_size": train_batch_size,
    }


def run_single_stage(args, stage, student_model_path, ref_model_path):
    """运行单个蒸馏阶段"""
    print(f"\n{'='*60}")
    print(f"Starting Stage {stage}")
    print(f"{'='*60}")
    print(f"Student model: {student_model_path}")
    print(f"Reference model: {ref_model_path}")
    print(f"Teacher (SFT) model: {args.teacher_model_path}")
    print(f"Mixing ratio: {args.mixing_ratio}")
    print(f"{'='*60}\n")
    
    # 构建本阶段的输出目录和run_name
    run_name = (f'GKD+{args.model_name}+{args.data_name}+'
                f'batch_{args.per_device_train_batch_size}+'
                f'ga_{args.gradient_accumulation_steps}+'
                f'mix_{args.mixing_ratio}+lr_{args.lr}+{args.exp_name}_step{stage}')
    output_dir = os.path.join(args.output_base_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存配置
    save_args(args, output_dir, stage)
    
    # 加载模型
    model, teacher_model, tokenizer = load_model_and_tokenizer(
        teacher_model_path=args.teacher_model_path,
        student_model_path=student_model_path,
        ref_model_path=ref_model_path,
        mixing_ratio=args.mixing_ratio
    )
    
    deepspeed_config = build_deepspeed_config(args)
    
    # 加载数据集
    train_dataset = load_data(args.data_path)
    eval_dataset = load_data(args.data_path, data_num=200)
    
    # 配置训练参数
    training_args = GKDConfig(
        output_dir=output_dir,
        temperature=args.temperature,
        lmbda=args.lmbda,
        beta=args.beta,
        max_new_tokens=args.max_new_tokens,
        max_length=args.max_length,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        report_to="none",
        save_steps=args.save_steps,
        num_train_epochs=args.epochs_per_stage,
        run_name=run_name,
        bf16=True,
        learning_rate=args.lr,
        deepspeed=deepspeed_config,
        
        # gradient_checkpointing=True,
        # gradient_checkpointing_kwargs={"use_reentrant": False},  # 推荐设置
        # gradient_accumulation_steps=16,
        
        # use_liger_kernel=True,
        # torch_compile=True,
    )
    
    # 创建Trainer并训练
    trainer = GKDTrainer(
        model=model,
        teacher_model=teacher_model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    trainer.train()

    final_model_path = save_stage_model(
        trainer=trainer,
        tokenizer=tokenizer,
        output_dir=output_dir,
        base_model_path=student_model_path,
    )

    print(f"\n{'='*60}")
    print(f"Stage {stage} completed!")
    print(f"Final model: {final_model_path}")
    print(f"{'='*60}\n")

    # 清理显存
    del model, teacher_model, trainer
    torch.cuda.empty_cache()

    return final_model_path


def main():
    args = get_args()
    
    print(f"\n{'#'*60}")
    print(f"Iterative GKD Training")
    print(f"Total stages: {args.num_stages}")
    print(f"Epochs per stage: {args.epochs_per_stage}")
    print(f"Mixing ratio: {args.mixing_ratio}")
    print(f"{'#'*60}\n")
    
    # 初始的student和ref模型路径
    current_student_path = args.student_model_path
    current_ref_path = args.ref_model_path
    
    # 迭代训练
    for stage in range(1, args.num_stages + 1):
        # 运行当前阶段
        final_model_path = run_single_stage(
            args=args,
            stage=stage,
            student_model_path=current_student_path,
            ref_model_path=current_ref_path
        )

        if final_model_path is None:
            print(f"Warning: No model saved after stage {stage}. Stopping.")
            break

        # 下一阶段使用本阶段训练得到的模型作为student和ref
        current_student_path = final_model_path
        current_ref_path = final_model_path

        print(f"Next stage will use model from: {final_model_path}")
    
    print(f"\n{'#'*60}")
    print(f"All {args.num_stages} stages completed!")
    print(f"Final model: {current_student_path}")
    print(f"{'#'*60}\n")


if __name__ == "__main__":
    main()
