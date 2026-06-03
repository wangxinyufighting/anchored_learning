from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl.experimental.gkd import GKDConfig, GKDTrainer
import json
import torch
from datetime import datetime
import os
import argparse
import glob


class LogitsMixingModel(torch.nn.Module):
    def __init__(self, sft_model, ref_model, mixing_ratio):
        super().__init__()
        self.sft_model = sft_model
        self.ref_model = ref_model
        self.mixing_ratio = mixing_ratio
        self.config = sft_model.config

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        out_sft = self.sft_model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        out_ref = self.ref_model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        mixed_logits = self.mixing_ratio * out_sft.logits + (1.0 - self.mixing_ratio) * out_ref.logits
        out_sft.logits = mixed_logits
        return out_sft
    
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.sft_model, name)


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
    student_model = AutoModelForCausalLM.from_pretrained(
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
    
    # 混合teacher模型
    teacher_model = LogitsMixingModel(sft_model, ref_model, mixing_ratio)
    
    return student_model, teacher_model, tokenizer


def get_latest_checkpoint(output_dir):
    """获取output_dir中最新的checkpoint路径"""
    checkpoint_dirs = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    if not checkpoint_dirs:
        return None
    
    # 按checkpoint编号排序，取最大的
    checkpoint_dirs.sort(key=lambda x: int(x.split("-")[-1]))
    return checkpoint_dirs[-1]


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
    parser.add_argument('--epochs_per_stage', type=float, default=5, help='Number of training epochs per stage')
    parser.add_argument('--num_stages', type=int, default=10, help='Total number of iterative stages')
    parser.add_argument('--max_new_tokens', type=int, default=4096, help='Maximum number of new tokens to generate')
    parser.add_argument('--max_length', type=int, default=4096, help='Maximum sequence length for training')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for generation (recommended > 0 for on-policy)')
    parser.add_argument('--lmbda', type=float, default=1.0, help='Lambda for on-policy distillation (0=off-policy, >0=on-policy)')
    parser.add_argument('--beta', type=float, default=0, help='Beta parameter for KL regularization')
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--exp_name', type=str, default='iter', help='Experiment name prefix')
    parser.add_argument('--output_base_dir', type=str, default='./outputs', help='Base output directory')
    args = parser.parse_args()
    return args


def save_args(args, output_dir, stage):
    """保存当前阶段的参数配置"""
    args_dict = vars(args).copy()
    args_dict['current_stage'] = stage
    with open(os.path.join(output_dir, 'args.json'), 'w') as f:
        json.dump(args_dict, f, ensure_ascii=False, indent=2)


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
                f't_{args.temperature}+β_{args.beta}+λ_{args.lmbda}+'
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
        report_to="none",
        # save_steps=args.save_steps,
        save_strategy="epoch",
        num_train_epochs=args.epochs_per_stage,
        run_name=run_name,
        bf16=True,
        learning_rate=args.lr,
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
    
    # 获取本阶段训练完成后的最新checkpoint
    latest_checkpoint = get_latest_checkpoint(output_dir)
    
    print(f"\n{'='*60}")
    print(f"Stage {stage} completed!")
    print(f"Latest checkpoint: {latest_checkpoint}")
    print(f"{'='*60}\n")
    
    # 清理显存
    del model, teacher_model, trainer
    torch.cuda.empty_cache()
    
    return latest_checkpoint


def main():
    args = get_args()

    print(f"\n{'#'*60}")
    print(f"Iterative GKD Training")
    print(f"Total stages: {args.num_stages}")
    print(f"Epochs per stage: {args.epochs_per_stage}")
    print(f"Mixing ratio: {args.mixing_ratio}")
    print(f"Mode: {'On-policy' if args.lmbda > 0 else 'Off-policy'} (λ={args.lmbda})")
    print(f"Temperature: {args.temperature}")
    print(f"{'#'*60}\n")
    
    # 初始的student和ref模型路径
    current_student_path = args.student_model_path
    current_ref_path = args.ref_model_path
    
    # 迭代训练
    for stage in range(1, args.num_stages + 1):
        # 运行当前阶段
        latest_checkpoint = run_single_stage(
            args=args,
            stage=stage,
            student_model_path=current_student_path,
            ref_model_path=current_ref_path
        )
        
        if latest_checkpoint is None:
            print(f"Warning: No checkpoint found after stage {stage}. Stopping.")
            break
        
        # 下一阶段使用本阶段训练得到的模型作为student和ref
        current_student_path = latest_checkpoint
        current_ref_path = latest_checkpoint
        
        print(f"Next stage will use model from: {latest_checkpoint}")
    
    print(f"\n{'#'*60}")
    print(f"All {args.num_stages} stages completed!")
    print(f"Final model: {current_student_path}")
    print(f"{'#'*60}\n")


if __name__ == "__main__":
    main()