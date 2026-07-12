import argparse
import json
import logging
import os

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from trl.experimental.gkd import GKDConfig, GKDTrainer


logger = logging.getLogger(__name__)


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


def load_data(data_path):
    with open(data_path, encoding="utf-8") as file:
        all_data = json.load(file)

    if not isinstance(all_data, list) or not all_data:
        raise ValueError(f"Dataset must be a non-empty JSON list: {data_path}")

    messages = []
    for index, example in enumerate(all_data):
        try:
            instruction = example["instruction"]
            additional_input = example.get("input") or ""
            output = example["output"]
        except (AttributeError, KeyError, TypeError) as error:
            raise ValueError(f"Invalid example at index {index}: expected instruction/input/output fields") from error

        if not all(isinstance(value, str) for value in (instruction, additional_input, output)):
            raise ValueError(f"Invalid example at index {index}: instruction/input/output must be strings")
        if not instruction.strip() or not output.strip():
            raise ValueError(f"Invalid example at index {index}: instruction and output must not be blank")

        user_content = instruction if not additional_input.strip() else f"{instruction}\n{additional_input}"
        messages.append(
            [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": output},
            ]
        )

    return Dataset.from_dict({"messages": messages})


def load_models(teacher_model_path, student_model_path, ref_model_path, mixing_ratio):
    model_kwargs = {"torch_dtype": torch.bfloat16}
    student_model = AutoModelForCausalLM.from_pretrained(student_model_path, **model_kwargs)
    sft_model = AutoModelForCausalLM.from_pretrained(teacher_model_path, **model_kwargs)
    ref_model = AutoModelForCausalLM.from_pretrained(ref_model_path, **model_kwargs)

    vocab_sizes = {
        "student": student_model.config.vocab_size,
        "teacher": sft_model.config.vocab_size,
        "reference": ref_model.config.vocab_size,
    }
    if len(set(vocab_sizes.values())) != 1:
        raise ValueError(f"Student, teacher, and reference vocab sizes must match: {vocab_sizes}")

    sft_model.requires_grad_(False).eval()
    ref_model.requires_grad_(False).eval()
    teacher_model = LogitsMixingModel(sft_model, ref_model, mixing_ratio)
    return student_model, teacher_model


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen2.5-3B-Instruct", help="Model name")
    parser.add_argument("--data_name", type=str, default="medcalc_train", help="Dataset name")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--teacher_model_path", type=str, required=True, help="Path to the teacher (SFT) model")
    parser.add_argument("--ref_model_path", type=str, required=True, help="Path to the initial reference model")
    parser.add_argument("--student_model_path", type=str, required=True, help="Path to the initial student model")
    parser.add_argument("--mixing_ratio", type=float, default=0.2, help="Weight for SFT model logits")
    parser.add_argument("--save_steps", type=int, default=500, help="Save interval in optimizer steps")
    parser.add_argument("--save_total_limit", type=int, default=2, help="Maximum number of checkpoints per stage")
    parser.add_argument("--per_device_train_batch_size", type=int, default=3)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--epochs_per_stage", type=float, default=5)
    parser.add_argument("--num_stages", type=int, default=10)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--lmbda", type=float, default=0)
    parser.add_argument("--beta", type=float, default=0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--exp_name", type=str, default="iter", help="Experiment name prefix")
    parser.add_argument("--output_base_dir", type=str, default="./outputs")
    args = parser.parse_args()

    if not 0.0 <= args.mixing_ratio <= 1.0:
        parser.error("--mixing_ratio must be in [0, 1]")
    if not 0.0 <= args.lmbda <= 1.0:
        parser.error("--lmbda must be in [0, 1]")
    if not 0.0 <= args.beta <= 1.0:
        parser.error("--beta must be in [0, 1]")
    if args.lmbda > 0 and args.temperature <= 0:
        parser.error("--temperature must be greater than 0 for on-policy training")
    if args.num_stages < 1 or args.epochs_per_stage <= 0:
        parser.error("--num_stages and --epochs_per_stage must be greater than 0")
    if args.per_device_train_batch_size < 1 or args.gradient_accumulation_steps < 1:
        parser.error("batch size and gradient accumulation steps must be greater than 0")
    if args.save_steps < 1 or args.save_total_limit < 1:
        parser.error("--save_steps and --save_total_limit must be greater than 0")
    if not 0.0 <= args.warmup_ratio < 1.0:
        parser.error("--warmup_ratio must be in [0, 1)")

    return args


def is_main_process():
    return os.environ.get("RANK", "0") == "0"


def save_args(args, output_dir, stage):
    args_dict = vars(args).copy()
    args_dict["current_stage"] = stage
    with open(os.path.join(output_dir, "args.json"), "w", encoding="utf-8") as file:
        json.dump(args_dict, file, ensure_ascii=False, indent=2)


def run_single_stage(args, stage, student_model_path, ref_model_path, tokenizer, train_dataset):
    model_name = args.model_name.replace("/", "_")
    data_name = args.data_name.replace("/", "_")
    exp_name = args.exp_name.replace("/", "_")
    run_name = (
        f"GKD+{model_name}+{data_name}+batch_{args.per_device_train_batch_size}+"
        f"t_{args.temperature}+beta_{args.beta}+lambda_{args.lmbda}+mix_{args.mixing_ratio}+"
        f"lr_{args.lr}+warmup_{args.warmup_ratio}+{exp_name}_stage{stage}"
    )
    output_dir = os.path.join(args.output_base_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)

    if is_main_process():
        logger.info("\n%s", "=" * 60)
        logger.info("Starting Stage %s/%s", stage, args.num_stages)
        logger.info("Student model: %s", student_model_path)
        logger.info("Reference model: %s", ref_model_path)
        logger.info("Teacher (SFT) model: %s", args.teacher_model_path)
        logger.info("Mixing ratio: %s", args.mixing_ratio)
        logger.info("%s\n", "=" * 60)
        save_args(args, output_dir, stage)

    model, teacher_model = load_models(
        teacher_model_path=args.teacher_model_path,
        student_model_path=student_model_path,
        ref_model_path=ref_model_path,
        mixing_ratio=args.mixing_ratio,
    )

    training_args = GKDConfig(
        output_dir=output_dir,
        temperature=args.temperature,
        lmbda=args.lmbda,
        beta=args.beta,
        max_new_tokens=args.max_new_tokens,
        max_length=args.max_length,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if args.gradient_checkpointing else None,
        dataloader_num_workers=args.dataloader_num_workers,
        report_to="none",
        eval_strategy="no",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        num_train_epochs=args.epochs_per_stage,
        run_name=run_name,
        bf16=True,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=1.0,
        logging_nan_inf_filter=False,
    )

    trainer = GKDTrainer(
        model=model,
        teacher_model=teacher_model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=train_dataset,
    )
    trainer.train()

    stage_final_dir = os.path.join(output_dir, "stage-final")
    trainer.save_model(stage_final_dir)
    trainer.accelerator.wait_for_everyone()

    if trainer.is_world_process_zero():
        logger.info("\n%s", "=" * 60)
        logger.info("Stage %s completed", stage)
        logger.info("Final model: %s", stage_final_dir)
        logger.info("%s\n", "=" * 60)

    del model, teacher_model, trainer
    torch.cuda.empty_cache()
    return stage_final_dir


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = get_args()
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model_path)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer must define either a pad token or an EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    train_dataset = load_data(args.data_path)

    if is_main_process():
        logger.info("\n%s", "#" * 60)
        logger.info("Iterative GKD Training")
        logger.info("Total stages: %s", args.num_stages)
        logger.info("Epochs per stage: %s", args.epochs_per_stage)
        logger.info("Mixing ratio: %s", args.mixing_ratio)
        logger.info("%s\n", "#" * 60)

    current_student_path = args.student_model_path
    current_ref_path = args.ref_model_path

    for stage in range(1, args.num_stages + 1):
        stage_final_dir = run_single_stage(
            args=args,
            stage=stage,
            student_model_path=current_student_path,
            ref_model_path=current_ref_path,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
        )
        current_student_path = stage_final_dir
        current_ref_path = stage_final_dir

    if is_main_process():
        logger.info("\n%s", "#" * 60)
        logger.info("All %s stages completed", args.num_stages)
        logger.info("Final model: %s", current_student_path)
        logger.info("%s\n", "#" * 60)


if __name__ == "__main__":
    main()
