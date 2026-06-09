import argparse
import glob
import json
import os
from datetime import datetime

import torch
from datasets import Dataset
from transformers import AutoTokenizer

try:
    from trl.experimental.distillation import DistillationConfig, DistillationTrainer
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "DistillationTrainer is in the experimental TRL namespace. "
        "Install a recent TRL from source or a version that contains "
        "trl.experimental.distillation, for example: pip install -U 'trl[vllm]'."
    ) from exc


def load_data(data_path, data_num=None, prompt_only=False):
    """Load LLaMA-Factory style JSON into TRL conversational LM format.

    Expected input item fields: instruction, input, output.
    For fully on-policy distillation (lmbda=1.0), assistant turns are optional;
    prompt_only=True removes them to avoid carrying unused gold completions.
    """
    with open(data_path, "r", encoding="utf-8") as f:
        all_data = json.load(f)

    if data_num is not None:
        all_data = all_data[:data_num]

    messages = []
    for d in all_data:
        prompt = f"{d.get('instruction', '')}{d.get('input', '')}"
        turns = [{"role": "user", "content": prompt}]
        if not prompt_only:
            turns.append({"role": "assistant", "content": d.get("output", "")})
        messages.append(turns)

    return Dataset.from_dict({"messages": messages})


def get_latest_checkpoint(output_dir):
    checkpoint_dirs = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    if not checkpoint_dirs:
        return None
    checkpoint_dirs.sort(key=lambda x: int(x.rsplit("-", 1)[-1]))
    return checkpoint_dirs[-1]


def str2bool(x):
    if isinstance(x, bool):
        return x
    return str(x).lower() in {"1", "true", "yes", "y"}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen3-4B")
    parser.add_argument("--data_name", type=str, default="medcalc_train")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--teacher_model_path", type=str, required=True)
    parser.add_argument("--student_model_path", type=str, required=True)

    # Kept for backward compatibility with your old script. DistillationTrainer's
    # external teacher server supports a single teacher distribution, so logit
    # mixing with a moving ref model is intentionally disabled here.
    parser.add_argument("--ref_model_path", type=str, default=None)
    parser.add_argument("--mixing_ratio", type=float, default=1.0)

    parser.add_argument("--epochs_per_stage", type=float, default=1)
    parser.add_argument("--num_stages", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--dataloader_num_workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.0)
    parser.add_argument("--weight_decay", type=float, default=0.0)

    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--max_prompt_length", type=int, default=1536)
    parser.add_argument("--max_completion_length", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--num_generations", type=int, default=1)
    parser.add_argument("--generation_batch_size", type=int, default=None)

    parser.add_argument("--lmbda", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--loss_top_k", type=int, default=1)
    parser.add_argument("--loss_add_tail", type=str2bool, default=True)

    parser.add_argument("--bf16", type=str2bool, default=True)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--use_flash_attention", action="store_true")
    parser.add_argument("--use_liger_kernel", action="store_true")

    # Teacher vLLM server. This server hosts the fixed teacher and returns top-k logprobs.
    parser.add_argument("--use_teacher_server", action="store_true")
    parser.add_argument("--teacher_model_server_url", type=str, default="http://127.0.0.1:8000")

    # Student vLLM integration. Server mode requires starting `trl vllm-serve`
    # with the initial student model on separate GPUs.
    parser.add_argument("--use_vllm", action="store_true")
    parser.add_argument("--vllm_mode", type=str, default="server", choices=["server", "colocate"])
    parser.add_argument("--vllm_server_base_url", type=str, default="http://127.0.0.1:8001")
    parser.add_argument("--vllm_server_timeout", type=float, default=600.0)
    parser.add_argument("--vllm_sync_frequency", type=int, default=1)
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.3)
    parser.add_argument("--vllm_tensor_parallel_size", type=int, default=1)
    parser.add_argument("--vllm_max_model_length", type=int, default=None)
    parser.add_argument("--vllm_enable_sleep_mode", action="store_true")

    parser.add_argument("--save_strategy", type=str, default="no")
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_data_num", type=int, default=200)
    parser.add_argument("--prompt_only", action="store_true")
    parser.add_argument("--exp_name", type=str, default="distill_vllm")
    parser.add_argument("--output_base_dir", type=str, default="./outputs")
    return parser.parse_args()


def save_args(args, output_dir, stage):
    args_dict = vars(args).copy()
    args_dict["current_stage"] = stage
    with open(os.path.join(output_dir, "args.json"), "w", encoding="utf-8") as f:
        json.dump(args_dict, f, ensure_ascii=False, indent=2)


def build_model_init_kwargs(args):
    kwargs = {"torch_dtype": torch.bfloat16 if args.bf16 else torch.float16}
    if args.use_flash_attention:
        kwargs["attn_implementation"] = "flash_attention_2"
    return kwargs


def run_single_stage(args, stage, student_model_path):
    if args.mixing_ratio != 1.0:
        print(
            "[Warning] DistillationTrainer + external teacher server supports a single "
            "teacher distribution. The old SFT/ref logits mixing is not used in this "
            "script. Set --mixing_ratio 1.0 or implement a custom teacher server if "
            "you need mixed teacher logits."
        )

    run_name = (
        f"Distill+{args.model_name}+{args.data_name}+"
        f"bs_{args.per_device_train_batch_size}+ga_{args.gradient_accumulation_steps}+"
        f"t_{args.temperature}+beta_{args.beta}+lambda_{args.lmbda}+"
        f"topk_{args.loss_top_k}+lr_{args.lr}+{args.exp_name}_stage{stage}"
    )
    output_dir = os.path.join(args.output_base_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)
    save_args(args, output_dir, stage)

    prompt_only = args.prompt_only or args.lmbda >= 1.0
    train_dataset = load_data(args.data_path, prompt_only=prompt_only)
    eval_dataset = load_data(args.data_path, data_num=args.eval_data_num, prompt_only=prompt_only)

    tokenizer = AutoTokenizer.from_pretrained(student_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_init_kwargs = build_model_init_kwargs(args)
    teacher_model_init_kwargs = build_model_init_kwargs(args)

    config = DistillationConfig(
        output_dir=output_dir,
        report_to="none",
        run_name=run_name,
        bf16=args.bf16,
        model_init_kwargs=model_init_kwargs,
        gradient_checkpointing=args.gradient_checkpointing,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        dataloader_num_workers=args.dataloader_num_workers,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        num_train_epochs=args.epochs_per_stage,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        num_generations=args.num_generations,
        generation_batch_size=args.generation_batch_size,
        lmbda=args.lmbda,
        beta=args.beta,
        loss_top_k=args.loss_top_k,
        loss_add_tail=args.loss_add_tail,
        use_liger_kernel=args.use_liger_kernel,
        teacher_model_name_or_path=None if args.use_teacher_server else args.teacher_model_path,
        teacher_model_init_kwargs=teacher_model_init_kwargs,
        use_teacher_server=args.use_teacher_server,
        teacher_model_server_url=args.teacher_model_server_url if args.use_teacher_server else None,
        use_vllm=args.use_vllm,
        vllm_mode=args.vllm_mode,
        vllm_server_base_url=args.vllm_server_base_url if args.use_vllm and args.vllm_mode == "server" else None,
        vllm_server_timeout=args.vllm_server_timeout,
        vllm_sync_frequency=args.vllm_sync_frequency,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
        vllm_max_model_length=args.vllm_max_model_length,
        vllm_enable_sleep_mode=args.vllm_enable_sleep_mode,
    )

    trainer = DistillationTrainer(
        model=student_model_path,
        teacher_model=None if args.use_teacher_server else args.teacher_model_path,
        args=config,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    print(f"\n{'=' * 80}")
    print(f"Stage {stage}/{args.num_stages}")
    print(f"student_model_path={student_model_path}")
    print(f"teacher={'server ' + args.teacher_model_server_url if args.use_teacher_server else args.teacher_model_path}")
    print(f"student_vllm={args.use_vllm}, mode={args.vllm_mode}, url={args.vllm_server_base_url}")
    print(f"max_length={args.max_length}, max_prompt_length={args.max_prompt_length}, max_completion_length={args.max_completion_length}")
    print(f"lmbda={args.lmbda}, beta={args.beta}, loss_top_k={args.loss_top_k}, generation_batch_size={args.generation_batch_size}")
    print(f"{'=' * 80}\n")

    trainer.train()

    stage_final_dir = os.path.join(output_dir, "stage-final")
    trainer.save_model(stage_final_dir)
    tokenizer.save_pretrained(stage_final_dir)

    del trainer
    torch.cuda.empty_cache()
    return stage_final_dir


def main():
    args = get_args()

    # External server restrictions documented by TRL.
    if args.use_teacher_server and args.beta > 0 and args.loss_top_k != 1:
        raise ValueError("With teacher server and beta > 0, DistillationTrainer requires --loss_top_k 1.")
    if args.use_teacher_server and args.beta == 0 and args.loss_top_k <= 0:
        raise ValueError("With teacher server and beta == 0, DistillationTrainer requires --loss_top_k > 0.")

    print(f"\n{'#' * 80}")
    print("DistillationTrainer + vLLM server training")
    print(f"TRL on-policy lambda={args.lmbda}; num_generations={args.num_generations}")
    print(f"Total stages={args.num_stages}; epochs_per_stage={args.epochs_per_stage}")
    print(f"Started at {datetime.now().isoformat(timespec='seconds')}")
    print(f"{'#' * 80}\n")

    current_student_path = args.student_model_path
    for stage in range(1, args.num_stages + 1):
        current_student_path = run_single_stage(args, stage, current_student_path)
        print(f"Next stage student: {current_student_path}")

    print(f"\nFinal model: {current_student_path}\n")


if __name__ == "__main__":
    main()
