#!/usr/bin/env bash
set -euo pipefail

# This script assumes 8 GPUs on one node.
# GPU allocation:
#   0,1: fixed teacher vLLM server for teacher logprobs
#   2,3: student vLLM server for on-policy generation
#   4,5,6,7: trainer DDP workers
# For Qwen3-4B on H100 this is a safe starting point. Adjust tensor parallel sizes if needed.

model_name=Qwen3-4B
data_name=medcalc_train
lr=1e-5
num_stages=1
epochs_per_stage=1

TEACHER_MODEL=/root/autodl-tmp/LlamaFactory/saves/${model_name}_${data_name}_1e-6/checkpoint-4410
STUDENT_MODEL=/root/autodl-tmp/models/${model_name}
DATA_PATH=/root/autodl-tmp/LlamaFactory/data/${data_name}.json

TEACHER_PORT=8000
STUDENT_PORT=8001

# 1) Start fixed teacher server.
CUDA_VISIBLE_DEVICES=0,1 trl vllm-serve \
  --model "${TEACHER_MODEL}" \
  --tensor-parallel-size 2 \
  --host 0.0.0.0 \
  --port ${TEACHER_PORT} \
  --dtype bfloat16 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.85 \
  --trust-remote-code \
  > teacher_vllm.log 2>&1 &
TEACHER_PID=$!

# 2) Start student generation server. TRL will sync updated student weights into this server.
CUDA_VISIBLE_DEVICES=2,3 trl vllm-serve \
  --model "${STUDENT_MODEL}" \
  --tensor-parallel-size 2 \
  --host 0.0.0.0 \
  --port ${STUDENT_PORT} \
  --dtype bfloat16 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.85 \
  --trust-remote-code \
  > student_vllm.log 2>&1 &
STUDENT_PID=$!

cleanup() {
  kill ${TEACHER_PID} ${STUDENT_PID} 2>/dev/null || true
}
trap cleanup EXIT

# Give the servers time to initialize. Increase this if your checkpoint is slow to load.
sleep 60

# 3) Train on separate GPUs. Do not overlap these GPUs with either vLLM server.
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 anchored_learning_distill_vllm.py \
  --teacher_model_path "${TEACHER_MODEL}" \
  --student_model_path "${STUDENT_MODEL}" \
  --data_path "${DATA_PATH}" \
  --model_name "${model_name}" \
  --data_name "${data_name}" \
  --epochs_per_stage ${epochs_per_stage} \
  --num_stages ${num_stages} \
  --lr ${lr} \
  --temperature 0.9 \
  --lmbda 1.0 \
  --beta 1.0 \
  --loss_top_k 1 \
  --num_generations 1 \
  --generation_batch_size 8 \
  --max_length 2048 \
  --max_prompt_length 1536 \
  --max_completion_length 256 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --gradient_checkpointing \
  --use_flash_attention \
  --use_teacher_server \
  --teacher_model_server_url http://127.0.0.1:${TEACHER_PORT} \
  --use_vllm \
  --vllm_mode server \
  --vllm_server_base_url http://127.0.0.1:${STUDENT_PORT} \
  --vllm_sync_frequency 1 \
  --dataloader_num_workers 8 \
  --save_strategy no \
  --exp_name distill_vllm_debug
