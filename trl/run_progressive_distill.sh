#!/bin/bash
#
# Run Progressive On-Policy Distillation
#
# IMPORTANT: Start the teacher server FIRST before running this script!
#   bash start_teacher_server.sh
#
# This script runs the student training with DistillationTrainer + vLLM
#

set -e

# ============================================================================
# Configuration - MODIFY THESE
# ============================================================================

# Model paths
STUDENT_MODEL="/path/to/your/base_model"  # Initial student = base model
TEACHER_SERVER_URL="http://localhost:8000"  # Teacher server URL

# Data
DATA_PATH="/path/to/your/training_data.json"  # LLaMA-Factory format
DATA_NAME="your_dataset_name"

# Progressive distillation
NUM_STAGES=10
EPOCHS_PER_STAGE=1
MIXING_RATIO=0.5  # Must match teacher server config

# Training hyperparameters
PER_DEVICE_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=4
LR=1e-5
TEMPERATURE=0.9

# Distillation parameters
LMBDA=1.0  # 1.0 = fully on-policy, 0.0 = off-policy
BETA=1.0   # 1.0 = reverse KL, 0.0 = forward KL, 0.5 = JSD
LOSS_TOP_K=1

# Sequence lengths
MAX_LENGTH=2048
MAX_PROMPT_LENGTH=1536
MAX_COMPLETION_LENGTH=512

# vLLM for student generation (highly recommended for speed)
USE_VLLM="--use_vllm"
VLLM_MODE="colocate"  # or "server" if you start separate vllm-serve
VLLM_GPU_MEMORY=0.3
VLLM_TENSOR_PARALLEL=1

# GPU configuration for student training
# Example: Use GPUs 2-7 for student (GPUs 0-1 reserved for teacher)
STUDENT_GPUS="2,3,4,5,6,7"
NPROC_PER_NODE=6  # Number of GPUs for student training

# Output
MODEL_NAME="Qwen3-4B"
EXP_NAME="progressive_onpolicy"
OUTPUT_DIR="./outputs"

# ============================================================================
# Pre-flight checks
# ============================================================================

echo "========================================="
echo "Progressive On-Policy Distillation"
echo "========================================="
echo "Student model:   $STUDENT_MODEL"
echo "Teacher server:  $TEACHER_SERVER_URL"
echo "Data:            $DATA_PATH"
echo "Stages:          $NUM_STAGES"
echo "Epochs/stage:    $EPOCHS_PER_STAGE"
echo "Mixing ratio:    $MIXING_RATIO"
echo "Lambda:          $LMBDA"
echo "GPUs:            $STUDENT_GPUS"
echo "vLLM:            $([ -n "$USE_VLLM" ] && echo 'enabled' || echo 'disabled')"
echo "========================================="
echo

# Check if teacher server is running
echo "Checking teacher server at $TEACHER_SERVER_URL..."
if ! curl -s --max-time 5 "$TEACHER_SERVER_URL/health" > /dev/null 2>&1; then
    echo "ERROR: Teacher server is not responding!"
    echo "Please start the teacher server first:"
    echo "  bash start_teacher_server.sh"
    exit 1
fi
echo "✓ Teacher server is running"
echo

# ============================================================================
# Launch Training
# ============================================================================

export CUDA_VISIBLE_DEVICES=$STUDENT_GPUS

torchrun \
    --nproc_per_node=$NPROC_PER_NODE \
    --master_port=29500 \
    progressive_distillation.py \
    --student_model_path "$STUDENT_MODEL" \
    --teacher_server_url "$TEACHER_SERVER_URL" \
    --data_path "$DATA_PATH" \
    --data_name "$DATA_NAME" \
    --num_stages $NUM_STAGES \
    --epochs_per_stage $EPOCHS_PER_STAGE \
    --mixing_ratio $MIXING_RATIO \
    --per_device_train_batch_size $PER_DEVICE_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --lr $LR \
    --temperature $TEMPERATURE \
    --lmbda $LMBDA \
    --beta $BETA \
    --loss_top_k $LOSS_TOP_K \
    --max_length $MAX_LENGTH \
    --max_prompt_length $MAX_PROMPT_LENGTH \
    --max_completion_length $MAX_COMPLETION_LENGTH \
    $USE_VLLM \
    --vllm_mode $VLLM_MODE \
    --vllm_gpu_memory_utilization $VLLM_GPU_MEMORY \
    --vllm_tensor_parallel_size $VLLM_TENSOR_PARALLEL \
    --bf16 true \
    --use_flash_attention \
    --gradient_checkpointing \
    --save_strategy epoch \
    --save_total_limit 2 \
    --logging_steps 10 \
    --model_name "$MODEL_NAME" \
    --exp_name "$EXP_NAME" \
    --output_base_dir "$OUTPUT_DIR" \
    --prompt_only  # Remove this if your dataset has gold completions you want to use

echo
echo "========================================="
echo "Training completed!"
echo "========================================="
