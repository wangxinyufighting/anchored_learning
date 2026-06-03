#!/bin/bash
set -e  # Exit on error
set -x  # Print commands

# ============================================
# Configuration
# ============================================
model_name=Qwen3-4B
data_name=medcalc_train
base_model_path=/root/autodl-tmp/models/${model_name} # 替换目录路径为实际路径
data_path=./LlamaFactory/data/${data_name}.json

# SFT parameters
sft_lr=1e-6
sft_epochs=15
sft_batch_size=16
sft_output_dir=./LlamaFactory/saves/${model_name}_${data_name}_${sft_lr}

# Anchored learning parameters
al_lr=1e-5
al_mixing_ratio=0.5
al_num_stages=10
al_epochs_per_stage=5
al_batch_size=2
al_temperature=0.9
al_lmbda=1.0 #on-policy

# GPU configuration
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NUM_GPUS=8

# ============================================
# Step 1: Run SFT Training
# ============================================
echo "========================================"
echo "Step 1: Starting SFT Training"
echo "========================================"

cd LlamaFactory

llamafactory-cli train examples/train_full/qwen3_full_sft.yaml \
    model_name_or_path=${base_model_path} \
    output_dir=${sft_output_dir} \
    learning_rate=${sft_lr} \
    num_train_epochs=${sft_epochs} \
    cutoff_len=4096 \
    per_device_train_batch_size=${sft_batch_size} \
    dataset=${data_name}

echo "SFT Training completed!"

# ============================================
# Step 2: Find Last SFT Checkpoint
# ============================================
echo "========================================"
echo "Step 2: Finding Last SFT Checkpoint"
echo "========================================"

# Find the latest checkpoint directory
last_checkpoint=$(ls -d ${sft_output_dir}/checkpoint-* 2>/dev/null | sort -V | tail -n 1)

if [ -z "$last_checkpoint" ]; then
    echo "ERROR: No checkpoint found in ${sft_output_dir}"
    exit 1
fi

echo "Found last checkpoint: ${last_checkpoint}"

# ============================================
# Step 3: Run Anchored Learning
# ============================================
echo "========================================"
echo "Step 3: Starting Anchored Learning"
echo "========================================"

cd ../trl

torchrun --nproc_per_node=${NUM_GPUS} anchored_learning.py \
    --teacher_model_path ${last_checkpoint} \
    --student_model_path ${base_model_path} \
    --ref_model_path ${base_model_path} \
    --data_path ${data_path} \
    --data_name ${data_name} \
    --epochs_per_stage ${al_epochs_per_stage} \
    --num_stages ${al_num_stages} \
    --mixing_ratio ${al_mixing_ratio} \
    --lr ${al_lr} \
    --temperature ${al_temperature} \
    --lmbda ${al_lmbda} \
    --per_device_train_batch_size ${al_batch_size} \
    --exp_name epoch_${al_epochs_per_stage}_stage_${al_num_stages}_on_policy

echo "========================================"
echo "Pipeline Completed Successfully!"
echo "========================================"
echo "SFT Checkpoint: ${last_checkpoint}"
echo "Anchored Learning output: Check trl/outputs directory"
