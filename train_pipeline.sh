#!/bin/bash
set -e
set -x

# ============================================
# Configuration
# ============================================
model_name=Qwen3-4B
data_name=medcalc_train
base_model_path=/root/autodl-tmp/models/${model_name}
data_path=./LlamaFactory/data/${data_name}.json

# SFT parameters
sft_lr=1e-6
sft_epochs=0.05
sft_batch_size=4
sft_output_dir=./saves/${model_name}_${data_name}_${sft_lr}

# Anchored learning parameters
al_lr=1e-5
al_mixing_ratio=0.5
al_num_stages=10
al_epochs_per_stage=5
al_batch_size=4
al_temperature=0.9
al_lmbda=1.0  # on-policy

# GPU configuration
export CUDA_VISIBLE_DEVICES=0,1
NUM_GPUS=2

# ============================================
# Resolve absolute paths BEFORE any cd
# ============================================
REPO_ROOT=$(pwd)
data_path_abs=$(realpath ${data_path})
echo "Repo root: ${REPO_ROOT}"
echo "Data path (abs): ${data_path_abs}"

# ============================================
# Step 1: Run SFT Training
# ============================================
echo "========================================"
echo "Step 1: Starting SFT Training"
echo "========================================"

cd ${REPO_ROOT}/LlamaFactory
llamafactory-cli train examples/train_full/qwen3_full_sft.yaml \
    model_name_or_path=${base_model_path} \
    output_dir=${sft_output_dir} \
    learning_rate=${sft_lr} \
    num_train_epochs=${sft_epochs} \
    cutoff_len=4096 \
    per_device_train_batch_size=${sft_batch_size} \
    dataset=${data_name}

echo "SFT Training completed!"

# Go back to repo root after SFT
cd ${REPO_ROOT}


# ============================================
# Step 2: Find Last SFT Checkpoint
# ============================================
echo "========================================"
echo "Step 2: Finding Last SFT Checkpoint"
echo "========================================"

sft_output_dir=${REPO_ROOT}/LlamaFactory/${sft_output_dir}

last_checkpoint=$(ls -d ${sft_output_dir}/checkpoint-* 2>/dev/null | sort -V | tail -n 1)

if [ -z "$last_checkpoint" ]; then
    echo "ERROR: No checkpoint found in ${sft_output_dir}"
    exit 1
fi

echo "Found last checkpoint: ${last_checkpoint}"

# Resolve to absolute path (now from REPO_ROOT, no doubling)
last_checkpoint_abs=$(realpath ${last_checkpoint})

echo "Resolved absolute checkpoint path: ${last_checkpoint_abs}"
echo "Resolved absolute data path: ${data_path_abs}"

# ============================================
# Step 3: Run Anchored Learning
# ============================================
echo "========================================"
echo "Step 3: Starting Anchored Learning"
echo "========================================"

cd ${REPO_ROOT}/trl
torchrun --nproc_per_node=${NUM_GPUS} anchored_learning.py \
    --teacher_model_path ${last_checkpoint_abs} \
    --student_model_path ${base_model_path} \
    --ref_model_path ${base_model_path} \
    --data_path ${data_path_abs} \
    --data_name ${data_name} \
    --epochs_per_stage ${al_epochs_per_stage} \
    --num_stages ${al_num_stages} \
    --mixing_ratio ${al_mixing_ratio} \
    --lr ${al_lr} \
    --temperature ${al_temperature} \
    --lmbda ${al_lmbda} \
    --per_device_train_batch_size ${al_batch_size} \
    --exp_name epoch_${al_epochs_per_stage}_stage_${al_num_stages}_on_policy

# ============================================
# Done
# ============================================
echo "========================================"
echo "Pipeline Completed Successfully!"
echo "========================================"
echo "SFT Checkpoint: ${last_checkpoint_abs}"
echo "Anchored Learning output: Check trl/outputs directory"