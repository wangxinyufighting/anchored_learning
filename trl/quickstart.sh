#!/bin/bash
#
# Quick Start Example for Progressive Distillation
#
# This is a minimal working example with common settings.
# Customize the paths and run!
#

# ============================================================================
# STEP 1: Edit these paths
# ============================================================================

# Your model paths
SFT_MODEL="/path/to/qwen3-4b-sft"          # Trained SFT model
BASE_MODEL="/path/to/qwen3-4b-base"        # Base model (pretrained)
TRAIN_DATA="/path/to/your_data.json"      # Training data

# GPU allocation (example for 8 GPUs)
TEACHER_GPUS="0,1"      # 2 GPUs for teacher
STUDENT_GPUS="2,3,4,5,6,7"  # 6 GPUs for student

# ============================================================================
# STEP 2: Start teacher server (in terminal 1)
# ============================================================================

start_teacher() {
    echo "Starting teacher server on GPUs $TEACHER_GPUS..."

    CUDA_VISIBLE_DEVICES=$TEACHER_GPUS python progressive_teacher_server.py \
        --sft-model "$SFT_MODEL" \
        --initial-ref-model "$BASE_MODEL" \
        --mixing-ratio 0.5 \
        --port 8000 \
        --tensor-parallel-size 2 \
        --gpu-memory-utilization 0.85 \
        --trust-remote-code
}

# ============================================================================
# STEP 3: Run training (in terminal 2)
# ============================================================================

run_training() {
    echo "Starting progressive distillation on GPUs $STUDENT_GPUS..."

    # Wait for teacher to be ready
    echo "Waiting for teacher server..."
    for i in {1..30}; do
        if curl -s http://localhost:8000/health > /dev/null 2>&1; then
            echo "✓ Teacher server ready!"
            break
        fi
        sleep 2
    done

    CUDA_VISIBLE_DEVICES=$STUDENT_GPUS torchrun \
        --nproc_per_node=6 \
        progressive_distillation.py \
        --student_model_path "$BASE_MODEL" \
        --teacher_server_url "http://localhost:8000" \
        --data_path "$TRAIN_DATA" \
        --data_name "quickstart" \
        --num_stages 10 \
        --epochs_per_stage 1 \
        --mixing_ratio 0.5 \
        --per_device_train_batch_size 2 \
        --gradient_accumulation_steps 2 \
        --lr 1e-5 \
        --temperature 0.9 \
        --lmbda 1.0 \
        --beta 1.0 \
        --loss_top_k 1 \
        --max_length 2048 \
        --max_prompt_length 1536 \
        --max_completion_length 512 \
        --use_vllm \
        --vllm_mode colocate \
        --vllm_gpu_memory_utilization 0.3 \
        --bf16 true \
        --use_flash_attention \
        --save_strategy epoch \
        --logging_steps 10 \
        --model_name "Qwen3-4B" \
        --exp_name "quickstart" \
        --output_base_dir "./outputs" \
        --prompt_only
}

# ============================================================================
# Main
# ============================================================================

case "$1" in
    teacher)
        start_teacher
        ;;
    train)
        run_training
        ;;
    *)
        echo "Usage:"
        echo "  Terminal 1: bash quickstart.sh teacher"
        echo "  Terminal 2: bash quickstart.sh train"
        echo ""
        echo "Or run both in background:"
        echo "  bash quickstart.sh teacher > teacher.log 2>&1 &"
        echo "  sleep 30  # wait for teacher to start"
        echo "  bash quickstart.sh train"
        ;;
esac
