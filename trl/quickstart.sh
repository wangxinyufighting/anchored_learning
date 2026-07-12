#!/usr/bin/env bash
#
# Quick Start Example for Progressive Distillation
#
# Usage from the trl directory:
#   bash quickstart.sh teacher
#   bash quickstart.sh train

set -eu

# ============================================================================
# STEP 1: Edit these paths
# ============================================================================

SFT_MODEL="/czsun/zhi/xywang/anchored_learning/LlamaFactory/saves/Qwen3-4B_medcalc_train_1e-6"
BASE_MODEL="/czsun/models/Qwen3-4B"
TRAIN_DATA="/czsun/zhi/xywang/anchored_learning/LlamaFactory/data/medcalc_train.json"

# GPU allocation.
# The teacher needs two separate vLLM backends: one SFT model and one reference model.
# Keep teacher GPUs and student GPUs disjoint.
TEACHER_SFT_GPU="0"
TEACHER_REF_GPU="1"
STUDENT_GPUS="2,3"
STUDENT_NPROC=2

# Teacher vLLM memory settings. Lower these first if teacher startup still OOMs.
TEACHER_GPU_MEMORY_UTILIZATION=0.75
TEACHER_MAX_MODEL_LEN=4096
TEACHER_DTYPE="bfloat16"

# ============================================================================
# STEP 2: Start teacher server (terminal 1)
# ============================================================================

start_teacher() {
    echo "Starting teacher proxy: SFT on GPU ${TEACHER_SFT_GPU}, ref on GPU ${TEACHER_REF_GPU}..."

    python progressive_teacher_proxy.py \
        --sft-model "$SFT_MODEL" \
        --initial-ref-model "$BASE_MODEL" \
        --mixing-ratio 0.5 \
        --port 8001 \
        --sft-gpu "$TEACHER_SFT_GPU" \
        --ref-gpu "$TEACHER_REF_GPU" \
        --gpu-memory-utilization "$TEACHER_GPU_MEMORY_UTILIZATION" \
        --max-model-len "$TEACHER_MAX_MODEL_LEN" \
        --dtype "$TEACHER_DTYPE"
}

# ============================================================================
# STEP 3: Run training (terminal 2)
# ============================================================================

run_training() {
    echo "Starting progressive distillation on GPUs ${STUDENT_GPUS}..."
    echo "Waiting for teacher server..."

    i=1
    while [ "$i" -le 30 ]; do
        if curl -s http://localhost:8001/health >/dev/null 2>&1; then
            echo "Teacher server ready!"
            break
        fi
        i=$((i + 1))
        sleep 2
    done

    CUDA_VISIBLE_DEVICES=$STUDENT_GPUS torchrun \
        --nproc_per_node=$STUDENT_NPROC \
        progressive_distillation.py \
        --student_model_path "$BASE_MODEL" \
        --teacher_server_url "http://localhost:8001" \
        --data_path "$TRAIN_DATA" \
        --data_name "quickstart" \
        --num_stages 10 \
        --epochs_per_stage 2 \
        --mixing_ratio 0.5 \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 4 \
        --lr 1e-5 \
        --temperature 0.9 \
        --lmbda 1.0 \
        --beta 1.0 \
        --loss_top_k 1 \
        --max_length 3500 \
        --max_prompt_length 1024 \
        --max_completion_length 2500 \
        --use_vllm \
        --vllm_mode colocate \
        --vllm_gpu_memory_utilization 0.2 \
        --bf16 true \
        --use_flash_attention \
        --save_strategy epoch \
        --logging_steps 10 \
        --model_name "Qwen3-4B" \
        --exp_name "quickstart" \
        --output_base_dir "./outputs" \
        --prompt_only
}

case "${1:-}" in
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
        echo "  sleep 60"
        echo "  bash quickstart.sh train"
        ;;
esac
