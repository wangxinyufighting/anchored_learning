#!/bin/bash
#
# Start Progressive Teacher Server
#
# This script starts the teacher server on dedicated GPUs.
# The teacher serves mixed logits: T = α * SFT + (1-α) * ref
#

set -e

# ============================================================================
# Configuration - MODIFY THESE
# ============================================================================

SFT_MODEL="/path/to/your/sft_model"
INITIAL_REF_MODEL="/path/to/your/base_model"
MIXING_RATIO=0.5
PORT=8000

# GPU configuration for teacher server
# Example: Use GPUs 0-1 for teacher (adjust based on your setup)
TEACHER_GPUS="0,1"
TENSOR_PARALLEL_SIZE=2

# Memory settings
GPU_MEMORY_UTILIZATION=0.85
MAX_MODEL_LEN=4096

# ============================================================================
# Launch Teacher Server
# ============================================================================

echo "========================================="
echo "Starting Progressive Teacher Server"
echo "========================================="
echo "SFT model:     $SFT_MODEL"
echo "Initial ref:   $INITIAL_REF_MODEL"
echo "Mixing ratio:  $MIXING_RATIO"
echo "Port:          $PORT"
echo "GPUs:          $TEACHER_GPUS"
echo "Tensor parallel: $TENSOR_PARALLEL_SIZE"
echo "========================================="
echo

export CUDA_VISIBLE_DEVICES=$TEACHER_GPUS

python progressive_teacher_server.py \
    --sft-model "$SFT_MODEL" \
    --initial-ref-model "$INITIAL_REF_MODEL" \
    --mixing-ratio $MIXING_RATIO \
    --host 0.0.0.0 \
    --port $PORT \
    --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
    --max-model-len $MAX_MODEL_LEN \
    --trust-remote-code

echo "Teacher server stopped."
