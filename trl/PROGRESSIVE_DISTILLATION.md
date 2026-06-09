# Progressive On-Policy Distillation with vLLM

Fast implementation of progressive on-policy knowledge distillation using TRL's `DistillationTrainer` + vLLM.

## Algorithm Overview

**Progressive distillation** trains a student model through N iterative stages, where the teacher model progressively shifts from a base model toward a domain-specific SFT model:

- **Stage i**: Teacher T_i = α × SFT + (1-α) × R_i
- **Reference model**: R_i = S_{i-1} (previous stage's trained student)
- **Initial**: R_0 = S_0 = base model

This creates a curriculum that improves domain performance while mitigating catastrophic forgetting.

## Architecture

### Two-Server Design

**Teacher Server (separate GPUs)**:
- Loads frozen SFT model + dynamically-updated reference model
- Mixes logits on-the-fly: `α * sft_logits + (1-α) * ref_logits`
- Serves teacher distributions via OpenAI-compatible API
- Provides `/update_reference` endpoint to swap R_i between stages

**Student Training (main GPUs)**:
- Uses `DistillationTrainer` with vLLM for fast on-policy generation
- Fetches teacher logits from server
- Trains with mixed on-policy/off-policy data

### Why This Is Fast

1. **vLLM for student generation**: 5-20× faster than `.generate()`
2. **vLLM for teacher inference**: Batched, KV-cached, optimized
3. **Separate GPU pools**: Teacher and student run in parallel
4. **No redundant model loading**: Each model loaded once, reused across stages

## Files

```
trl/
├── progressive_teacher_server.py    # Custom teacher server with logit mixing
├── progressive_distillation.py      # Main training script (DistillationTrainer)
├── start_teacher_server.sh          # Launch teacher server
└── run_progressive_distill.sh       # Run student training
```

## Setup

### 1. Install Dependencies

```bash
pip install -U transformers accelerate datasets
pip install -U 'trl[vllm]'  # Includes vLLM
pip install fastapi uvicorn  # For teacher server
```

### 2. Configure GPU Allocation

Example for 8 GPUs:
- **Teacher server**: GPUs 0-1 (needs to fit SFT + ref model)
- **Student training**: GPUs 2-7 (for DDP training)

Adjust based on your hardware and model sizes.

### 3. Edit Configuration Files

**start_teacher_server.sh**:
```bash
SFT_MODEL="/path/to/your/sft_model"
INITIAL_REF_MODEL="/path/to/your/base_model"
MIXING_RATIO=0.5
TEACHER_GPUS="0,1"
TENSOR_PARALLEL_SIZE=2
```

**run_progressive_distill.sh**:
```bash
STUDENT_MODEL="/path/to/your/base_model"
DATA_PATH="/path/to/training_data.json"
NUM_STAGES=10
EPOCHS_PER_STAGE=1
MIXING_RATIO=0.5  # Must match teacher config
STUDENT_GPUS="2,3,4,5,6,7"
NPROC_PER_NODE=6
```

## Usage

### Step 1: Start Teacher Server

In terminal 1:

```bash
cd trl
bash start_teacher_server.sh
```

Wait for:
```
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Step 2: Run Progressive Training

In terminal 2:

```bash
cd trl
bash run_progressive_distill.sh
```

The script will:
1. Verify teacher server is ready
2. Run stage 1: Train S_0 → S_1 with teacher T_1 = α×SFT + (1-α)×base
3. Update teacher: R_2 = S_1
4. Run stage 2: Train S_1 → S_2 with teacher T_2 = α×SFT + (1-α)×S_1
5. Repeat for N stages

### Monitor Progress

Check teacher server status:
```bash
curl http://localhost:8000/status
```

Watch training logs in terminal 2.

## Data Format

Training data must be JSON with LLaMA-Factory format:

```json
[
  {
    "instruction": "What is photosynthesis?",
    "input": "",
    "output": "Photosynthesis is the process..."
  },
  {
    "instruction": "Translate to Spanish:",
    "input": "Hello world",
    "output": "Hola mundo"
  }
]
```

## Key Parameters

### Progressive Distillation

- `--num_stages`: Number of iterative stages (default: 10)
- `--epochs_per_stage`: Training epochs per stage (default: 1)
- `--mixing_ratio`: α for teacher mixing (0.0 = pure ref, 1.0 = pure SFT)

### On-Policy vs Off-Policy

- `--lmbda`: On-policy probability (0.0 = off-policy, 1.0 = fully on-policy)
- `--beta`: KL divergence type (0.0 = forward KL, 1.0 = reverse KL, 0.5 = JSD)
- `--temperature`: Sampling temperature for generation (recommended: 0.7-1.0)

### Performance Tuning

- `--use_vllm`: Enable vLLM for student generation (highly recommended)
- `--vllm_mode`: `colocate` (automatic) or `server` (manual vllm-serve)
- `--gradient_checkpointing`: Reduce memory at cost of 20% slower training
- `--use_flash_attention`: Use Flash Attention 2 (requires installation)

## Expected Speed

**Without vLLM** (baseline):
- ~100-300 samples/second per GPU (depends on model size)

**With vLLM** (this implementation):
- ~500-2000 samples/second per GPU
- **5-20× speedup** depending on sequence length and batch size

For 10 stages × 1 epoch × 10k samples on 6 GPUs:
- **Without vLLM**: ~5-10 hours
- **With vLLM**: ~30-120 minutes

## Troubleshooting

### Teacher server won't start

**Error**: `CUDA out of memory`
- Reduce `--gpu-memory-utilization` (try 0.7 or 0.5)
- Use more GPUs for tensor parallelism
- Use smaller models

### Student training fails

**Error**: `Teacher server not responding`
```bash
# Check server status
curl http://localhost:8000/health

# Restart server
pkill -f progressive_teacher_server
bash start_teacher_server.sh
```

**Error**: `vLLM initialization failed`
- Check CUDA_VISIBLE_DEVICES doesn't overlap with teacher GPUs
- Reduce `--vllm_gpu_memory_utilization` (default: 0.3)
- Try `--vllm_mode server` with manual vllm-serve

### Slow training

1. **Ensure vLLM is enabled**: Check logs for "Using vLLM"
2. **Increase batch size**: Raise `--per_device_train_batch_size` if memory allows
3. **Use Flash Attention**: Add `--use_flash_attention`
4. **Check generation batch size**: Auto-computed as `batch_size × grad_accum / num_generations`

## Advanced: Manual Teacher Update

You can manually update the reference model during training:

```bash
curl -X POST http://localhost:8000/update_reference \
  -H "Content-Type: application/json" \
  -d '{"model_path": "/path/to/new/checkpoint"}'
```

## Comparison with Original Implementation

| Feature | Old (anchored_learning.py) | New (progressive_distillation.py) |
|---------|---------------------------|-----------------------------------|
| Framework | GKDTrainer | DistillationTrainer |
| Student generation | model.generate() | vLLM |
| Teacher inference | PyTorch forward pass | vLLM server |
| Memory | All models in GPU memory | Separate GPU pools |
| Speed | Baseline | **5-20× faster** |
| Multi-GPU | DDP only | DDP + vLLM tensor parallelism |

## Citation

Progressive distillation builds on:

```bibtex
@inproceedings{agarwal2024on-policy,
  title={On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes},
  author={Agarwal, Rishabh and Vieillard, Nino and Zhou, Yongchao and Stanczyk, Piotr and Ramos Garea, Sabela and Geist, Matthieu and Bachem, Olivier},
  booktitle={ICLR},
  year={2024}
}
```

## Notes

- The teacher server keeps both SFT and reference models in memory. Plan GPU allocation accordingly.
- Reference model update takes ~10-30 seconds depending on model size and tensor parallelism.
- For very large models (70B+), consider using lower precision (int8/int4) for the teacher.
- Checkpoints are saved at `outputs/<run_name>/stage-final/` after each stage.
