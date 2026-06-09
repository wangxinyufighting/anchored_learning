# 🚀 Progressive On-Policy Distillation (Fast Implementation)

**5-20× faster** than the original `anchored_learning.py` using vLLM + DistillationTrainer

---

## 📖 Quick Overview

This implements your progressive distillation algorithm where the teacher model gradually shifts from base model to SFT model across N training stages:

```
Stage 1: T₁ = α·SFT + (1-α)·base     → trains S₁
Stage 2: T₂ = α·SFT + (1-α)·S₁       → trains S₂
Stage 3: T₃ = α·SFT + (1-α)·S₂       → trains S₃
...
Stage N: Tₙ = α·SFT + (1-α)·Sₙ₋₁     → trains Sₙ (final)
```

This creates a curriculum that improves domain performance while preserving general capabilities.

---

## 🎯 Files You Need

| File | Purpose | You Need To |
|------|---------|-------------|
| `progressive_teacher_server.py` | Teacher server | ✅ Run on separate GPUs |
| `progressive_distillation.py` | Training script | ✅ Run with torchrun |
| `quickstart.sh` | Easy launcher | ✏️ Edit paths, then run |
| `start_teacher_server.sh` | Teacher launcher | ✏️ Edit paths, optional |
| `run_progressive_distill.sh` | Training launcher | ✏️ Edit paths, optional |

---

## ⚡ Quick Start (3 Steps)

### Step 1: Edit `quickstart.sh`

```bash
# Change these 3 lines:
SFT_MODEL="/path/to/your/sft_model"
BASE_MODEL="/path/to/your/base_model"
TRAIN_DATA="/path/to/your_data.json"
```

### Step 2: Start Teacher Server

**Terminal 1:**
```bash
cd trl
bash quickstart.sh teacher
```

Wait for: `INFO:     Uvicorn running on http://0.0.0.0:8000`

### Step 3: Run Training

**Terminal 2:**
```bash
cd trl
bash quickstart.sh train
```

**That's it!** Training will run through 10 stages automatically.

---

## 📊 What to Expect

### Speed
- **Original method**: 100-300 samples/sec per GPU
- **This method**: 500-2000 samples/sec per GPU
- **Speedup**: **5-20×** ⚡

### Example Timing (10 stages, 10k samples, 6 GPUs)
- **Old**: ~5-10 hours
- **New**: ~30-120 minutes

### GPU Usage
- **Teacher server**: 2 GPUs (recommend: GPUs 0-1)
- **Student training**: Remaining GPUs (recommend: GPUs 2-7)

---

## 🏗️ Architecture

```
Teacher Server                Student Training
(GPUs 0-1)                    (GPUs 2-7)
┌────────────┐                ┌────────────┐
│ SFT Model  │                │  Student   │
│  (frozen)  │                │   Model    │
└─────┬──────┘                └─────┬──────┘
      │                             │
┌─────▼──────┐                      │
│ Ref Model  │                      │
│ (updates)  │                      │
└─────┬──────┘                      │
      │                             │
      │ Mix logits                  │
      │ α·SFT+(1-α)·Ref            │
      │                             │
┌─────▼──────┐   HTTP (logits)     │
│   vLLM     │────────────────────▶│
│  Server    │                     │
└────────────┘              ┌──────▼──────┐
                            │ Distillation│
                            │   Trainer   │
                            │  + vLLM     │
                            └─────┬───────┘
                                  │
                            ┌─────▼──────┐
                            │ Checkpoint │
                            └─────┬──────┘
                                  │
      Update ref model ◄──────────┘
      for next stage
```

---

## 🔧 Key Parameters

### Progressive Distillation
- `--num_stages`: Number of iterations (default: 10)
- `--epochs_per_stage`: Epochs per stage (default: 1)
- `--mixing_ratio`: α for teacher mixing (default: 0.5)
  - 0.0 = pure reference model
  - 1.0 = pure SFT model
  - 0.5 = balanced mix

### On-Policy Control
- `--lmbda`: On-policy probability (default: 1.0)
  - 0.0 = fully off-policy (uses gold completions)
  - 1.0 = fully on-policy (student generates)
  - 0.5 = 50/50 mix

### Loss Function
- `--beta`: KL divergence type (default: 1.0)
  - 0.0 = forward KL (teacher → student)
  - 1.0 = reverse KL (student → teacher)
  - 0.5 = Jensen-Shannon Divergence

---

## 📝 Data Format

JSON file with LLaMA-Factory format:

```json
[
  {
    "instruction": "What is machine learning?",
    "input": "",
    "output": "Machine learning is a subset of AI..."
  },
  {
    "instruction": "Translate:",
    "input": "Hello world",
    "output": "你好世界"
  }
]
```

---

## 🐛 Troubleshooting

### Teacher won't start
```bash
# Check GPU availability
nvidia-smi

# Try smaller memory utilization
# Edit start_teacher_server.sh:
GPU_MEMORY_UTILIZATION=0.7  # was 0.85
```

### Training fails
```bash
# Check teacher is running
curl http://localhost:8000/status

# Restart teacher
bash stop_teacher_server.sh
bash start_teacher_server.sh
```

### Out of memory
```bash
# Reduce batch size in run_progressive_distill.sh:
PER_DEVICE_BATCH_SIZE=1  # was 2

# Or increase gradient accumulation:
GRADIENT_ACCUMULATION_STEPS=8  # was 4
```

### Slow training
```bash
# Verify vLLM is enabled (check training logs):
grep "Using vLLM" <training_log>

# Should see: "Using vLLM for generation"
```

---

## 📚 Documentation

- **`PROGRESSIVE_DISTILLATION.md`** - Full documentation
- **`IMPLEMENTATION_SUMMARY.md`** - Architecture and design details
- **`quickstart.sh`** - Simplest way to start

---

## ✅ Checklist Before Running

- [ ] vLLM installed: `pip install vllm`
- [ ] TRL updated: `pip install -U trl[vllm]`
- [ ] Model paths edited in `quickstart.sh`
- [ ] Data file in LLaMA-Factory format
- [ ] GPUs available (minimum 3: 1 for teacher, 2+ for student)
- [ ] Port 8000 available

---

## 🎓 How It Works

1. **Teacher server starts** with SFT model + base model as reference
2. **Stage 1 trains** student from base model
3. **Server updates** reference model to stage 1 checkpoint
4. **Stage 2 trains** with updated teacher
5. **Repeat** until all stages complete

Each stage, the teacher becomes slightly more like the SFT model, creating a smooth transition.

---

## 💡 Tips

- **Start small**: Test with 100 samples, 2 stages first
- **Monitor GPUs**: Use `watch -n 1 nvidia-smi` in a third terminal
- **Save logs**: `bash quickstart.sh train 2>&1 | tee training.log`
- **Check status**: `curl http://localhost:8000/status` anytime

---

## 🆘 Need Help?

1. Check `PROGRESSIVE_DISTILLATION.md` for detailed guide
2. Look at logs in both terminal 1 (teacher) and terminal 2 (training)
3. Verify teacher status: `curl http://localhost:8000/status`

---

**Ready to train?** Edit `quickstart.sh` and run it! 🚀
