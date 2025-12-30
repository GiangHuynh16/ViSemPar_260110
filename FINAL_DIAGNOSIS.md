# Final Diagnosis - Baseline 7B OOM Issue

**Date**: 2025-12-30
**Status**: Persistent OOM despite all optimizations

---

## Summary of Issue

Training baseline 7B consistently fails with OOM:
- GPU: 24GB Quadro RTX 6000
- Model: Qwen 2.5 7B + LoRA rank 128
- Error: Model uses ~22GB, leaving only ~1GB for training

**Critical observation**: MTUP 7B trains successfully on SAME hardware, but baseline 7B does NOT.

---

## All Optimizations Attempted

### 1. ✅ Memory Optimizations
- [x] max_memory reduced: 20GB → 16GB → 14GB
- [x] batch_size reduced: 2 → 1
- [x] max_seq_length reduced: 2048 → 1024 → 512
- [x] gradient_accumulation increased: 8 → 16
- [x] Gradient checkpointing enabled
- [x] CPU offload enabled
- [x] FP16 training
- [x] Monkey-patched Trainer to skip device movement

### 2. ✅ Code Fixes
- [x] Copied EXACT model loading from MTUP
- [x] Fixed device_map conflict
- [x] Removed quantization
- [x] Used adamw_torch optimizer

### 3. ✅ Environment
- [x] Using SAME environment as MTUP (lora_py310)
- [x] Same PyTorch version: 2.9.1+cu128
- [x] Same Transformers: 4.46.3
- [x] Same PEFT: 0.13.2

---

## Current Configuration

```python
# config/config.py
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
MAX_SEQ_LENGTH = 512  # Extremely reduced
batch_size = 1
gradient_accumulation = 16
max_memory = {0: "14GB", "cpu": "50GB"}
```

**Result**: Still OOM - model uses ~22GB on GPU

---

## Root Cause Analysis

### Why does MTUP work but baseline doesn't?

Hypothesis: **LoRA application timing**

When we:
1. Load model with `max_memory={0: "14GB"}` → Some layers on GPU, some on CPU
2. Apply LoRA with `get_peft_model()` → LoRA params added to GPU
3. LoRA params NOT counted in original max_memory limit
4. Total GPU usage exceeds 24GB → OOM

MTUP might be doing something different in LoRA application or has different config.

---

## Possible Solutions

### Option 1: Check MTUP's Actual Memory Usage

**On server, while MTUP is training or has trained:**

```bash
# If MTUP tmux session still exists
tmux attach -t mtup_7b
# Press Ctrl+C to pause if training
nvidia-smi

# Check actual memory usage
watch -n 1 nvidia-smi
```

**Question**: How much GPU memory does MTUP actually use during training?
- If MTUP also uses ~22GB → Something else is different
- If MTUP uses ~16-18GB → Different configuration somewhere

---

### Option 2: Use DeepSpeed ZeRO

DeepSpeed ZeRO-3 can offload optimizer states and gradients to CPU:

```python
# deepspeed_config.json
{
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {"device": "cpu"},
        "offload_param": {"device": "cpu"}
    }
}
```

Then train with:
```bash
deepspeed train_baseline.py --deepspeed deepspeed_config.json
```

**Complexity**: Medium - requires DeepSpeed setup

---

### Option 3: Switch to 3B Model

Accept that 7B doesn't fit and use 3B instead:

```python
# config/config.py
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
MAX_SEQ_LENGTH = 2048  # Can use full length
batch_size = 2  # Can use original
```

**Pros**:
- Will definitely work
- Still demonstrates single-task vs multi-task
- Faster training (~8h vs 15h)

**Cons**:
- Not directly comparable with MTUP 7B
- Different model size

---

### Option 4: Reduce LoRA Rank

```python
# config/config.py
LORA_CONFIG = {
    "r": 64,  # From 128
    "lora_alpha": 128,  # From 256
}
```

**Memory saved**: ~50% of LoRA params (~500MB-1GB)

**Trade-off**: Reduced model capacity, affects comparison

---

### Option 5: Train on Different Hardware

- Rent cloud GPU with more VRAM (48GB A6000)
- Cost: ~$0.4/hour × 15 hours = ~$6
- Platforms: RunPod, Vast.ai, Lambda Labs

---

## Recommended Next Steps

### Step 1: Verify MTUP Memory Usage

```bash
# On server
ps aux | grep train_mtup
# Note the PID

# Check memory of that process
nvidia-smi
# Look at memory used by Python process

# Or check logs
grep "memory" logs/mtup_*.log
```

**If MTUP uses < 20GB**: Something different in MTUP config
**If MTUP uses ~22GB**: Same issue, but MTUP might have more margin somehow

---

### Step 2: Decision Tree

```
Did MTUP training use < 20GB GPU?
├─ YES → Investigate MTUP difference (check logs, config, versions)
└─ NO (MTUP also uses ~22GB)
    ├─ Option A: Use DeepSpeed ZeRO-3
    ├─ Option B: Switch to 3B model
    ├─ Option C: Reduce LoRA rank to 64
    └─ Option D: Rent cloud GPU with 48GB VRAM
```

---

## Files to Check

### Check MTUP actual configuration used:

```bash
# Check MTUP training logs
cat logs/training_mtup*.log | grep -A 10 "Training Configuration"

# Check actual batch size used
cat logs/training_mtup*.log | grep "batch"

# Check actual memory allocation
cat logs/training_mtup*.log | grep -i "memory\|GB"
```

### Check if MTUP uses different model loading:

```bash
# Compare model loading between MTUP and baseline
diff -u <(grep -A 20 "AutoModelForCausalLM.from_pretrained" train_mtup.py) \
        <(grep -A 20 "AutoModelForCausalLM.from_pretrained" train_baseline.py)
```

---

## Conclusion

The OOM issue persists despite:
- ✅ Extreme memory optimizations (max_seq_length=512, batch_size=1)
- ✅ Same environment as MTUP
- ✅ Exact same model loading code
- ✅ Monkey-patching Trainer

**Next action required**:
1. Check MTUP's actual GPU usage during training
2. Based on result, choose from Options 1-5 above

If MTUP truly uses same ~22GB and works, there's a configuration difference we haven't found yet. Most likely in how LoRA is applied or in some Trainer setting.

---

## Quick Commands Reference

```bash
# Check current GPU usage
nvidia-smi

# Check Python process memory
ps aux | grep python | grep train

# Check MTUP logs
ls -lh logs/training_mtup*.log
cat logs/training_mtup*.log | tail -100

# Try baseline with 3B (fallback)
# Edit config/config.py:
# MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
# MAX_SEQ_LENGTH = 2048
# batch_size = 2
python train_baseline.py --epochs 15
```

---

**Status**: Awaiting decision on next approach based on MTUP memory investigation.
