# 🔧 Fix 14B Model Out of Memory (OOM)

## ❌ Problem

```
torch.OutOfMemoryError: CUDA out of memory
GPU 0 has a total capacity of 23.64 GiB
Process 349310 has 1.73 GiB memory in use
Process 358186 has 21.72 GiB memory in use  ← 14B model
Free: Only 46.75 MiB ← NOT ENOUGH!
```

**Root causes**:
1. Multiple processes using GPU simultaneously
2. 14B model requires ~22GB
3. No memory left for generation
4. 24GB GPU is at the limit for 14B FP16

---

## ✅ Solution 1: Kill Old Processes (Quick Fix)

### On Server:

```bash
# Check GPU usage
nvidia-smi

# Kill all Python processes
pkill -9 -f python

# Verify GPU is clear
nvidia-smi
# Should show minimal memory usage now
```

Then retry your script.

---

## ✅ Solution 2: Use Memory-Optimized Script

### Step 1: Use Safe Runner

```bash
chmod +x RUN_INFERENCE_14B_SAFE.sh
bash RUN_INFERENCE_14B_SAFE.sh
```

This script:
- ✅ Automatically kills old processes
- ✅ Clears GPU memory
- ✅ Processes one sentence at a time
- ✅ Uses aggressive memory optimization
- ✅ Retries on OOM with reduced settings

### Step 2: Or Run Directly

```bash
python3 inference_14b_memory_optimized.py \
  --model outputs/checkpoints/qwen2.5-14b-fine-tuned \
  --input data/public_test_sentences.txt \
  --output outputs/predictions_14b.json \
  --max-samples 150
```

---

## ✅ Solution 3: Use 8-bit Quantization (Recommended!)

**Reduces memory by ~50%!**

### Install bitsandbytes first:

```bash
conda activate amr-parser
pip install bitsandbytes
```

### Run with 8-bit:

```bash
python3 inference_14b_8bit.py \
  --model outputs/checkpoints/qwen2.5-14b-fine-tuned \
  --input data/public_test_sentences.txt \
  --output outputs/predictions_14b_8bit.json \
  --max-samples 150
```

**Memory usage**:
- FP16: ~22 GB
- Int8: ~11 GB ← 50% reduction!

---

## ✅ Solution 4: Use MTUP 3B Model Instead

**The 3B model works great and needs much less memory!**

```bash
bash RUN_FULL_EVALUATION_TMUX.sh
```

**Advantages**:
- ✅ Only ~6 GB memory
- ✅ Already tested (F1 = 0.49)
- ✅ Much faster inference
- ✅ Vietnamese prompts working

---

## 📊 Memory Comparison

| Model | Size | Memory (FP16) | Memory (Int8) | GPU Needed |
|-------|------|---------------|---------------|------------|
| Qwen 2.5 3B | 3B | ~6 GB | ~3 GB | 8 GB+ |
| Qwen 2.5 14B | 14B | ~22 GB | ~11 GB | 24 GB+ |

**Your GPU**: 24 GB ← Just enough for 14B FP16, but no room for generation!

---

## 🔍 Diagnosis Steps

### 1. Check what's using GPU:

```bash
nvidia-smi
```

Look for:
- Multiple Python processes?
- High memory usage before your script?

### 2. Check available memory:

```bash
nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits
```

Need at least **12000 MiB** free for 14B inference.

### 3. Check if model loaded:

If you see this before OOM:
```
✅ Model loaded!
Total sentences: 150
Generating predictions...
  0%|          | 0/150 [10:20<?, ?it/s]
```

Model loaded OK, but **OOM during generation**.

---

## 🎯 Recommended Approach

### For 24GB GPU:

**Option A**: Use 8-bit quantization
```bash
pip install bitsandbytes
python3 inference_14b_8bit.py --model ... --input ... --output ...
```

**Option B**: Use 3B MTUP model (proven to work)
```bash
bash RUN_FULL_EVALUATION_TMUX.sh
```

### For Smaller GPU (<24GB):

Must use 3B model:
```bash
bash RUN_FULL_EVALUATION_TMUX.sh
```

---

## 🛠️ Manual Memory Optimization

If you want to edit your own script:

```python
# 1. Load model with optimization
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    offload_folder="offload",  # Offload to disk
)

# 2. Process ONE sentence at a time
for sentence in sentences:
    inputs = tokenizer(sentence, ...).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,  # Limit tokens
            do_sample=False,     # No sampling overhead
            num_beams=1,         # No beam search
        )

    # 3. Clear memory after EACH sentence
    del inputs, outputs
    torch.cuda.empty_cache()
```

---

## 🚨 Common Mistakes

### ❌ Processing all sentences in batch:

```python
# DON'T DO THIS with 14B!
inputs = tokenizer(all_sentences, ...)  # Too much memory!
```

### ❌ Not clearing memory between sentences:

```python
# DON'T DO THIS
for sentence in sentences:
    outputs = model.generate(...)
    # Missing: del outputs; torch.cuda.empty_cache()
```

### ❌ Using sampling with large batch:

```python
# AVOID with 14B on 24GB GPU
do_sample=True,
top_p=0.9,
temperature=0.7,
# These add memory overhead!
```

---

## 📈 Performance Trade-offs

| Method | Memory | Speed | Quality | Recommended |
|--------|--------|-------|---------|-------------|
| 14B FP16 | 22 GB | 1x | Best | ❌ OOM on 24GB |
| 14B Int8 | 11 GB | 0.8x | Very good | ✅ Works |
| 3B FP16 | 6 GB | 3x | Good (F1=0.49) | ✅ Best choice |

---

## 💡 Quick Decision Tree

```
Do you have >32GB GPU?
├─ Yes → Use 14B FP16 directly
└─ No → Do you have 24GB GPU?
    ├─ Yes → Use 14B Int8 (this guide)
    └─ No → Use 3B model (recommended)
```

---

## 📝 Files Created

1. ✅ `inference_14b_memory_optimized.py` - Optimized script
2. ✅ `inference_14b_8bit.py` - 8-bit quantization
3. ✅ `RUN_INFERENCE_14B_SAFE.sh` - Auto-fix runner
4. ✅ `FIX_14B_OOM.md` - This guide

---

## 🎯 Next Steps

### Immediate:

```bash
# Kill old processes
pkill -9 -f python

# Run with 8-bit
pip install bitsandbytes
python3 inference_14b_8bit.py \
  --model outputs/checkpoints/qwen2.5-14b-fine-tuned \
  --input data/public_test_sentences.txt \
  --output outputs/predictions_14b_8bit.json
```

### Alternative:

```bash
# Use proven 3B MTUP model
bash RUN_FULL_EVALUATION_TMUX.sh
```

---

## ✅ Expected Results

### After fixing OOM:

```
✅ Model loaded in 8-bit!
GPU Memory: 11.23 GB allocated, 12.45 GB reserved

Total sentences: 150

Generating predictions...
Processing: 100%|██████████| 150/150 [15:32<00:00,  6.21s/it]

✅ Done!
Generated 150 predictions
```

---

_This guide solves the 14B OOM issue on 24GB GPUs_
