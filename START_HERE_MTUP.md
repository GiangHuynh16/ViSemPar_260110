# 🚀 MTUP FIXED - Quick Start Guide

## 📋 Overview

**MTUP (Multi-Task Unified Prompt) Fixed** is an improved version of the two-stage AMR parsing approach, applying all lessons learned from the successful Baseline method.

**Key Improvements:**
- ✅ **Instruction masking** (train only on final AMR output)
- ✅ **Minimal prompt** with Penman examples (3 lines vs old 20+)
- ✅ **2 epochs** (prevent overfitting, down from 15)
- ✅ **Save every 100 steps** (early stopping)
- ✅ **bfloat16 precision** (memory efficient)
- ✅ **Penman format emphasis** throughout prompts

**Expected Results:**
- Structural validity: >90% (baseline achieved 91.3%)
- SMATCH F1: ~0.50-0.55 (hypothesis: MTUP improves over baseline's 0.47)
- Training time: ~4 hours on A6000 GPU

---

## 🎯 What is MTUP?

MTUP is a two-stage generation approach:

**Stage 1: Generate AMR without variables**
```
Input:  Anh ấy đã hoàn thành công việc.
Output: (hoàn_thành :agent (anh) :theme (công_việc) :aspect (đã))
```

**Stage 2: Add variables in Penman format**
```
Input:  (hoàn_thành :agent (anh) :theme (công_việc) :aspect (đã))
Output: (h / hoàn_thành
            :agent (a / anh)
            :theme (c / công_việc)
            :aspect (đ / đã))
```

**Hypothesis:** Explicit decomposition helps the model learn AMR structure better than direct generation.

---

## 📁 File Structure

### Core Implementation
```
config/
├── prompt_templates_fixed.py       # Minimal prompts with Penman examples
├── config_mtup_fixed.py            # Training configuration (2 epochs, masking, etc.)

train_mtup_fixed.py                 # Training script with instruction masking
predict_mtup_fixed.py               # Two-stage inference pipeline
preprocess_mtup.py                  # Convert training data to MTUP format
```

### Helper Scripts
```
TRAIN_MTUP_FIXED.sh                 # Training wrapper
TEST_MTUP_FIXED.sh                  # Single example test
EVALUATE_MTUP_CHECKPOINTS.sh        # Find best checkpoint
```

### Documentation
```
START_HERE_MTUP.md                  # This file
FINAL_DOCUMENT_BASELINE_UPDATE.md   # Baseline results for comparison
```

---

## ⚡ Quick Start (5 Steps)

### Step 1: Preprocess Training Data (5 minutes)

Convert training data to MTUP two-stage format:

```bash
python3 preprocess_mtup.py \
    --input data/train_amr_1.txt \
    --output data/train_amr_mtup_preprocessed.txt
```

**Expected output:**
- Creates `train_amr_mtup_preprocessed.txt` with sentence → (AMR no vars) → (AMR with vars)

---

### Step 2: Train the Model (~4 hours)

```bash
bash TRAIN_MTUP_FIXED.sh
```

**What happens:**
1. Loads Qwen 2.5 7B base model
2. Applies LoRA adapters (11M trainable params)
3. Trains for 2 epochs with instruction masking
4. Saves checkpoint every 100 steps
5. Total: ~16 checkpoints

**Monitor training:**
```bash
# In another terminal
tail -f logs/training_mtup_fixed_*.log
```

**Expected checkpoints:**
```
outputs/mtup_fixed_YYYYMMDD_HHMMSS/
├── checkpoint-100/
├── checkpoint-200/
├── ...
├── checkpoint-1500/
└── checkpoint-1600/
```

---

### Step 3: Evaluate All Checkpoints (30 minutes)

Find the best checkpoint based on structural validity:

```bash
bash EVALUATE_MTUP_CHECKPOINTS.sh outputs/mtup_fixed_YYYYMMDD_HHMMSS
```

**Output:**
```
Checkpoint       Valid_AMRs  Total  Validity_Percentage
checkpoint-100   42          50     84.0
checkpoint-200   45          50     90.0
checkpoint-300   46          50     92.0  ← Best
checkpoint-400   44          50     88.0
...
```

**Select the checkpoint with highest validity.**

---

### Step 4: Test on Full Dataset (1 hour)

Run predictions on 150 test samples:

```bash
python3 predict_mtup_fixed.py \
    --model outputs/mtup_fixed_YYYYMMDD_HHMMSS/checkpoint-300 \
    --test-file data/public_test.txt \
    --output evaluation_results/mtup_fixed_predictions.txt \
    --verbose
```

**Expected output:**
- `mtup_fixed_predictions.txt` with 150 AMR graphs
- Validation summary: X/150 valid (target: >90%)

---

### Step 5: Calculate SMATCH F1 (5 minutes)

Filter valid AMRs and calculate SMATCH:

```bash
# Filter to valid AMRs only
python3 filter_valid_amrs.py \
    --predictions evaluation_results/mtup_fixed_predictions.txt \
    --ground-truth data/public_test_ground_truth.txt \
    --output-pred evaluation_results/mtup_valid.txt \
    --output-gold evaluation_results/gold_valid.txt

# Calculate SMATCH
python -m smatch -f \
    evaluation_results/mtup_valid.txt \
    evaluation_results/gold_valid.txt \
    --significant 4
```

**Expected SMATCH output:**
```
F-score: 0.52
Precision: 0.54
Recall: 0.50
```

**Compare with Baseline:**
| Method   | F1   | Validity | Trainable Params |
|----------|------|----------|------------------|
| Baseline | 0.47 | 91.3%    | 11M (0.15%)      |
| MTUP     | 0.52 | 92.0%    | 11M (0.15%)      |

---

## 🧪 Quick Test (Before Full Training)

Test a single example to verify the pipeline works:

```bash
bash TEST_MTUP_FIXED.sh
```

**What it does:**
1. Tests on one sentence: "Anh ấy đã hoàn thành công việc."
2. Shows two-stage generation:
   - Step 1: AMR without variables
   - Step 2: Penman format with variables
3. Validates output format

**Use this to:**
- Verify code works before 4-hour training
- Debug prompt/extraction issues
- Check model loading

---

## 🔍 Key Differences from Old MTUP

| Aspect                | Old MTUP (Broken)        | MTUP Fixed               |
|-----------------------|--------------------------|--------------------------|
| **Prompt Length**     | 20+ lines                | ~10 lines                |
| **Penman Examples**   | None                     | 1-2 clear examples       |
| **Instruction Mask**  | ❌ No masking            | ✅ Proper masking        |
| **Epochs**            | 15 (overfitting)         | 2 (early stopping)       |
| **Save Frequency**    | Every 200 steps          | Every 100 steps          |
| **Precision**         | fp16                     | bfloat16                 |
| **Output Format**     | Long explanations        | Clean Penman AMR         |
| **Expected Validity** | <50%                     | >90%                     |

---

## 📊 Configuration Details

### Prompt Template (Ultra Minimal)

**File:** [config/prompt_templates_fixed.py](config/prompt_templates_fixed.py)

```python
MTUP_ULTRA_MINIMAL = """Chuyển câu tiếng Việt sau sang AMR theo chuẩn PENMAN.

VÍ DỤ:
Câu: Anh ấy đã hoàn thành công việc.
AMR không biến: (hoàn_thành :agent (anh) :theme (công_việc) :aspect (đã))
AMR chuẩn PENMAN:
(h / hoàn_thành
    :agent (a / anh)
    :theme (c / công_việc)
    :aspect (đ / đã))

---

Câu: {sentence}

AMR không biến:
{amr_no_vars}

AMR chuẩn PENMAN:
{amr_with_vars}"""
```

**Key features:**
- Emphasizes "chuẩn PENMAN" (Penman standard)
- One clear example showing both stages
- Clear markers for extraction

---

### Training Configuration

**File:** [config/config_mtup_fixed.py](config/config_mtup_fixed.py)

```python
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
MAX_SEQ_LENGTH = 512

LORA_CONFIG = {
    "r": 64,
    "lora_alpha": 128,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
}

TRAINING_CONFIG = {
    "learning_rate": 2e-4,
    "num_train_epochs": 2,              # FIXED: was 15
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 16,
    "save_steps": 100,                  # FIXED: was 200
    "bf16": True,                       # bfloat16
}

MTUP_CONFIG = {
    "template_type": "ultra_minimal",
    "use_instruction_masking": True,     # NEW: Critical fix
    "step1_marker": "AMR không biến:",
    "step2_marker": "AMR chuẩn PENMAN:",
}
```

---

## 🐛 Troubleshooting

### Problem: Invalid AMRs (unbalanced parentheses)

**Check:**
1. Instruction masking is enabled in config
2. Training completed at least 200 steps
3. Using best checkpoint (not latest)

**Fix:**
```bash
# Verify masking is working
grep "use_instruction_masking" config/config_mtup_fixed.py
# Should show: True

# Try checkpoint 300-500 instead of 100-200
python3 predict_mtup_fixed.py --model outputs/.../checkpoint-400
```

---

### Problem: Model generates explanations instead of AMR

**Check:**
1. Prompt template has Penman examples
2. Extraction markers are correct
3. EOS token is being removed

**Fix:**
```bash
# Verify prompt template
grep "chuẩn PENMAN" config/prompt_templates_fixed.py

# Check extraction logic in predict_mtup_fixed.py
# Should split on "AMR chuẩn PENMAN:" marker
```

---

### Problem: OOM (Out of Memory) during training

**Reduce memory usage:**

Edit [config/config_mtup_fixed.py](config/config_mtup_fixed.py):
```python
TRAINING_CONFIG = {
    "per_device_train_batch_size": 1,    # Already at minimum
    "gradient_accumulation_steps": 32,   # Increase to 32 (was 16)
}

# Or reduce LoRA rank
LORA_CONFIG = {
    "r": 32,            # Reduce from 64
    "lora_alpha": 64,   # 2x rank
}
```

---

### Problem: Training too slow

**Speed up:**
1. Reduce validation frequency (already at 100 steps)
2. Use smaller model (Qwen 2.5 3B instead of 7B)
3. Reduce max_seq_length to 256 (if sentences are short)

---

## 📈 Expected Timeline

| Step | Duration | Notes |
|------|----------|-------|
| Preprocess data | 5 min | One-time setup |
| Training (2 epochs) | ~4 hours | A6000 GPU |
| Evaluate checkpoints | 30 min | ~16 checkpoints |
| Test on 150 samples | 1 hour | Two-stage inference |
| Calculate SMATCH | 5 min | Filter + score |
| **Total** | **~6 hours** | Hands-on: ~1 hour |

---

## 🎯 Success Criteria

**Minimum acceptable:**
- ✅ Structural validity: >85%
- ✅ SMATCH F1: >0.45 (baseline level)
- ✅ Training converges (loss decreases)

**Target (hypothesis):**
- 🎯 Structural validity: >90%
- 🎯 SMATCH F1: >0.50 (improvement over baseline)
- 🎯 Fewer invalid AMRs than baseline

**Excellent:**
- 🏆 Structural validity: >92%
- 🏆 SMATCH F1: >0.52
- 🏆 Demonstrates MTUP advantage over direct generation

---

## 📚 Comparison with Baseline

### Baseline (Direct Generation)
- **Method:** Single-stage generation (sentence → AMR)
- **Results:** F1=0.47, 91.3% validity
- **Pros:** Fast inference (1 pass), simple
- **Cons:** No explicit decomposition

### MTUP Fixed (Two-Stage)
- **Method:** Two-stage (sentence → AMR no vars → Penman AMR)
- **Results:** TBD (hypothesis: F1>0.50, validity>90%)
- **Pros:** Explicit decomposition, better structure learning
- **Cons:** Slower inference (2 passes)

**Research Question:** Does explicit task decomposition improve AMR parsing quality?

---

## 📝 For Thesis

### If MTUP improves over Baseline:

**Section 4.5: MTUP Method**
> We propose a two-stage MTUP approach that decomposes AMR generation into:
> (1) generating structure without variables, and (2) adding variables in Penman format.
> This explicit decomposition achieved **F1=0.XX** (+Y% over baseline) with **Z%
> structural validity**, demonstrating that task decomposition aids learning.

### If MTUP performs similarly to Baseline:

> MTUP achieved comparable results (F1=0.XX) to the direct generation baseline,
> suggesting that modern LLMs can learn complex AMR structure end-to-end without
> explicit decomposition. However, MTUP may offer advantages in interpretability
> and debugging.

---

## 🔗 Related Files

- [FINAL_DOCUMENT_BASELINE_UPDATE.md](FINAL_DOCUMENT_BASELINE_UPDATE.md) - Baseline results for comparison
- [train_mtup_fixed.py](train_mtup_fixed.py) - Training implementation
- [predict_mtup_fixed.py](predict_mtup_fixed.py) - Inference implementation
- [config/prompt_templates_fixed.py](config/prompt_templates_fixed.py) - Prompt templates
- [config/config_mtup_fixed.py](config/config_mtup_fixed.py) - Configuration

---

## ✅ Next Steps

1. **Preprocess data:** `python3 preprocess_mtup.py`
2. **Start training:** `bash TRAIN_MTUP_FIXED.sh`
3. **Monitor progress:** `tail -f logs/training_mtup_fixed_*.log`
4. **Evaluate checkpoints:** `bash EVALUATE_MTUP_CHECKPOINTS.sh`
5. **Calculate SMATCH:** Use `filter_valid_amrs.py` + `smatch`
6. **Compare with baseline:** Document in thesis

---

**Good luck! 🚀**

If you encounter issues, check:
1. GPU memory (needs ~26GB for training)
2. Instruction masking is enabled
3. Prompt template has Penman examples
4. Using best checkpoint, not latest
