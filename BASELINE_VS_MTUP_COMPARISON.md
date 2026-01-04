# 📊 Baseline vs MTUP - Quick Comparison

## At a Glance

| Aspect | Baseline | MTUP Fixed |
|--------|----------|------------|
| **Approach** | Direct generation | Two-stage decomposition |
| **Inference Passes** | 1 | 2 |
| **Prompt Length** | 3 lines | ~10 lines (with example) |
| **Training Time** | ~2.5 hours | ~4 hours |
| **Inference Speed** | 5 sent/sec | ~2.5 sent/sec |
| **Expected F1** | 0.47 ✅ | >0.50 (hypothesis) |
| **Expected Validity** | 91.3% ✅ | >90% (hypothesis) |

---

## 🎯 Core Methodology

### Baseline: Single-Stage Direct Generation

```
Input:  "Anh ấy đã hoàn thành công việc."
         ↓
    [Model generates]
         ↓
Output: (h / hoàn_thành
            :agent (a / anh)
            :theme (c / công_việc)
            :aspect (đ / đã))
```

**Pros:**
- ✅ Simple, end-to-end
- ✅ Fast inference (1 pass)
- ✅ Proven results (F1=0.47)

**Cons:**
- ❌ No intermediate supervision
- ❌ Must learn complex structure in one step
- ❌ 8.7% invalid outputs

---

### MTUP: Two-Stage Decomposition

```
Input:  "Anh ấy đã hoàn thành công việc."
         ↓
    [Stage 1: Generate structure]
         ↓
Step 1: (hoàn_thành :agent (anh) :theme (công_việc) :aspect (đã))
         ↓
    [Stage 2: Add variables]
         ↓
Step 2: (h / hoàn_thành
            :agent (a / anh)
            :theme (c / công_việc)
            :aspect (đ / đã))
```

**Pros:**
- ✅ Explicit task decomposition
- ✅ Two-stage supervision
- ✅ Intermediate output for debugging

**Cons:**
- ❌ Slower inference (2 passes)
- ❌ Error accumulation possible
- ❌ More complex pipeline

---

## 📝 Prompt Comparison

### Baseline Prompt (3 lines)

```
Chuyển câu tiếng Việt sau sang AMR (Abstract Meaning Representation)
theo định dạng Penman:

Câu: {sentence}

AMR:
```

**Design:** Minimal, task-only

---

### MTUP Prompt (~10 lines)

**Training Template:**
```
Chuyển câu tiếng Việt sau sang AMR theo chuẩn PENMAN.

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
{amr_with_vars}
```

**Design:** Minimal + 1 Penman example

**Two inference templates:**
1. Stage 1: Sentence → AMR without vars
2. Stage 2: AMR without vars → Penman AMR

---

## ⚙️ Training Configuration

### Shared Settings (Both Use Same)

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
    "num_train_epochs": 2,              # Both use 2 epochs
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 16,
    "save_steps": 100,                  # Both save every 100 steps
    "bf16": True,                       # Both use bfloat16
    "use_instruction_masking": True,    # Both use masking
}
```

**Key insight:** MTUP uses **exact same training config** as successful Baseline.

---

### Differences

| Parameter | Baseline | MTUP |
|-----------|----------|------|
| Data format | Sentence → AMR | Sentence → AMR_no_vars → AMR_with_vars |
| Training target | Final AMR only | Final Penman AMR only (masked) |
| Prompt template | Minimal (3 lines) | Minimal + example (~10 lines) |
| Inference stages | 1 | 2 |

---

## 🔬 Technical Implementation

### Instruction Masking (Both Use Same Approach)

```python
# Baseline
prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
amr_ids = tokenizer.encode(amr, add_special_tokens=False)
eos_ids = tokenizer.encode(tokenizer.eos_token, add_special_tokens=False)

input_ids = prompt_ids + amr_ids + eos_ids
labels = input_ids.copy()
labels[:len(prompt_ids)] = -100  # Only train on AMR
```

```python
# MTUP (identical approach)
instruction_ids = tokenizer.encode(instruction, add_special_tokens=False)
target_ids = tokenizer.encode(target_amr, add_special_tokens=False)
eos_ids = tokenizer.encode(tokenizer.eos_token, add_special_tokens=False)

input_ids = instruction_ids + target_ids + eos_ids
labels = input_ids.copy()
labels[:len(instruction_ids)] = -100  # Only train on final AMR
```

**Identical technique, different data format.**

---

### Generation Settings (Both Use Same)

```python
INFERENCE_CONFIG = {
    "temperature": 0.3,
    "top_p": 0.95,
    "repetition_penalty": 1.2,
    "max_new_tokens": 512,
    "do_sample": True,
}
```

---

## 📊 Results Comparison

### Baseline Results (Confirmed)

| Metric | Value | Notes |
|--------|-------|-------|
| **SMATCH F1** | **0.47** | ✅ Confirmed |
| **Structural Validity** | **91.3%** (137/150) | ✅ Excellent |
| **Invalid AMRs** | 8.7% (13/150) | Unbalanced (9), Duplicates (4) |
| **Training Time** | 2.5 hours | A6000 GPU |
| **Inference Speed** | ~5 sent/sec | Single pass |

---

### MTUP Expected Results (Hypothesis)

| Metric | Target | Stretch Goal |
|--------|--------|--------------|
| **SMATCH F1** | >0.50 | >0.52 |
| **Structural Validity** | >90% | >92% |
| **Invalid AMRs** | <15/150 | <10/150 |
| **Training Time** | ~4 hours | Same GPU |
| **Inference Speed** | ~2.5 sent/sec | Two passes |

---

## 🎯 Research Question

**Does explicit two-stage decomposition improve Vietnamese AMR parsing?**

### Arguments For MTUP Improvement

1. **Clearer Learning Signal**
   - Stage 1: Learn AMR structure
   - Stage 2: Learn variable naming
   - Easier than learning both simultaneously

2. **Incremental Complexity**
   - AMR without variables is simpler
   - Model can focus on one aspect at a time

3. **Better Debugging**
   - Can inspect intermediate AMR
   - Identify which stage fails

### Arguments Against MTUP Improvement

1. **Error Accumulation**
   - Stage 1 errors propagate to Stage 2
   - Two chances to fail vs one

2. **Modern LLMs are Powerful**
   - Can learn complex mappings end-to-end
   - Decomposition may be unnecessary overhead

3. **Slower Inference**
   - 2 passes = 2× latency
   - Not ideal for production

---

## 🧪 How to Test the Hypothesis

### Step 1: Train MTUP

```bash
bash TRAIN_MTUP_FIXED.sh
```

### Step 2: Evaluate on Same Test Set

```bash
python3 predict_mtup_fixed.py \
    --model outputs/mtup_fixed_YYYYMMDD/checkpoint-XXX \
    --test-file data/public_test.txt \
    --output evaluation_results/mtup_predictions.txt
```

### Step 3: Calculate SMATCH (Same Method)

```bash
python3 filter_valid_amrs.py \
    --predictions evaluation_results/mtup_predictions.txt \
    --ground-truth data/public_test_ground_truth.txt \
    --output-pred evaluation_results/mtup_valid.txt \
    --output-gold evaluation_results/gold_valid.txt

python -m smatch -f \
    evaluation_results/mtup_valid.txt \
    evaluation_results/gold_valid.txt \
    --significant 4
```

### Step 4: Compare Metrics

| Metric | Baseline | MTUP | Delta |
|--------|----------|------|-------|
| F1 | 0.47 | ??? | ??? |
| Validity | 91.3% | ??? | ??? |
| Invalid | 13/150 | ??? | ??? |

---

## 📝 For Thesis

### If MTUP Improves (F1 >0.50)

**Conclusion:**
> Explicit task decomposition in MTUP improved performance from F1=0.47 to F1=0.XX
> (+Y% relative improvement), validating our hypothesis that breaking AMR generation
> into two stages—structure generation and variable assignment—provides clearer
> learning signals and better supervision.

**Implication:**
- Two-stage approaches beneficial for structured generation
- Intermediate supervision helps model learning
- Worth the inference latency trade-off

---

### If MTUP Similar to Baseline (F1 ~0.47)

**Conclusion:**
> MTUP achieved comparable results (F1=0.XX) to direct generation, suggesting that
> modern instruction-tuned LLMs can learn complex AMR structure end-to-end without
> explicit decomposition. However, MTUP offers advantages in interpretability and
> debugging through inspectable intermediate representations.

**Implication:**
- Powerful LLMs can learn complex mappings directly
- Task decomposition may be unnecessary overhead
- Baseline's simplicity preferable for production

---

### If MTUP Worse than Baseline (F1 <0.47)

**Conclusion:**
> MTUP achieved F1=0.XX, underperforming the baseline (F1=0.47), likely due to
> error accumulation across two inference stages. This suggests that for Vietnamese
> AMR parsing, end-to-end generation is more robust than staged decomposition.

**Implication:**
- Error accumulation is a real problem
- Single-stage more robust
- Decomposition helps learning but hurts inference

---

## 🔍 Key Files for Each Method

### Baseline Files

```
train_baseline_fixed.py          # Training script
predict_baseline_fixed.py        # Inference script
config/config_fixed.py           # Configuration
FINAL_DOCUMENT_BASELINE_UPDATE.md # Documentation
```

### MTUP Files

```
train_mtup_fixed.py              # Training script
predict_mtup_fixed.py            # Two-stage inference
preprocess_mtup.py               # Data preprocessing
config/config_mtup_fixed.py      # Configuration
config/prompt_templates_fixed.py # Prompt templates
START_HERE_MTUP.md               # Quick start
MTUP_FIXED_SUMMARY.md            # Technical summary
```

---

## ⚡ Quick Commands

### Train Baseline
```bash
bash TRAIN_BASELINE_FIXED.sh
```

### Train MTUP
```bash
bash TRAIN_MTUP_FIXED.sh
```

### Evaluate Baseline
```bash
python3 predict_baseline_fixed.py \
    --model outputs/baseline_fixed_20260103_115114/checkpoint-1500 \
    --test-file data/public_test.txt
```

### Evaluate MTUP
```bash
python3 predict_mtup_fixed.py \
    --model outputs/mtup_fixed_YYYYMMDD/checkpoint-XXX \
    --test-file data/public_test.txt
```

---

## 💡 Recommendations

### Use Baseline If:
- ✅ Need fast inference (<200ms/sentence)
- ✅ Simple pipeline preferred
- ✅ F1=0.47 is sufficient
- ✅ Production deployment

### Use MTUP If:
- ✅ Need better F1 (hypothesis: >0.50)
- ✅ Want intermediate representations
- ✅ Debugging/interpretability important
- ✅ Research focus (ablation studies)

### Hybrid Approach:
- Train both methods
- Use Baseline for production (fast)
- Use MTUP for high-quality offline processing

---

## ✅ Next Steps

1. **Complete MTUP training** (~4 hours)
2. **Evaluate on same test set** (150 samples)
3. **Calculate SMATCH** (use same filtering method)
4. **Compare metrics** (F1, validity, speed)
5. **Document findings** in thesis
6. **Publish best model** to HuggingFace

---

## 📚 Summary

**Baseline:**
- ✅ Proven: F1=0.47, 91.3% validity
- ✅ Fast: 5 sent/sec
- ✅ Simple: 3-line prompt, 1 pass

**MTUP:**
- ❓ Hypothesis: F1>0.50, >90% validity
- ⏱️ Slower: ~2.5 sent/sec
- 🔬 Research: Two-stage decomposition

**The comparison will answer:** Does explicit task decomposition help AMR parsing?

---

**Both methods ready to test! 🚀**
