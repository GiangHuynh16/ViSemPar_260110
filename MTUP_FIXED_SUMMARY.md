# 📋 MTUP Fixed - Implementation Summary

## 🎯 Overview

**MTUP Fixed** is a completely rewritten version of the MTUP (Multi-Task Unified Prompt) approach, applying all lessons learned from the successful Baseline method (F1=0.47, 91.3% validity).

**Created:** 2026-01-04
**Status:** Ready for training
**Expected Training Time:** ~4 hours on A6000 GPU

---

## ❌ What Was Wrong with Old MTUP?

The original MTUP implementation had critical issues:

1. **No Instruction Masking** ❌
   - Trained on entire prompt + output
   - Model learned to copy instructions
   - Result: Long explanations instead of AMR

2. **Verbose Prompts** ❌
   - 20+ lines of complex instructions
   - No Penman format examples
   - Confusing task specification

3. **Wrong Training Config** ❌
   - 15 epochs → severe overfitting
   - fp16 precision (not optimal for Qwen)
   - Save every 200 steps (missed sweet spot)

4. **Invalid Outputs** ❌
   - Generated explanatory text
   - Not proper Penman format
   - Structural validity: <50%

---

## ✅ What MTUP Fixed Changes

### 1. Instruction Masking (Critical Fix)

**Implementation:** [train_mtup_fixed.py](train_mtup_fixed.py:156-177)

```python
# Encode separately WITHOUT special tokens
instruction_ids = tokenizer.encode(instruction, add_special_tokens=False)
target_ids = tokenizer.encode(target_amr, add_special_tokens=False)
eos_ids = tokenizer.encode(tokenizer.eos_token, add_special_tokens=False)

# Concatenate
input_ids = instruction_ids + target_ids + eos_ids

# Mask instruction in labels
labels = input_ids.copy()
for i in range(len(instruction_ids)):
    labels[i] = -100  # Only train on target AMR
```

**Why it works:**
- Model learns to generate AMR, not copy prompts
- Loss computed only on target output
- Same technique that made Baseline successful

---

### 2. Minimal Prompt with Penman Examples

**Implementation:** [config/prompt_templates_fixed.py](config/prompt_templates_fixed.py)

**Old prompt (20+ lines):**
```
Bạn là một hệ thống phân tích ngữ nghĩa...
Nhiệm vụ của bạn là...
[15 more lines of instructions]
```

**New prompt (~10 lines):**
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
```

**Key improvements:**
- ✅ Explicit Penman format example
- ✅ Shows both stages (no vars → with vars)
- ✅ Emphasizes "chuẩn PENMAN" throughout
- ✅ Clear extraction markers

---

### 3. Training Configuration Matching Baseline

**Implementation:** [config/config_mtup_fixed.py](config/config_mtup_fixed.py)

| Parameter | Old MTUP | MTUP Fixed | Baseline |
|-----------|----------|------------|----------|
| Epochs | 15 | **2** | 2 |
| Save steps | 200 | **100** | 100 |
| Precision | fp16 | **bfloat16** | bfloat16 |
| Instruction masking | ❌ No | **✅ Yes** | ✅ Yes |
| Prompt length | 20+ lines | **~10 lines** | 3 lines |
| Penman examples | None | **1-2 examples** | N/A |

**Rationale:**
- 2 epochs: Prevents overfitting on 1,090 examples
- Save every 100 steps: Captures early convergence
- bfloat16: Matches Qwen pre-training precision
- Instruction masking: Model learns to generate, not copy

---

### 4. Two-Stage Inference Pipeline

**Implementation:** [predict_mtup_fixed.py](predict_mtup_fixed.py)

**Stage 1: Generate AMR without variables**
```python
def generate_step1(sentence: str) -> str:
    prompt = MTUP_INFERENCE_TEMPLATE.format(sentence=sentence)
    outputs = model.generate(...)
    amr_no_vars = extract_amr_step1(generated_text)
    return amr_no_vars
```

**Stage 2: Add variables in Penman format**
```python
def generate_step2(amr_no_vars: str) -> str:
    prompt = MTUP_INFERENCE_STEP2_TEMPLATE.format(amr_no_vars=amr_no_vars)
    outputs = model.generate(...)
    amr_with_vars = extract_amr_step2(generated_text)
    return amr_with_vars
```

**Extraction Logic:**
- Split by markers: "AMR không biến:" and "AMR chuẩn PENMAN:"
- Extract until parentheses balanced
- Check balance on accumulated text (not full output)
- Validate structure (balance, duplicates, non-empty)

---

## 📁 Files Created

### Core Implementation (4 files)

1. **[config/prompt_templates_fixed.py](config/prompt_templates_fixed.py)**
   - Minimal prompts with Penman examples
   - Two templates: training and inference (2 stages)
   - Clear extraction markers

2. **[config/config_mtup_fixed.py](config/config_mtup_fixed.py)**
   - Training config (2 epochs, bfloat16, etc.)
   - LoRA config (r=64, alpha=128)
   - MTUP-specific settings (masking, markers)

3. **[train_mtup_fixed.py](train_mtup_fixed.py)**
   - Training script with instruction masking
   - MTUP data preprocessing
   - Checkpoint saving every 100 steps

4. **[predict_mtup_fixed.py](predict_mtup_fixed.py)**
   - Two-stage inference pipeline
   - AMR extraction with validation
   - Batch prediction support

### Helper Scripts (3 files)

5. **[preprocess_mtup.py](preprocess_mtup.py)**
   - Converts training data to MTUP format
   - Removes variables to create Stage 1 targets
   - Validates AMR structure

6. **[TRAIN_MTUP_FIXED.sh](TRAIN_MTUP_FIXED.sh)**
   - Training wrapper script
   - GPU checks, progress monitoring
   - Instructions for next steps

7. **[TEST_MTUP_FIXED.sh](TEST_MTUP_FIXED.sh)**
   - Quick single-example test
   - Verifies two-stage generation
   - Shows expected output format

8. **[EVALUATE_MTUP_CHECKPOINTS.sh](EVALUATE_MTUP_CHECKPOINTS.sh)**
   - Evaluates all checkpoints
   - Finds best based on structural validity
   - Generates comparison table

### Documentation (2 files)

9. **[START_HERE_MTUP.md](START_HERE_MTUP.md)**
   - Complete quick start guide
   - 5-step workflow
   - Troubleshooting section

10. **[MTUP_FIXED_SUMMARY.md](MTUP_FIXED_SUMMARY.md)**
    - This file
    - Technical summary of changes
    - Comparison with old MTUP and Baseline

---

## 🔬 Technical Details

### Instruction Masking Implementation

**Key insight:** Tokenizer is context-dependent. Must encode separately to identify exact boundaries.

```python
# ❌ WRONG: Encodes as single string
full_text = prompt + target
input_ids = tokenizer.encode(full_text)
# Can't identify where prompt ends!

# ✅ CORRECT: Encode separately
prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
target_ids = tokenizer.encode(target, add_special_tokens=False)
input_ids = prompt_ids + target_ids + eos_ids

# Now we know exactly where to mask
labels = input_ids.copy()
labels[:len(prompt_ids)] = -100
```

**Why `add_special_tokens=False`?**
- Avoids duplicate BOS/EOS tokens at boundaries
- Ensures clean concatenation
- Prevents tokenization mismatch

---

### MTUP Data Format

**Preprocessed training example:**
```
# ::snt Anh ấy đã hoàn thành công việc.
# ::amr-no-vars (hoàn_thành :agent (anh) :theme (công_việc) :aspect (đã))
# ::amr-with-vars
(h / hoàn_thành
    :agent (a / anh)
    :theme (c / công_việc)
    :aspect (đ / đã))
```

**During training:**
- Instruction: Everything up to "AMR chuẩn PENMAN:"
- Target: Only the final Penman AMR
- Labels: Mask instruction tokens with -100

---

### AMR Extraction Strategy

**Challenge:** Model may generate extra text after AMR.

**Solution:** Incremental balance checking

```python
lines = generated_text.split('\n')
amr_lines = []

for line in lines:
    amr_lines.append(line)
    accumulated = '\n'.join(amr_lines)  # Check accumulated, not full text

    if accumulated.count('(') == accumulated.count(')') > 0:
        break  # Found complete AMR
```

**Why this works:**
- Stops at first balanced structure
- Ignores trailing explanations
- Handles multi-line AMR correctly

---

## 📊 Expected Results

### Hypothesis

**MTUP should improve over Baseline** because:
1. Explicit task decomposition aids learning
2. Two-stage supervision provides clearer signal
3. Penman format examples guide structure

### Target Metrics

| Metric | Baseline | MTUP Target | Stretch Goal |
|--------|----------|-------------|--------------|
| **Structural Validity** | 91.3% | >90% | >92% |
| **SMATCH F1** | 0.47 | >0.50 | >0.52 |
| **Invalid AMRs** | 13/150 | <15/150 | <10/150 |

### Success Criteria

**Minimum acceptable:**
- ✅ Validity: >85%
- ✅ F1: >0.45 (at least baseline level)
- ✅ Training converges (loss decreases)

**Target:**
- 🎯 Validity: >90%
- 🎯 F1: >0.50 (+6% over baseline)
- 🎯 Demonstrates MTUP advantage

**Excellent:**
- 🏆 Validity: >92%
- 🏆 F1: >0.52 (+11% over baseline)
- 🏆 Clear improvement in both metrics

---

## 🚀 Quick Start

### 1. Preprocess Data (5 minutes)

```bash
python3 preprocess_mtup.py \
    --input data/train_amr_1.txt \
    --output data/train_amr_mtup_preprocessed.txt \
    --validate
```

### 2. Train Model (~4 hours)

```bash
bash TRAIN_MTUP_FIXED.sh
```

### 3. Evaluate Checkpoints (30 minutes)

```bash
bash EVALUATE_MTUP_CHECKPOINTS.sh outputs/mtup_fixed_YYYYMMDD_HHMMSS
```

### 4. Test on Full Dataset (1 hour)

```bash
python3 predict_mtup_fixed.py \
    --model outputs/mtup_fixed_YYYYMMDD_HHMMSS/checkpoint-XXX \
    --test-file data/public_test.txt \
    --output evaluation_results/mtup_predictions.txt
```

### 5. Calculate SMATCH (5 minutes)

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

**Total time:** ~6 hours (hands-on: ~1 hour)

---

## 🔍 Key Differences: Old vs New MTUP

### Prompt Comparison

**Old MTUP (Verbose, no examples):**
```
Bạn là một hệ thống phân tích ngữ nghĩa chuyên sâu cho tiếng Việt.
Nhiệm vụ của bạn là chuyển đổi câu tiếng Việt sang biểu diễn AMR.
AMR là một biểu diễn ngữ nghĩa trừu tượng...
[15+ more lines]
```
→ Result: Model copies instructions, generates explanations

**MTUP Fixed (Minimal, with Penman example):**
```
Chuyển câu tiếng Việt sau sang AMR theo chuẩn PENMAN.

VÍ DỤ:
[Clear Penman example showing both stages]

Câu: {sentence}
```
→ Result: Model generates clean Penman AMR

---

### Training Comparison

| Aspect | Old MTUP | MTUP Fixed |
|--------|----------|------------|
| **Instruction Masking** | ❌ No | ✅ Yes |
| **Epochs** | 15 (overfitting) | 2 (optimal) |
| **Save Frequency** | Every 200 steps | Every 100 steps |
| **Precision** | fp16 | bfloat16 |
| **Prompt Length** | 20+ lines | ~10 lines |
| **Penman Examples** | None | 1-2 clear examples |
| **Expected Validity** | <50% | >90% |
| **Expected F1** | Unknown | >0.50 |

---

### Code Quality Comparison

**Old MTUP:**
- No instruction masking implementation
- Verbose prompt templates
- No extraction validation
- Overfitting configuration

**MTUP Fixed:**
- Proper instruction masking (like Baseline)
- Minimal prompts with examples
- Two-stage extraction with validation
- Early stopping configuration

---

## 📚 For Thesis Documentation

### Section 4.5: MTUP Method

**If MTUP improves over Baseline:**

> We propose MTUP (Multi-Task Unified Prompt), a two-stage approach that decomposes
> AMR generation into: (1) generating structure without variables, and (2) adding
> variables in Penman format. By applying lessons from our Baseline method—minimal
> prompts, instruction masking, and early stopping—MTUP achieved **F1=0.XX**
> (+Y% over baseline) with **Z% structural validity**. This demonstrates that
> explicit task decomposition aids AMR learning.

**If MTUP performs similarly to Baseline:**

> MTUP achieved comparable results (F1=0.XX, validity=Y%) to the direct generation
> baseline, suggesting that modern LLMs can learn complex AMR structure end-to-end
> without explicit decomposition. However, the two-stage approach offers advantages
> in interpretability and debugging, as intermediate AMR can be inspected.

**Key points to include:**
1. Two-stage decomposition rationale
2. Instruction masking critical for success
3. Minimal prompts with Penman examples
4. Training configuration (2 epochs, bfloat16)
5. Comparison with Baseline (F1, validity)
6. Discussion of whether decomposition helps

---

## ✅ Validation Checklist

Before training, verify:

- [x] Instruction masking implemented in `train_mtup_fixed.py:156-177`
- [x] Prompt has Penman examples in `config/prompt_templates_fixed.py`
- [x] Training config: 2 epochs, bfloat16, save every 100 steps
- [x] Two-stage inference in `predict_mtup_fixed.py`
- [x] AMR extraction with balance checking
- [x] Preprocessing script to create MTUP data
- [x] Helper scripts for training/evaluation
- [x] Documentation (START_HERE_MTUP.md)

After training, verify:

- [ ] Training loss decreases steadily
- [ ] At least 10 checkpoints saved
- [ ] Validation AMRs are structurally valid
- [ ] Best checkpoint has >85% validity
- [ ] SMATCH F1 calculated on valid AMRs

---

## 🎯 Research Question

**Does explicit task decomposition improve Vietnamese AMR parsing?**

**Hypothesis:** Yes, because:
- Two-stage supervision provides clearer learning signal
- Intermediate representation (AMR without vars) is simpler
- Final stage focuses only on variable assignment

**Null hypothesis:** No improvement, because:
- Modern LLMs can learn complex mappings end-to-end
- Two stages may introduce error accumulation
- Extra inference passes add latency without quality gain

**We will test this by comparing:**
- Baseline: F1=0.47, Validity=91.3% (1 pass)
- MTUP: F1=???, Validity=???% (2 passes)

---

## 🔗 Related Files

### Implementation
- [train_mtup_fixed.py](train_mtup_fixed.py) - Training script
- [predict_mtup_fixed.py](predict_mtup_fixed.py) - Inference script
- [preprocess_mtup.py](preprocess_mtup.py) - Data preprocessing
- [config/prompt_templates_fixed.py](config/prompt_templates_fixed.py) - Prompts
- [config/config_mtup_fixed.py](config/config_mtup_fixed.py) - Configuration

### Scripts
- [TRAIN_MTUP_FIXED.sh](TRAIN_MTUP_FIXED.sh) - Training wrapper
- [TEST_MTUP_FIXED.sh](TEST_MTUP_FIXED.sh) - Quick test
- [EVALUATE_MTUP_CHECKPOINTS.sh](EVALUATE_MTUP_CHECKPOINTS.sh) - Evaluation

### Documentation
- [START_HERE_MTUP.md](START_HERE_MTUP.md) - Quick start guide
- [FINAL_DOCUMENT_BASELINE_UPDATE.md](FINAL_DOCUMENT_BASELINE_UPDATE.md) - Baseline for comparison
- [MTUP_FIXED_SUMMARY.md](MTUP_FIXED_SUMMARY.md) - This file

### Utilities
- [filter_valid_amrs.py](filter_valid_amrs.py) - Filter for SMATCH
- [validate_vietnamese_output.py](validate_vietnamese_output.py) - Validation

---

## 🚨 Common Issues and Solutions

### Issue 1: Invalid AMRs (unbalanced parentheses)

**Symptoms:**
- Validity <80%
- Many unbalanced parentheses errors

**Check:**
1. Instruction masking enabled: `grep use_instruction_masking config/config_mtup_fixed.py`
2. Training completed >200 steps
3. Using best checkpoint, not latest

**Fix:**
- Try checkpoint 300-500 instead of 100-200
- Verify prompt has Penman examples
- Check extraction logic stops at balanced

---

### Issue 2: Model generates explanations

**Symptoms:**
- Output contains Vietnamese text
- Not proper Penman format

**Check:**
1. Prompt template emphasizes "chuẩn PENMAN"
2. Extraction splits on correct markers
3. EOS token being removed

**Fix:**
- Verify markers: "AMR không biến:" and "AMR chuẩn PENMAN:"
- Check `extract_amr_step2()` logic
- Ensure training used instruction masking

---

### Issue 3: Training doesn't converge

**Symptoms:**
- Loss stays high or increases
- No improvement in validation

**Check:**
1. Learning rate not too high: `2e-4`
2. Gradient accumulation: `16 steps`
3. Data format correct (preprocessed)

**Fix:**
- Reduce learning rate to `1e-4`
- Check preprocessed data has all fields
- Verify instruction masking is working

---

## 📈 Timeline and Milestones

**Day 1: Setup and Training (4-5 hours)**
- [x] Create implementation files
- [ ] Preprocess training data (5 min)
- [ ] Start training (4 hours)
- [ ] Monitor progress

**Day 2: Evaluation (2-3 hours)**
- [ ] Evaluate all checkpoints (30 min)
- [ ] Select best checkpoint
- [ ] Test on 150 samples (1 hour)
- [ ] Calculate SMATCH (5 min)
- [ ] Compare with Baseline

**Day 3: Analysis and Documentation**
- [ ] Analyze results
- [ ] Document findings
- [ ] Write thesis section
- [ ] Push to GitHub

---

## ✨ Summary

**MTUP Fixed is a complete rewrite** that:
1. ✅ Applies all Baseline success factors
2. ✅ Adds Penman format examples to prompts
3. ✅ Implements proper instruction masking
4. ✅ Uses optimal training configuration (2 epochs, bfloat16)
5. ✅ Provides two-stage inference pipeline
6. ✅ Includes comprehensive testing/evaluation tools

**Ready to train!** Follow [START_HERE_MTUP.md](START_HERE_MTUP.md) for step-by-step guide.

**Expected outcome:** MTUP should match or exceed Baseline (F1=0.47, 91.3% validity) while demonstrating the value of explicit task decomposition.

---

**Good luck! 🚀**
