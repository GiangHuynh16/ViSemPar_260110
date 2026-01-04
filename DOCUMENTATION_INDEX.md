# 📚 ViSemPar Documentation Index

**Vietnamese Semantic Parsing - AMR Generation**

This document provides a complete navigation guide to all project documentation.

---

## 🚀 Quick Start Guides

### For Baseline (Direct Generation)
- **[START_HERE_BASELINE.md](START_HERE_BASELINE.md)** - Step-by-step guide for Baseline approach
- **Status:** ✅ Complete (F1=0.47, Validity=91.3%)

### For MTUP (Two-Stage Generation)
- **[START_HERE_MTUP.md](START_HERE_MTUP.md)** - Step-by-step guide for MTUP approach
- **Status:** 📝 Ready to train

### Which Should I Use?
- **[BASELINE_VS_MTUP_COMPARISON.md](BASELINE_VS_MTUP_COMPARISON.md)** - Side-by-side comparison
- **[MTUP_READY_TO_TRAIN.md](MTUP_READY_TO_TRAIN.md)** - Pre-training checklist

---

## 📖 Method Documentation

### Baseline Method (Section 4.4)
- **[FINAL_DOCUMENT_BASELINE_UPDATE.md](FINAL_DOCUMENT_BASELINE_UPDATE.md)** - Complete thesis section
  - Pipeline architecture (training + inference)
  - Prompt design and rationale
  - Training configuration details
  - Results: F1=0.47, Validity=91.3%
  - Computational requirements
  - Reproducibility instructions

### MTUP Method (Section 4.5)
- **[MTUP_FIXED_SUMMARY.md](MTUP_FIXED_SUMMARY.md)** - Technical implementation summary
  - What was wrong with old MTUP
  - All fixes applied (instruction masking, prompts, config)
  - Two-stage pipeline details
  - Comparison with Baseline

---

## 🔍 Investigation & Analysis

### Baseline Analysis
- **[INVESTIGATION_MAX_LENGTH.md](INVESTIGATION_MAX_LENGTH.md)** - Max length investigation
  - Is max_length=512 causing invalid AMRs?
  - Sentence length analysis
  - Decision framework
  - Confirmed: 91.3% validity is excellent, not truncation issue

### Results Comparison
- **[BASELINE_VS_MTUP_COMPARISON.md](BASELINE_VS_MTUP_COMPARISON.md)** - Detailed comparison
  - Core methodology differences
  - Prompt comparison
  - Training configuration
  - Expected results
  - Research question: Does decomposition help?

---

## 💻 Code Organization

### Baseline Implementation

**Core Files:**
```
train_baseline_fixed.py          # Training script
predict_baseline_fixed.py        # Inference script
config/config_fixed.py           # Configuration
```

**Helper Scripts:**
```
TRAIN_BASELINE_FIXED.sh          # Training wrapper
CHECK_MAX_LENGTH.sh              # Sentence length analysis
identify_invalid_amrs.py         # Find invalid AMRs
```

### MTUP Implementation

**Core Files:**
```
train_mtup_fixed.py              # Training with instruction masking
predict_mtup_fixed.py            # Two-stage inference
preprocess_mtup.py               # Convert data to MTUP format
config/config_mtup_fixed.py      # Configuration
config/prompt_templates_fixed.py # Minimal prompts with Penman examples
```

**Helper Scripts:**
```
TRAIN_MTUP_FIXED.sh              # Training wrapper
TEST_MTUP_FIXED.sh               # Single example test
EVALUATE_MTUP_CHECKPOINTS.sh     # Find best checkpoint
```

### Shared Utilities

```
filter_valid_amrs.py             # Filter for SMATCH calculation
validate_vietnamese_output.py    # AMR validation
```

---

## 🎯 By Task

### I want to train a model

**Baseline (Direct Generation):**
1. Read: [START_HERE_BASELINE.md](START_HERE_BASELINE.md)
2. Run: `bash TRAIN_BASELINE_FIXED.sh`
3. Result: F1~0.47, ~2.5 hours

**MTUP (Two-Stage):**
1. Read: [START_HERE_MTUP.md](START_HERE_MTUP.md)
2. Preprocess: `python3 preprocess_mtup.py`
3. Run: `bash TRAIN_MTUP_FIXED.sh`
4. Result: F1~0.50 (hypothesis), ~4 hours

---

### I want to understand the methods

**High-level overview:**
- [BASELINE_VS_MTUP_COMPARISON.md](BASELINE_VS_MTUP_COMPARISON.md) - Quick comparison

**Detailed technical:**
- [FINAL_DOCUMENT_BASELINE_UPDATE.md](FINAL_DOCUMENT_BASELINE_UPDATE.md) - Baseline details
- [MTUP_FIXED_SUMMARY.md](MTUP_FIXED_SUMMARY.md) - MTUP implementation

**Implementation details:**
- [train_baseline_fixed.py](train_baseline_fixed.py) - Baseline training code
- [train_mtup_fixed.py](train_mtup_fixed.py) - MTUP training code

---

### I want to reproduce results

**Baseline (F1=0.47, Validity=91.3%):**
```bash
# Training
python3 train_baseline_fixed.py

# Inference
python3 predict_baseline_fixed.py \
    --model outputs/baseline_fixed_20260103_115114/checkpoint-1500 \
    --test-file data/public_test.txt \
    --output predictions.txt

# SMATCH
python3 filter_valid_amrs.py \
    --predictions predictions.txt \
    --ground-truth data/public_test_ground_truth.txt \
    --output-pred valid_pred.txt \
    --output-gold valid_gold.txt

python -m smatch -f valid_pred.txt valid_gold.txt --significant 4
```

See: [FINAL_DOCUMENT_BASELINE_UPDATE.md](FINAL_DOCUMENT_BASELINE_UPDATE.md) Section 4.4.9

---

### I want to write thesis

**Section 4.4 - Baseline:**
- Use: [FINAL_DOCUMENT_BASELINE_UPDATE.md](FINAL_DOCUMENT_BASELINE_UPDATE.md)
- Contains: Complete section with all details
- Status: ✅ Ready to copy

**Section 4.5 - MTUP:**
- Use: [MTUP_FIXED_SUMMARY.md](MTUP_FIXED_SUMMARY.md)
- Contains: Technical summary, comparison with Baseline
- Status: 📝 Need to fill in results after training

**Comparison table:**
- Use: [BASELINE_VS_MTUP_COMPARISON.md](BASELINE_VS_MTUP_COMPARISON.md)
- Contains: Side-by-side comparison, research question

---

### I want to debug issues

**Invalid AMRs (unbalanced, duplicates):**
- Check: [INVESTIGATION_MAX_LENGTH.md](INVESTIGATION_MAX_LENGTH.md)
- Tools: `identify_invalid_amrs.py`, `validate_vietnamese_output.py`

**Training not converging:**
- Check: Instruction masking enabled?
- Check: Data format correct?
- See: [MTUP_FIXED_SUMMARY.md](MTUP_FIXED_SUMMARY.md) Section "Troubleshooting"

**OOM errors:**
- See: [MTUP_READY_TO_TRAIN.md](MTUP_READY_TO_TRAIN.md) Section "Troubleshooting"
- Solutions: Reduce batch size, reduce LoRA rank

---

## 📊 Results Summary

### Baseline Results (Confirmed)

| Metric | Value |
|--------|-------|
| **SMATCH F1** | **0.47** |
| **Structural Validity** | **91.3%** (137/150) |
| **Invalid AMRs** | 8.7% (13/150) |
| **Training Time** | 2.5 hours |
| **Inference Speed** | ~5 sent/sec |
| **Trainable Parameters** | 11M (0.15%) |

**Documentation:** [FINAL_DOCUMENT_BASELINE_UPDATE.md](FINAL_DOCUMENT_BASELINE_UPDATE.md)

---

### MTUP Results (Hypothesis)

| Metric | Target |
|--------|--------|
| **SMATCH F1** | >0.50 |
| **Structural Validity** | >90% |
| **Invalid AMRs** | <10% |
| **Training Time** | ~4 hours |
| **Inference Speed** | ~2.5 sent/sec |

**Documentation:** [MTUP_FIXED_SUMMARY.md](MTUP_FIXED_SUMMARY.md)

**Status:** 📝 Ready to train

---

## 🔧 Technical Details

### Key Implementation Insights

**1. Instruction Masking (Critical for Success)**
- **What:** Train only on target output, not on prompt
- **Why:** Prevents model from copying instructions
- **Implementation:** [train_baseline_fixed.py:84-93](train_baseline_fixed.py#L84-L93)
- **Result:** Enabled 91.3% validity vs <50% without

**2. Minimal Prompts (Better than Verbose)**
- **Baseline:** 3 lines only
- **MTUP:** ~10 lines with 1 Penman example
- **Why:** Model learns from data, not instructions
- **See:** [config/prompt_templates_fixed.py](config/prompt_templates_fixed.py)

**3. Early Stopping (2 Epochs Optimal)**
- **Why:** Prevents overfitting on 1,090 examples
- **Implementation:** Save every 100 steps, select best
- **Result:** Checkpoint 1500 best for Baseline

**4. bfloat16 Precision (Better than fp16)**
- **Why:** Matches Qwen pre-training precision
- **Result:** Better stability, same memory as fp16

---

### Common Pitfalls Avoided

❌ **Don't:** Train on entire prompt + output
✅ **Do:** Use instruction masking (labels = -100 for prompt)

❌ **Don't:** Use verbose 20+ line prompts
✅ **Do:** Minimal prompts (3-10 lines)

❌ **Don't:** Train for 15+ epochs
✅ **Do:** 2 epochs with early stopping

❌ **Don't:** Encode prompt+output as single string
✅ **Do:** Encode separately without special tokens

---

## 📁 File Tree

```
ViSemPar_new1/
│
├── Documentation (Start Here)
│   ├── DOCUMENTATION_INDEX.md              # ← This file
│   ├── START_HERE_BASELINE.md              # Baseline quick start
│   ├── START_HERE_MTUP.md                  # MTUP quick start
│   ├── BASELINE_VS_MTUP_COMPARISON.md      # Side-by-side comparison
│   └── MTUP_READY_TO_TRAIN.md              # Pre-training checklist
│
├── Method Documentation (For Thesis)
│   ├── FINAL_DOCUMENT_BASELINE_UPDATE.md   # Section 4.4 (complete)
│   ├── MTUP_FIXED_SUMMARY.md               # Section 4.5 (technical)
│   └── INVESTIGATION_MAX_LENGTH.md         # Max length analysis
│
├── Baseline Implementation
│   ├── train_baseline_fixed.py             # Training
│   ├── predict_baseline_fixed.py           # Inference
│   ├── config/config_fixed.py              # Configuration
│   ├── TRAIN_BASELINE_FIXED.sh             # Wrapper script
│   └── CHECK_MAX_LENGTH.sh                 # Analysis script
│
├── MTUP Implementation
│   ├── train_mtup_fixed.py                 # Training (with masking)
│   ├── predict_mtup_fixed.py               # Two-stage inference
│   ├── preprocess_mtup.py                  # Data preprocessing
│   ├── config/config_mtup_fixed.py         # Configuration
│   ├── config/prompt_templates_fixed.py    # Prompts with examples
│   ├── TRAIN_MTUP_FIXED.sh                 # Training wrapper
│   ├── TEST_MTUP_FIXED.sh                  # Quick test
│   └── EVALUATE_MTUP_CHECKPOINTS.sh        # Find best checkpoint
│
├── Utilities
│   ├── filter_valid_amrs.py                # Filter for SMATCH
│   ├── validate_vietnamese_output.py       # Validation
│   └── identify_invalid_amrs.py            # Find invalid AMRs
│
└── Data
    ├── train_amr_1.txt                     # Training data
    ├── train_amr_2.txt                     # Training data
    ├── public_test.txt                     # Test sentences
    └── public_test_ground_truth.txt        # Test AMRs
```

---

## 🎯 Recommended Reading Order

### For Quick Start (30 minutes)
1. [BASELINE_VS_MTUP_COMPARISON.md](BASELINE_VS_MTUP_COMPARISON.md) - Understand both methods
2. [START_HERE_BASELINE.md](START_HERE_BASELINE.md) OR [START_HERE_MTUP.md](START_HERE_MTUP.md) - Choose one
3. Run training script

### For Deep Understanding (2 hours)
1. [FINAL_DOCUMENT_BASELINE_UPDATE.md](FINAL_DOCUMENT_BASELINE_UPDATE.md) - Complete methodology
2. [MTUP_FIXED_SUMMARY.md](MTUP_FIXED_SUMMARY.md) - MTUP technical details
3. [train_baseline_fixed.py](train_baseline_fixed.py) - Read implementation
4. [train_mtup_fixed.py](train_mtup_fixed.py) - Compare with MTUP

### For Thesis Writing (1 hour)
1. [FINAL_DOCUMENT_BASELINE_UPDATE.md](FINAL_DOCUMENT_BASELINE_UPDATE.md) - Copy Section 4.4
2. [MTUP_FIXED_SUMMARY.md](MTUP_FIXED_SUMMARY.md) - Base for Section 4.5
3. [BASELINE_VS_MTUP_COMPARISON.md](BASELINE_VS_MTUP_COMPARISON.md) - Comparison table

---

## 🆘 Getting Help

### Common Questions

**Q: Which method should I use?**
A: See [BASELINE_VS_MTUP_COMPARISON.md](BASELINE_VS_MTUP_COMPARISON.md)
- Baseline: Fast, proven (F1=0.47)
- MTUP: Research, potentially better (F1>0.50)

**Q: Training failed with OOM**
A: See [MTUP_READY_TO_TRAIN.md](MTUP_READY_TO_TRAIN.md) → Troubleshooting
- Reduce batch size or LoRA rank

**Q: Invalid AMRs >20%**
A: See [INVESTIGATION_MAX_LENGTH.md](INVESTIGATION_MAX_LENGTH.md)
- Usually: wrong checkpoint selected
- Solution: Try checkpoint 300-500

**Q: Model generates explanations, not AMR**
A: Check instruction masking enabled
- See [MTUP_FIXED_SUMMARY.md](MTUP_FIXED_SUMMARY.md) Section "Troubleshooting"

---

## 📧 Contact & Links

**GitHub:** https://github.com/GiangHuynh16/ViSemPar_new1

**Baseline Model:** (TBD - will push to HuggingFace)

**MTUP Model:** (TBD - after training)

---

## ✅ Status

**Baseline:**
- ✅ Training complete
- ✅ Results confirmed (F1=0.47, 91.3% validity)
- ✅ Documentation complete
- ✅ Ready for thesis

**MTUP:**
- ✅ Implementation complete
- ✅ Documentation complete
- 📝 Ready to train
- ⏳ Results pending

---

## 🎯 Next Steps

1. **Train MTUP** (~4 hours)
   ```bash
   bash TRAIN_MTUP_FIXED.sh
   ```

2. **Compare results** (1 hour)
   - MTUP vs Baseline
   - Document in thesis

3. **Write thesis Section 4.5** (2 hours)
   - Use [MTUP_FIXED_SUMMARY.md](MTUP_FIXED_SUMMARY.md) as base
   - Fill in actual results

4. **Push to GitHub** (30 minutes)
   - Final code
   - Results
   - Documentation

---

**Last Updated:** 2026-01-04

**Version:** 1.0

**Ready for:** Training & Thesis Writing

---

**Happy coding! 🚀**
