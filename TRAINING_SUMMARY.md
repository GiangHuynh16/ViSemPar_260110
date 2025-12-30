# Vietnamese AMR Training Summary

**Date**: 2025-12-30
**Status**: MTUP completed ✅, Baseline ready to train ⏳

---

## Training Overview

| Model | Approach | Status | F1 (Expected) |
|-------|----------|--------|---------------|
| **MTUP 7B** | Multi-Task (2 prompts) | ✅ **COMPLETED & PUSHED** | 0.51-0.52 |
| **Baseline 7B** | Single-Task (1 prompt) | ⏳ **READY TO TRAIN** | TBD |

---

## 1. MTUP 7B (Multi-Task) - ✅ COMPLETED

### Approach
- **Stage 1**: Sentence → AMR structure (no variables)
- **Stage 2**: Add variable binding to structure
- **Prompt**: Vietnamese multi-task unified prompt (2 stages in 1 prompt)

### Results
- **Status**: Training completed, model pushed to HuggingFace
- **Checkpoint**: `outputs/checkpoints_mtup/mtup_full_training_final`
- **Training Time**: ~12-15 hours
- **Config**: `config/config_mtup.py`
- **Expected F1**: 0.51-0.52 (based on 7B vs 3B improvement)

### Key Files
- Training: `train_mtup.py`
- Config: `config/config_mtup.py`
- Evaluation: `evaluate_mtup_model.py`
- Launcher: `START_MTUP_7B_TRAINING.sh`
- Docs: `UPGRADE_TO_7B_MODEL.md`

---

## 2. Baseline 7B (Single-Task) - ⏳ READY TO TRAIN

### Approach
- **Single Stage**: Direct Sentence → Complete AMR
- **Prompt**: English instruction-style prompt (standard single-task)
- **No decomposition**: Model learns full AMR generation in one step

### Configuration
- **Model**: Qwen/Qwen2.5-7B-Instruct (same as MTUP)
- **LoRA rank**: 128 (same as MTUP)
- **Training epochs**: 15 (same as MTUP)
- **Batch size**: 2, grad accum 8 (effective=16, same as MTUP)
- **Learning rate**: 2e-4 (same as MTUP)

### Files Created
- ✅ Training: `train_baseline.py`
- ✅ Config: `config/config.py` (updated to 7B)
- ✅ Evaluation: `evaluate_baseline_model.py`
- ✅ Launcher: `START_BASELINE_7B_TRAINING.sh`
- ✅ Docs: `BASELINE_7B_TRAINING_GUIDE.md`

### How to Train

```bash
# Start tmux session
tmux new -s baseline_7b

# Run training
bash START_BASELINE_7B_TRAINING.sh
```

Or manually:

```bash
python train_baseline.py --epochs 15 --show-sample
```

### Expected Output
- **Checkpoint**: `outputs/checkpoints/baseline_7b_final`
- **Training Time**: ~12-15 hours (same as MTUP)
- **Trainable Params**: ~239M (same as MTUP)

---

## 3. Key Differences: MTUP vs Baseline

### Prompt Format

**MTUP (Vietnamese, 2-stage)**:
```
### NHIỆM VỤ: Chuyển đổi câu tiếng Việt sang AMR (2 bước)

### Câu cần phân tích:
{sentence}

### Kết quả phân tích:

## Bước 1 - Tạo cấu trúc AMR (chưa có biến):
[Model generates structure without variables]

## Bước 2 - Gán biến và hoàn thiện:
AMR hoàn chỉnh:
[Model generates final AMR with variables]
```

**Baseline (English, 1-stage)**:
```
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Convert the following Vietnamese sentence to Abstract Meaning Representation (AMR) format. Ensure proper concept alignment and preserve co-references.

### Input:
{sentence}

### Response:
[Model generates complete AMR directly]
```

### Training Data Format

**MTUP**:
- Input: Sentence
- Output: Two-stage response (structure + variables)
- Supervision: 2 signals (one per stage)

**Baseline**:
- Input: Sentence
- Output: Complete AMR
- Supervision: 1 signal (final AMR)

### Theoretical Advantages

**MTUP**:
- ✅ Explicit task decomposition → Easier learning
- ✅ Intermediate supervision → Better gradients
- ✅ Handles variable binding separately → Less confusion
- ✅ Vietnamese prompts → Language consistency

**Baseline**:
- ✅ Simpler training pipeline
- ✅ Faster inference (1 generation vs 2 stages)
- ✅ No risk of error propagation between stages
- ✅ Standard approach (easier to compare with literature)

---

## 4. Evaluation & Comparison

### After Baseline Training Completes

```bash
# 1. Evaluate baseline
python evaluate_baseline_model.py \
  --checkpoint outputs/checkpoints/baseline_7b_final \
  --test-file data/public_test_ground_truth.txt \
  --output results/baseline_7b_evaluation.json

# 2. Compare results
echo "=== MTUP 7B ==="
cat results/mtup_7b_evaluation.json

echo ""
echo "=== Baseline 7B ==="
cat results/baseline_7b_evaluation.json
```

### Expected Comparison

| Metric | MTUP 7B (Expected) | Baseline 7B (TBD) |
|--------|-------------------|-------------------|
| F1 | 0.51-0.52 | ? |
| Precision | ~0.52 | ? |
| Recall | ~0.50 | ? |
| Success Rate | 90%+ (135-140/150) | ? |

**Hypothesis**: MTUP should outperform baseline due to explicit task decomposition and intermediate supervision.

---

## 5. Next Steps

### Immediate (Today)
1. ⏳ Start baseline 7B training (12-15 hours)
2. ⏳ Monitor training progress
3. ⏳ Wait for completion

### After Training (Tomorrow)
1. ⏳ Evaluate baseline on public test
2. ⏳ Compare MTUP vs Baseline results
3. ⏳ Analyze error patterns
4. ⏳ Push baseline to HuggingFace (if good results)

### Thesis Integration
1. ⏳ Update results table with baseline 7B
2. ⏳ Add MTUP vs Baseline comparison section
3. ⏳ Statistical significance testing
4. ⏳ Error analysis and discussion
5. ⏳ Final conclusions

---

## 6. Commands Cheat Sheet

### Training
```bash
# MTUP (already done)
bash START_MTUP_7B_TRAINING.sh

# Baseline (ready to run)
bash START_BASELINE_7B_TRAINING.sh
```

### Evaluation
```bash
# MTUP
python evaluate_mtup_model.py \
  --checkpoint outputs/checkpoints_mtup/mtup_full_training_final \
  --test-file data/public_test_ground_truth.txt \
  --output results/mtup_7b_evaluation.json

# Baseline
python evaluate_baseline_model.py \
  --checkpoint outputs/checkpoints/baseline_7b_final \
  --test-file data/public_test_ground_truth.txt \
  --output results/baseline_7b_evaluation.json
```

### Monitoring
```bash
# Attach to tmux
tmux attach -t mtup_7b      # For MTUP
tmux attach -t baseline_7b  # For Baseline

# Check logs
tail -f logs/training.log
tail -f logs/training_mtup.log

# GPU usage
watch -n 1 nvidia-smi
```

### HuggingFace Upload
```bash
# Check username
huggingface-cli whoami

# Upload MTUP (already done)
hf upload YOUR-USERNAME/vietnamese-amr-mtup-7b \
  outputs/checkpoints_mtup/mtup_full_training_final

# Upload Baseline (after evaluation)
hf upload YOUR-USERNAME/vietnamese-amr-baseline-7b \
  outputs/checkpoints/baseline_7b_final
```

---

## 7. File Organization

```
ViSemPar_new1/
├── config/
│   ├── config.py              # Baseline config (7B) ✅
│   └── config_mtup.py         # MTUP config (7B) ✅
├── train_baseline.py          # Baseline training ✅
├── train_mtup.py              # MTUP training ✅
├── evaluate_baseline_model.py # Baseline eval ✅
├── evaluate_mtup_model.py     # MTUP eval ✅
├── START_BASELINE_7B_TRAINING.sh  ✅
├── START_MTUP_7B_TRAINING.sh      ✅
├── BASELINE_7B_TRAINING_GUIDE.md  ✅
├── UPGRADE_TO_7B_MODEL.md         ✅
├── TRAINING_SUMMARY.md            ✅ (this file)
└── outputs/
    └── checkpoints/
        ├── baseline_7b_final/     ⏳ (after training)
        └── checkpoints_mtup/
            └── mtup_full_training_final/  ✅
```

---

## 8. Timeline

| Date | Task | Status |
|------|------|--------|
| 2025-12-29 | MTUP 7B training started | ✅ |
| 2025-12-30 | MTUP 7B completed & pushed | ✅ |
| 2025-12-30 | Baseline setup completed | ✅ |
| 2025-12-30 | **Baseline training (start now)** | ⏳ |
| 2025-12-31 | Baseline training complete | ⏳ |
| 2025-12-31 | Evaluation & comparison | ⏳ |
| 2025-12-31 | Thesis update | ⏳ |

---

**Ready to train baseline**: Run `bash START_BASELINE_7B_TRAINING.sh` in tmux!
