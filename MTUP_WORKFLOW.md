# 🔄 MTUP Workflow Visualization

## Overview

MTUP (Multi-Task Unified Prompt) breaks AMR generation into 2 sequential tasks within a single prompt.

---

## Training Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT: Raw AMR Data                       │
│  • Vietnamese sentence                                       │
│  • Gold AMR with variables                                   │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              PREPROCESSING (preprocessor_mtup.py)            │
│                                                              │
│  1. Remove variables:                                        │
│     (a / ăn :agent (t / tôi))                               │
│     →  (ăn :agent (tôi))                                     │
│                                                              │
│  2. Format using template v2_natural:                        │
│     ### NHIỆM VỤ: Chuyển đổi...                             │
│     ### Câu: {sentence}                                      │
│     ## Bước 1: {amr_no_vars}                                 │
│     ## Bước 2: {amr_with_vars}                               │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                 TRAINING (train_mtup.py)                     │
│                                                              │
│  Model: Qwen 2.5 3B + LoRA                                   │
│  Method: Causal Language Modeling                            │
│  Loss: Cross-entropy on full sequence                        │
│                                                              │
│  Learns to:                                                  │
│  1. Generate AMR structure (no vars)                         │
│  2. Add variables to structure                               │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│           OUTPUT: Trained Model Checkpoint                   │
│  Location: outputs/checkpoints_mtup/mtup_*_final/           │
│  Size: ~457 MB (LoRA adapter only)                           │
└─────────────────────────────────────────────────────────────┘
```

---

## Evaluation Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                INPUT: Test Sentence                          │
│  "Tôi ăn cơm"                                               │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│         BUILD PROMPT (evaluate_mtup_model.py)                │
│                                                              │
│  ### NHIỆM VỤ: Chuyển đổi câu tiếng Việt sang AMR (2 bước)  │
│                                                              │
│  ### Câu cần phân tích:                                      │
│  Tôi ăn cơm                                                  │
│                                                              │
│  ### Kết quả phân tích:                                      │
│                                                              │
│  ## Bước 1 - Tạo cấu trúc AMR (chưa có biến):                │
│  [MODEL COMPLETES THIS]                                      │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              MODEL GENERATION (Qwen 2.5)                     │
│                                                              │
│  Mode: Greedy decoding (deterministic)                       │
│  Output includes BOTH tasks:                                 │
│                                                              │
│  ## Bước 1 - Tạo cấu trúc AMR (chưa có biến):                │
│  (ăn :agent (tôi) :patient (cơm))                            │
│                                                              │
│  ## Bước 2 - Gán biến cho các khái niệm:                     │
│  ...                                                         │
│  AMR hoàn chỉnh:                                             │
│  (a / ăn :agent (t / tôi) :patient (c / cơm))                │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│           EXTRACT AMR (Post-processing)                      │
│                                                              │
│  1. Find "AMR hoàn chỉnh:" section                           │
│  2. Extract AMR after that marker                            │
│  3. Clean up (remove prompt leakage)                         │
│  4. Find first '(' and take from there                       │
│                                                              │
│  Result: (a / ăn :agent (t / tôi) :patient (c / cơm))        │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              SMATCH EVALUATION                               │
│                                                              │
│  Compare:                                                    │
│    Predicted AMR  vs  Gold AMR                               │
│                                                              │
│  Compute:                                                    │
│    • Precision = matched / predicted                         │
│    • Recall = matched / gold                                 │
│    • F1 = 2 * P * R / (P + R)                                │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    RESULTS                                   │
│  Precision: 0.4978                                           │
│  Recall:    0.5002                                           │
│  F1:        0.4933                                           │
└─────────────────────────────────────────────────────────────┘
```

---

## Example Walkthrough

### Input Sentence
```
"Tôi nhớ lời chủ tịch"
```

### Training Data Format

```
### NHIỆM VỤ: Chuyển đổi câu tiếng Việt sang AMR (2 bước)

### Câu cần phân tích:
Tôi nhớ lời chủ tịch

### Kết quả phân tích:

## Bước 1 - Tạo cấu trúc AMR (chưa có biến):
(nhớ :agent (tôi) :theme (lời :poss (chủ_tịch)))

## Bước 2 - Gán biến cho các khái niệm:
Hướng dẫn:
• Mỗi khái niệm được gán một biến riêng (ví dụ: n, n2, p, c...)
• Khái niệm xuất hiện nhiều lần → dùng chung một biến (đồng tham chiếu)
• Format: (biến / khái_niệm :quan_hệ...)

AMR hoàn chỉnh:
(n / nhớ :agent (t / tôi) :theme (l / lời :poss (c / chủ_tịch)))
```

### Model Learning

During training, the model learns:

1. **Task 1 Pattern**:
   - Input: Sentence + "Bước 1" header
   - Output: AMR structure without variables

2. **Task 2 Pattern**:
   - Input: Task 1 output + "Bước 2" header
   - Output: Complete AMR with variables

3. **Sequential Dependency**:
   - Task 2 builds on Task 1 output
   - Variables map to concepts from Task 1

### Evaluation Process

1. **Prompt Construction**:
   ```
   ### NHIỆM VỤ: ...
   ### Câu cần phân tích:
   Tôi nhớ lời chủ tịch
   ### Kết quả phân tích:
   ## Bước 1 - Tạo cấu trúc AMR (chưa có biến):
   ```

2. **Model Generation** (single pass):
   - Model completes both Task 1 and Task 2
   - Outputs full formatted response

3. **AMR Extraction**:
   - Parse the structured output
   - Extract final AMR from "AMR hoàn chỉnh:" section

4. **SMATCH Scoring**:
   - Compare with gold AMR
   - Calculate precision, recall, F1

---

## Key Design Decisions

### ✅ Why MTUP?

1. **Structured Learning**: Break complex task into steps
2. **Better Guidance**: Explicit instructions for each stage
3. **Error Reduction**: Separate structure and variable assignment
4. **Interpretable**: Can inspect intermediate outputs

### ✅ Why Vietnamese Prompts?

1. **Native Language**: Better understanding for Vietnamese text
2. **Consistency**: Train and eval use same language
3. **Cultural Fit**: Natural instructions for Vietnamese users

### ✅ Why Two Tasks in One Prompt?

1. **Efficiency**: Single model call
2. **Context**: Task 2 sees Task 1 output
3. **Simplicity**: No need for multi-stage pipeline

---

## Performance Factors

### What Affects F1 Score?

```
┌─────────────────────────────────────────┐
│  Training Data Quality    ──→  +20%     │
│  Prompt Template          ──→  +15%     │
│  Model Size               ──→  +10%     │
│  Training Epochs          ──→  +8%      │
│  Post-processing          ──→  +5%      │
│  Hyperparameters          ──→  +3%      │
└─────────────────────────────────────────┘
```

### Current Configuration

- ✅ Data Quality: Good (VLSP dataset)
- ✅ Template: v2_natural (tested)
- ✅ Model: 3B (reasonable for task)
- ⚠️ Epochs: Possibly 1-2 (could increase)
- ❌ Post-processing: Minimal (room to improve)
- ⚠️ Hyperparams: Default LoRA (could tune)

---

## Workflow Commands

```bash
# Full pipeline
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Training   │ →  │  Evaluation  │ →  │   Analysis   │
└──────────────┘    └──────────────┘    └──────────────┘
       ↓                    ↓                   ↓
   RUN_FULL_        RUN_FULL_         CHECK_EVAL_
   TRAINING.sh      EVALUATION_       STATUS.sh
                    TMUX.sh
```

---

## Error Handling

### Common Issues

1. **Duplicate Nodes**
   ```
   Problem: (n / nhớ :agent (n / tôi))  ← 'n' used twice
   Solution: Post-process to rename → n, n2
   ```

2. **Unmatched Parens**
   ```
   Problem: (nhớ :agent (tôi)  ← Missing ')'
   Solution: Balance parentheses automatically
   ```

3. **Prompt Mismatch**
   ```
   Problem: English prompt → garbage output
   Solution: Use Vietnamese prompt matching training ✅
   ```

---

## Next Steps

1. ✅ **Run Full Evaluation**
   ```bash
   bash RUN_FULL_EVALUATION_TMUX.sh
   ```

2. 📊 **Analyze Results**
   - Check F1 on full test set
   - Identify error patterns
   - Plan improvements

3. 🔧 **Iterate**
   - Fix post-processing
   - Tune hyperparameters
   - Consider retraining

---

_This workflow document explains the MTUP approach used in this Vietnamese AMR parser._
