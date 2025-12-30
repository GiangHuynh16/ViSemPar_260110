# Baseline Training Improvements - Vietnamese Prompt Template

**Date**: 2025-12-30
**Changes**: Enhanced prompt template + minimal preprocessing for better LLM performance

---

## Thay đổi chính

### 1. Prompt Template - Chuyển sang tiếng Việt

**Trước đây** (English template):
```
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Convert the following Vietnamese sentence to Abstract Meaning Representation (AMR) format. Ensure proper concept alignment and preserve co-references.

### Input:
{sentence}

### Response:
```

**Bây giờ** (Vietnamese template với explicit rules):
```
Bạn là chuyên gia phân tích ngữ nghĩa tiếng Việt. Hãy chuyển đổi câu sau sang định dạng AMR (Abstract Meaning Representation).

Quy tắc quan trọng:
- Sử dụng khái niệm tiếng Việt có dấu gạch dưới (ví dụ: chủ_tịch, môi_trường)
- Gán biến cho mỗi khái niệm (ví dụ: c / chủ_tịch)
- Sử dụng quan hệ chuẩn AMR (:ARG0, :ARG1, :time, :location, etc.)
- Giữ nguyên cấu trúc cây với dấu ngoặc đơn cân bằng
- Đảm bảo tất cả biến được định nghĩa trước khi sử dụng

Câu tiếng Việt: {sentence}

AMR:
```

### 2. Lý do thay đổi

**Vấn đề với English template**:
- ❌ Language mismatch: English instruction cho Vietnamese data
- ❌ Quá generic, không có explicit AMR rules
- ❌ Không guide model về format cụ thể
- ❌ Thiếu examples về underscore concepts

**Ưu điểm của Vietnamese template**:
- ✅ **Language consistency**: Vietnamese input → Vietnamese instruction
- ✅ **Explicit rules**: 5 quy tắc rõ ràng về AMR format
- ✅ **Concrete examples**: Ví dụ về underscore (chủ_tịch), variables (c / chủ_tịch)
- ✅ **Error prevention**: Nhắc về parentheses balance, variable definition
- ✅ **Role-based prompting**: "Bạn là chuyên gia..." → Better engagement

### 3. Preprocessing Philosophy

**Thay đổi từ "Heavy preprocessing" → "Minimal preprocessing"**

**Trước**:
```python
PREPROCESSING_CONFIG = {
    "normalize_concepts": True,     # Normalize variations
    "handle_multiword": True,       # Convert to underscores
    ...
}
```

**Bây giờ**:
```python
PREPROCESSING_CONFIG = {
    "normalize_concepts": False,    # Let LLM learn variations
    "handle_multiword": False,      # Let LLM learn underscore patterns
    "clean_whitespace": True,       # Only basic cleaning
    "validate_structure": True,     # Only validate parentheses
}
```

**Lý do**:
- Modern LLMs (Qwen 2.5 7B) học tốt hơn từ raw data
- Preprocessing có thể introduce artifacts
- LLM có thể generalize patterns tốt hơn rule-based preprocessing

---

## Files đã sửa đổi

### 1. [config/config.py](config/config.py)

**Changes**:
- Line 117-130: New Vietnamese prompt template
- Line 101-111: Minimal preprocessing config

### 2. [evaluate_baseline_model.py](evaluate_baseline_model.py)

**Changes**:
- Line 22-51: Updated `post_process_amr_conservative()` with Vietnamese markers
- Line 75-91: Updated `generate_baseline_prediction()` with new prompt
- Line 109-126: Updated extraction logic for "AMR:" marker

---

## So sánh: Baseline vs MTUP Prompts

| Aspect | Baseline (Single-Task) | MTUP (Multi-Task) |
|--------|------------------------|-------------------|
| **Language** | Vietnamese ✅ | Vietnamese ✅ |
| **Explicit Rules** | ✅ 5 rules | ✅ 2-stage guidance |
| **Examples** | ✅ Inline | ✅ Structured |
| **Task Decomposition** | ❌ Single output | ✅ Two outputs |
| **Role-based** | ✅ "Chuyên gia" | ✅ Task-focused |

**Hypothesis**:
- Baseline với Vietnamese prompt sẽ tốt hơn English prompt đáng kể
- MTUP vẫn có thể tốt hơn Baseline nhờ explicit task decomposition
- Gap giữa MTUP và Baseline sẽ nhỏ hơn (do Baseline improvement)

---

## Template hiển thị đầy đủ

### Baseline Prompt Template (Full)

```python
PROMPT_TEMPLATE = """Bạn là chuyên gia phân tích ngữ nghĩa tiếng Việt. Hãy chuyển đổi câu sau sang định dạng AMR (Abstract Meaning Representation).

Quy tắc quan trọng:
- Sử dụng khái niệm tiếng Việt có dấu gạch dưới (ví dụ: chủ_tịch, môi_trường)
- Gán biến cho mỗi khái niệm (ví dụ: c / chủ_tịch)
- Sử dụng quan hệ chuẩn AMR (:ARG0, :ARG1, :time, :location, etc.)
- Giữ nguyên cấu trúc cây với dấu ngoặc đơn cân bằng
- Đảm bảo tất cả biến được định nghĩa trước khi sử dụng

Câu tiếng Việt: {sentence}

AMR:
"""
```

### Example Training Instance

**Input**:
```
Bạn là chuyên gia phân tích ngữ nghĩa tiếng Việt. Hãy chuyển đổi câu sau sang định dạng AMR (Abstract Meaning Representation).

Quy tắc quan trọng:
- Sử dụng khái niệm tiếng Việt có dấu gạch dưới (ví dụ: chủ_tịch, môi_trường)
- Gán biến cho mỗi khái niệm (ví dụ: c / chủ_tịch)
- Sử dụng quan hệ chuẩn AMR (:ARG0, :ARG1, :time, :location, etc.)
- Giữ nguyên cấu trúc cây với dấu ngoặc đơn cân bằng
- Đảm bảo tất cả biến được định nghĩa trước khi sử dụng

Câu tiếng Việt: Chủ tịch nước gặp đại sứ Mỹ tại Hà Nội

AMR:
```

**Expected Output**:
```
(g / gặp
  :ARG0 (c / chủ_tịch
          :mod (n / nước))
  :ARG1 (đ / đại_sứ
          :mod (m / mỹ))
  :location (h / hà_nội))
```

---

## Postprocessing Updates

### Updated Markers

**Thêm Vietnamese markers vào postprocessing**:
```python
markers = [
    # English markers (legacy)
    'Instruction', 'Input', 'Response', '###',
    # Vietnamese markers (new)
    'Bạn là', 'Quy tắc', 'Câu tiếng Việt', 'AMR:'
]
```

### Extraction Logic

**Priority order**:
1. Try "AMR:" marker (Vietnamese template)
2. Fallback to "### Response:" (English template - legacy)
3. Fallback to first '(' (emergency)

---

## Hướng dẫn Pull và Train trên Server

### Bước 1: Pull code mới

```bash
# SSH vào server
ssh your-server

# Vào project directory
cd ViSemPar_new1

# Pull latest changes
git pull origin main

# Hoặc nếu có uncommitted changes
git stash
git pull origin main
git stash pop
```

### Bước 2: Verify changes

```bash
# Kiểm tra prompt template mới
grep -A 10 "PROMPT_TEMPLATE" config/config.py

# Should see Vietnamese prompt starting with "Bạn là chuyên gia..."
```

**Expected output**:
```
PROMPT_TEMPLATE = """Bạn là chuyên gia phân tích ngữ nghĩa tiếng Việt...
```

### Bước 3: Start training

```bash
# Create tmux session
tmux new -s baseline_7b

# Run training
bash START_BASELINE_7B_TRAINING.sh

# Detach: Ctrl+B, then D
```

### Bước 4: Monitor progress

```bash
# Reattach to tmux
tmux attach -t baseline_7b

# Or check logs
tail -f logs/training.log

# GPU usage
watch -n 1 nvidia-smi
```

---

## Expected Performance Improvement

### Before (English Template)

**Estimated**:
- F1: 0.40-0.45 (generic English instruction)
- Common errors: Format issues, missing underscores, undefined variables

### After (Vietnamese Template)

**Expected**:
- F1: 0.45-0.50 (explicit Vietnamese guidance)
- Fewer format errors (thanks to explicit rules)
- Better underscore usage (explicit examples)
- Fewer variable errors (explicit rule about definition)

### Comparison with MTUP

| Model | Template | Expected F1 | Improvement |
|-------|----------|-------------|-------------|
| Baseline (old) | English | 0.40-0.45 | - |
| **Baseline (new)** | **Vietnamese** | **0.45-0.50** | **+0.05** |
| MTUP 7B | Vietnamese + 2-stage | 0.51-0.52 | +0.06-0.07 |

**Gap narrowed**: From 0.11 → 0.06 (42% reduction in gap!)

---

## Git Commands để Push Changes

```bash
# Stage changes
git add config/config.py
git add evaluate_baseline_model.py
git add BASELINE_IMPROVEMENTS.md

# Commit
git commit -m "Improve baseline with Vietnamese prompt template and minimal preprocessing"

# Push
git push origin main
```

---

## Verification Checklist

Trước khi train, verify:

- [ ] Prompt template trong `config/config.py` là Vietnamese
- [ ] `evaluate_baseline_model.py` có Vietnamese markers
- [ ] Preprocessing config set to minimal
- [ ] `START_BASELINE_7B_TRAINING.sh` executable
- [ ] Data files exist trong `data/` directory
- [ ] VRAM >= 18GB available

---

## Summary

**Key Improvements**:
1. ✅ Vietnamese prompt template với explicit AMR rules
2. ✅ Minimal preprocessing (let LLM learn)
3. ✅ Updated postprocessing markers
4. ✅ Better extraction logic

**Expected Impact**:
- Baseline performance: 0.40-0.45 → **0.45-0.50** (+0.05 F1)
- Gap vs MTUP: Reduced by ~42%
- Better AMR format compliance
- Fewer variable definition errors

**Ready to train**: Pull code và run `bash START_BASELINE_7B_TRAINING.sh`! 🚀
