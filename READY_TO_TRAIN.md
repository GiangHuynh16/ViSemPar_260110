# Baseline 7B - Ready to Train! 🚀

**Date**: 2025-12-30
**Status**: ✅ All changes completed, ready to pull and train

---

## ✅ Hoàn thành

### 1. Vietnamese Prompt Template

**File**: `config/config.py` lines 117-130

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

**Ưu điểm**:
- ✅ Vietnamese (match với MTUP)
- ✅ Explicit rules (5 quy tắc rõ ràng)
- ✅ Concrete examples (chủ_tịch, c / chủ_tịch)
- ✅ Error prevention (parentheses, variables)

### 2. Minimal Postprocessing

**File**: `evaluate_baseline_model.py` lines 22-40

```python
def post_process_amr_conservative(amr_string: str) -> str:
    """
    Minimal post-processing - extract AMR only, NO heavy processing
    Philosophy: Let LLM output speak for itself, trust the model
    """
    if not amr_string or len(amr_string) < 3:
        return "(amr-empty)"

    amr = amr_string.strip()

    # Simply find first '(' and take everything from there
    if '(' in amr:
        amr = amr[amr.index('('):]

    # Basic whitespace normalization only
    amr = re.sub(r'\s+', ' ', amr).strip()

    return amr
```

**Philosophy**:
- ✅ NO parentheses balancing (trust LLM)
- ✅ NO marker removal (simple extraction)
- ✅ NO structural fixes (LLM should get it right)
- ✅ ONLY basic whitespace cleaning

### 3. Minimal Preprocessing

**File**: `config/config.py` lines 101-111

```python
PREPROCESSING_CONFIG = {
    "preserve_coreference": True,       # Keep coreference
    "normalize_concepts": False,        # Let LLM learn
    "handle_multiword": False,          # Let LLM learn underscores
    "fix_malformed_amr": True,          # Only fix broken data
    "remove_variables": False,          # Keep variables
    "clean_whitespace": True,           # Basic cleaning
    "validate_structure": True,         # Validate parentheses
}
```

---

## 📊 Template So Sánh

### Baseline Template (Vietnamese)

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

### MTUP Template (Vietnamese, 2-stage)

```
### NHIỆM VỤ: Chuyển đổi câu tiếng Việt sang AMR (2 bước)

### Câu cần phân tích:
{sentence}

### Kết quả phân tích:

## Bước 1 - Tạo cấu trúc AMR (chưa có biến):
[...]

## Bước 2 - Gán biến và hoàn thiện:
AMR hoàn chỉnh:
[...]
```

### So sánh

| Aspect | Baseline | MTUP |
|--------|----------|------|
| **Language** | Vietnamese ✅ | Vietnamese ✅ |
| **Stages** | 1 (direct) | 2 (decomposed) |
| **Rules** | 5 explicit rules | 2-stage guidance |
| **Examples** | Inline (chủ_tịch, c / chủ_tịch) | Structured |
| **Postprocessing** | Minimal (find '(') | Conservative |

**Fair comparison**: Cả hai đều Vietnamese, chỉ khác ở task decomposition!

---

## 🔧 Hướng dẫn Pull và Train

### Bước 1: SSH vào server

```bash
ssh your-server
cd ViSemPar_new1
```

### Bước 2: Pull code mới

```bash
# Stash local changes (if any)
git stash

# Pull latest
git pull origin main

# Apply stashed changes (if needed)
git stash pop
```

### Bước 3: Verify changes

```bash
# Check prompt template
head -n 20 config/config.py | grep -A 15 "PROMPT_TEMPLATE"
```

**Expected output** (Vietnamese template):
```
PROMPT_TEMPLATE = """Bạn là chuyên gia phân tích ngữ nghĩa tiếng Việt...
```

✅ **Nếu thấy "Bạn là chuyên gia"** → Pull thành công!
❌ **Nếu thấy "Below is an instruction"** → Pull lại hoặc check branch

### Bước 4: Verify postprocessing

```bash
grep -A 10 "def post_process_amr" evaluate_baseline_model.py
```

**Expected**: Should see "Minimal post-processing" comment and simple logic (no balancing)

### Bước 5: Start training

```bash
# Create tmux session
tmux new -s baseline_7b

# Run training
bash START_BASELINE_7B_TRAINING.sh

# Detach: Ctrl+B, then D
```

### Bước 6: Monitor

```bash
# Reattach
tmux attach -t baseline_7b

# Check logs
tail -f logs/training.log

# GPU
watch -n 1 nvidia-smi
```

---

## 📝 Quick Command Summary

```bash
# Pull code
cd ViSemPar_new1 && git pull origin main

# Verify template
grep "Bạn là chuyên gia" config/config.py

# Start training
tmux new -s baseline_7b
bash START_BASELINE_7B_TRAINING.sh
# Ctrl+B, D to detach

# Monitor
tmux attach -t baseline_7b
tail -f logs/training.log
```

---

## 🎯 Expected Results

### Training
- **Model**: Qwen 2.5 7B
- **LoRA rank**: 128
- **Epochs**: 15
- **Time**: ~12-15 hours
- **Trainable params**: ~239M (same as MTUP)

### Performance Hypothesis

| Model | Template | Expected F1 | Notes |
|-------|----------|-------------|-------|
| MTUP 7B | Vietnamese + 2-stage | 0.51-0.52 | ✅ Completed |
| Baseline 7B | Vietnamese + 1-stage | 0.47-0.50 | ⏳ To train |

**Gap**: 0.02-0.04 F1 (MTUP advantage from task decomposition)

**Why smaller gap?**
- Both use Vietnamese (language consistency)
- Both have explicit rules (AMR guidance)
- Only difference: Task decomposition (1-stage vs 2-stage)

---

## ✅ Checklist Before Training

- [ ] Code pulled (`git pull origin main`)
- [ ] Prompt is Vietnamese (`grep "Bạn là chuyên gia" config/config.py`)
- [ ] Postprocessing is minimal (check `evaluate_baseline_model.py`)
- [ ] Data files exist (`ls data/train_amr_*.txt`)
- [ ] VRAM >= 18GB (`nvidia-smi`)
- [ ] In tmux session (`echo $TMUX` not empty)

---

## 🚀 Ready!

**Tất cả đã sẵn sàng!** Chỉ cần:

1. SSH vào server
2. `git pull origin main`
3. Verify template là Vietnamese
4. `bash START_BASELINE_7B_TRAINING.sh`

**Good luck!** 🎯
