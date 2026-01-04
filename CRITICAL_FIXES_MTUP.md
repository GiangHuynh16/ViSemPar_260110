# 🚨 CRITICAL FIXES - MTUP Training Issues

## Vấn đề phát hiện (Problems Discovered)

Sau khi train và chạy prediction, phát hiện **3 vấn đề nghiêm trọng** khiến model generate output rất ngắn và thiếu:

### 1. **INDENT MISMATCH - Nghiêm trọng nhất!** ❌

**Vấn đề:** Ví dụ trong prompt có indent, nhưng training data không có

**Prompt template cũ:**
```python
VÍ DỤ:
Câu: Anh ấy đã hoàn thành công việc.
AMR chuẩn PENMAN:
(h / hoàn_thành
    :agent (a / anh)        ← CÓ INDENT 4 SPACES!
    :theme (c / công_việc)
    :aspect (đ / đá))
```

**Training data thực tế:**
```
(b / bi_kịch
:domain(c / chỗ        ← KHÔNG CÓ INDENT!
:mod(đ / đó)))
```

**Hậu quả:**
- Model bị confused giữa 2 format
- Không biết phải generate format nào
- Kết quả: output ngắn, thiếu thông tin

**Fix:** ✅ Dùng ví dụ THỰC từ training data

---

### 2. **Extract Logic - CHỈ LẤY 1 DÒNG!** ❌

**Code cũ trong `predict_mtup_fixed.py`:**
```python
# Line 92
result = result.split('\n')[0]  # ← CHỈ LẤY DÒNG ĐẦU TIÊN!
```

**Hậu quả:**
AMR nhiều dòng bị cắt ngắn:
```
Input:  (bi_kịch\n:domain(chỗ\n:mod(đó)))
Output: (bi_kịch        ← CHỈ CÓ THẾ NÀY!
```

**Fix:** ✅ Xóa dòng đó, return full AMR

---

### 3. **Ví dụ không match với data** ❌

**Ví dụ cũ:** "Anh ấy đã hoàn thành công việc" (không có trong training data)

**Data thực:** "Bi kịch là ở chỗ đó !" (example #1 trong training data)

**Hậu quả:**
- Model chưa từng thấy pattern từ ví dụ
- Ví dụ không representative

**Fix:** ✅ Dùng ví dụ thực từ data

---

## Các fix đã apply

### Fix 1: Update Prompt Template

File: `config/prompt_templates_fixed.py`

**Before:**
```python
MTUP_ULTRA_MINIMAL = """...
VÍ DỤ:
Câu: Anh ấy đã hoàn thành công việc.
AMR không biến: (hoàn_thành :agent (anh) :theme (công_việc) :aspect (đá))
AMR chuẩn PENMAN:
(h / hoàn_thành
    :agent (a / anh)      ← INDENT!
    :theme (c / công_việc)
    :aspect (đ / đá))
```

**After:**
```python
MTUP_ULTRA_MINIMAL = """Chuyển câu tiếng Việt sau sang AMR theo chuẩn PENMAN.

VÍ DỤ:
Câu: Bi kịch là ở chỗ đó !
AMR không biến: (bi_kịch :domain (chỗ :mod (đó)))
AMR chuẩn PENMAN:
(b / bi_kịch
:domain(c / chỗ    ← KHÔNG INDENT, MATCH DATA!
:mod(đ / đó)))

---

Câu: {sentence}

AMR không biến:
{amr_no_vars}

AMR chuẩn PENMAN:
{amr_with_vars}"""
```

**Changes:**
- ✅ Dùng ví dụ thực từ training data (example #1)
- ✅ Xóa indent để match với data format
- ✅ Format chính xác: `:domain(c` không có space

---

### Fix 2: Update Extraction Logic

File: `predict_mtup_fixed.py`

**Before:**
```python
result = '\n'.join(amr_lines).strip()

# Clean up
result = result.split('\n')[0] if result else ""  # ← CHỈ LẤY 1 DÒNG!
return result.strip()
```

**After:**
```python
result = '\n'.join(amr_lines).strip()

# Return full AMR (can be multi-line)
return result
```

---

### Fix 3: Update All Templates Consistently

**MTUP_INFERENCE_TEMPLATE:** ✅ Updated
**MTUP_INFERENCE_STEP2_TEMPLATE:** ✅ Updated
**MTUP_ULTRA_MINIMAL:** ✅ Updated

Tất cả đều dùng cùng 1 ví dụ thực từ data.

---

## Tại sao phải train lại?

### Model đã train SAI format!

**Training với prompt cũ:**
```
Model thấy ví dụ:    (h / hoàn_thành
                          :agent (a / anh)    ← 4 spaces indent

Model phải generate: (b / bi_kịch
                     :domain(c / chỗ         ← NO indent
```

→ **MISMATCH!** Model confused, không học đúng format

### Model sau khi train lại:

**Training với prompt mới:**
```
Model thấy ví dụ:    (b / bi_kịch
                     :domain(c / chỗ          ← NO indent

Model phải generate: (b / bi_kịch
                     :domain(c / chỗ          ← NO indent
```

→ **CONSISTENT!** Model học đúng format

---

## Timeline

### Lần train cũ (WRONG):
- Preprocessed: 1,262 examples ✅
- Training config: 2 epochs, bfloat16 ✅
- **Prompt template: WRONG FORMAT** ❌
- **Result: F1=0.11, output ngắn** ❌

### Lần train mới (FIXED):
- Preprocessed: 1,262 examples (giữ nguyên) ✅
- Training config: 2 epochs, bfloat16 (giữ nguyên) ✅
- **Prompt template: FIXED FORMAT** ✅
- **Expected: F1 > 0.47, output đầy đủ** ✅

---

## Bước tiếp theo

### 1. Pull code mới (trên server)

```bash
ssh islabworker2@islab-server2
cd /mnt/nghiepth/giangha/visempar/ViSemPar_new1
git pull origin main
```

### 2. Xóa model cũ (để tránh nhầm lẫn)

```bash
rm -rf outputs/mtup_fixed_20260104_082506
```

### 3. Train lại (~4 giờ)

```bash
bash TRAIN_MTUP_FIXED.sh
```

**Kết quả mong đợi:**
- Training: 148 steps (2 epochs)
- Checkpoints: 100, 148
- Output folder: `outputs/mtup_fixed_YYYYMMDD_HHMMSS/`

### 4. Predict với checkpoint tốt nhất

```bash
# Test small first
bash TEST_PREDICTION_SMALL.sh

# If OK, run full
bash RESUME_PREDICTION.sh
```

### 5. Calculate SMATCH

```bash
python3 filter_valid_amrs.py \
    --predictions evaluation_results/mtup_predictions_FIXED.txt \
    --ground-truth data/public_test_ground_truth.txt \
    --output-pred evaluation_results/mtup_valid.txt \
    --output-gold evaluation_results/gold_valid.txt

python -m smatch -f \
    evaluation_results/mtup_valid.txt \
    evaluation_results/gold_valid.txt \
    --significant 4
```

---

## Expected Results (sau khi train lại)

### Best Case (Hypothesis)

| Metric | Baseline | MTUP (Old) | MTUP (Fixed) |
|--------|----------|------------|--------------|
| F1 | 0.47 | 0.11 ❌ | **0.50-0.52** ✅ |
| Validity | 91.3% | <50% ❌ | **>90%** ✅ |
| Output | Full | Ngắn ❌ | **Full** ✅ |

### Worst Case (Still OK)

| Metric | Value |
|--------|-------|
| F1 | 0.45-0.47 (same as Baseline) |
| Validity | >85% |
| Output | Full AMR, không còn ngắn |

→ Vẫn là improvement lớn so với lần train cũ (F1=0.11)

---

## Tổng kết

### Vấn đề gốc rễ:

**INDENT MISMATCH** giữa ví dụ trong prompt và training data actual format

### Tại sao lại nghiêm trọng:

1. Model học từ VÍ DỤ trong prompt
2. VÍ DỤ có indent → Model nghĩ phải generate indent
3. Nhưng TARGET (training data) không có indent → Contradiction!
4. Model confused → Generate ngắn để avoid risk

### Lesson learned:

**CRITICAL:** Ví dụ trong prompt PHẢI match 100% với training data format!

Không chỉ về content, mà cả:
- Spacing
- Indentation
- Line breaks
- Punctuation

---

## Files Changed

1. ✅ `config/prompt_templates_fixed.py` - Fixed all templates
2. ✅ `predict_mtup_fixed.py` - Fixed extraction logic
3. ✅ `CRITICAL_FIXES_MTUP.md` - This document

---

## Checklist để train lại

- [ ] Pull code mới: `git pull origin main`
- [ ] Verify templates: `python3 config/prompt_templates_fixed.py`
- [ ] Xóa model cũ: `rm -rf outputs/mtup_fixed_20260104_*`
- [ ] Train mới: `bash TRAIN_MTUP_FIXED.sh`
- [ ] Monitor: `tail -f logs/training_mtup_fixed_*.log`
- [ ] Test prediction: `bash TEST_PREDICTION_SMALL.sh`
- [ ] Full prediction: `bash RESUME_PREDICTION.sh`
- [ ] Calculate SMATCH
- [ ] Compare với Baseline

---

**Ready to retrain! Lần này sẽ đúng! 🚀**
