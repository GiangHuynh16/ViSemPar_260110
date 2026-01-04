# 🚨 URGENT: MTUP Prediction Alignment Issues

## VẤN ĐỀ PHÁT HIỆN (Critical Issues Found)

Sau khi analyze kết quả F-score = 0.1040, phát hiện **3 vấn đề nghiêm trọng**:

### 1. **DATA MISALIGNMENT** ❌

**Test sentence #1:** "tôi nhớ lời anh chủ tịch xã Bùi Văn Luyến..."

**Prediction #1:** `(t1 / thấy :pivot(e / em) :tense(s / sẽ) :manner(n2 / ngưỡng mộ))`

→ Prediction về câu **"em sẽ ngưỡng mộ..."** chứ không phải câu test!

**→ Predictions KHÔNG tương ứng với test sentences!**

---

### 2. **HIGH DUPLICATION RATE** ❌

```
Total predictions: 150
Unique predictions: 73
Duplication rate: 51%
```

→ Model đang repeat cùng output cho nhiều câu khác nhau!

**Examples:**
- Pred #1: `(t1 / thấy :pivot(e / em)...` với `ngưỡng mộ` (có dấu _)
- Pred #2: `(t1 / thấy :pivot(e / em)...` với `ngưỡng_mộ` (không dấu _)
- Pred #3: `(t1 / thất_vọng :pivot(e / em)...`

---

### 3. **GROUND TRUTH FORMAT ISSUE** ❌

Ground truth có `#::snt` markers:
```
#::snt tôi nhớ lời anh chủ tịch xã...
(n / nhớ
    :pivot(t / tôi)
    ...)
```

Script `compare_predictions.py` cũ split bằng `\n\n` → Parse sai!

**→ So sánh không đúng!**

---

## 🔍 NGUYÊN NHÂN (Root Cause)

### Nguyên nhân chính: Test file format SAI

Kiểm tra test file:
```bash
head -3 data/public_test.txt
```

**Output thực tế:**
```
em sẽ ngưỡng mộ anh .
em sẽ thất vọng về anh .
em sẽ ca ngợi anh .
```

**Nhưng ground truth sentence #1:**
```
tôi nhớ lời anh chủ tịch xã Bùi Văn Luyến...
```

→ **`public_test.txt` KHÔNG PHẢI file test sentences đúng!**

---

## ✅ GIẢI PHÁP (Solutions)

### Solution 1: Extract sentences từ ground truth

Ground truth có format:
```
#::snt sentence
(amr...)

#::snt sentence2
(amr2...)
```

→ Extract sentences:

```bash
cd /mnt/nghiepth/giangha/visempar/ViSemPar_new1

# Extract sentences
grep "^#::snt" data/public_test_ground_truth.txt | sed 's/^#::snt //' > data/public_test_sentences_CORRECT.txt

# Verify
echo "Extracted $(wc -l < data/public_test_sentences_CORRECT.txt) sentences"
head -3 data/public_test_sentences_CORRECT.txt
```

**Expected output:**
```
Extracted 149 sentences
tôi nhớ lời anh chủ tịch xã Bùi Văn Luyến...
hiện nay xã có 68 tổ nhân dân...
chủ trương tốt nhưng dân không hiểu...
```

---

### Solution 2: Run prediction với file đúng

```bash
# Delete old predictions
rm -f evaluation_results/mtup_predictions_FIXED*.txt

# Run with CORRECT test file
python3 predict_mtup_fixed.py \
    --model outputs/mtup_fixed_20260104_105638/checkpoint-148 \
    --test-file data/public_test_sentences_CORRECT.txt \
    --output evaluation_results/mtup_predictions_ALIGNED.txt \
    --verbose 2>&1 | tee prediction_aligned.log
```

**Timeline:** ~30-60 minutes

---

### Solution 3: Compare với alignment đúng

```bash
python3 compare_predictions.py \
    evaluation_results/mtup_predictions_ALIGNED.txt \
    data/public_test_ground_truth.txt | head -200
```

**Expected:**
- Prediction #1 về "tôi nhớ lời..." ✅
- Prediction #2 về "hiện nay xã có..." ✅
- Structure similar (6-8 lines, 5-7 relations) ✅

---

## 🎯 AUTOMATED SCRIPT

Đã tạo script tự động:

```bash
bash FIX_PREDICTION_NOW.sh
```

Script sẽ:
1. Verify test file format
2. Extract sentences nếu cần
3. Choose model checkpoint
4. Run prediction
5. Verify results
6. Show next steps

---

## 📊 EXPECTED RESULTS (Sau khi fix)

### Before (Wrong alignment):

| Metric | Value | Issue |
|--------|-------|-------|
| F-score | 0.10 | ❌ Predictions sai câu |
| Alignment | Wrong | ❌ Pred #1 ≠ Test #1 |
| Unique preds | 73/150 (49%) | ❌ High duplication |

### After (Correct alignment):

| Metric | Expected | Why |
|--------|----------|-----|
| F-score | **0.45-0.50** | ✅ Predictions đúng câu |
| Alignment | Correct | ✅ Pred #1 = Test #1 |
| Unique preds | **140+/150 (93%+)** | ✅ Low duplication |

---

## 🔧 DEBUG CHECKLIST

Nếu sau khi chạy lại mà vẫn thấp, check:

### 1. Verify alignment

```bash
python3 << 'EOF'
# Load test sentences
with open('data/public_test_sentences_CORRECT.txt', 'r') as f:
    sentences = [line.strip() for line in f]

# Load predictions
with open('evaluation_results/mtup_predictions_ALIGNED.txt', 'r') as f:
    preds = f.read().strip().split('\n\n')

# Check first 3
for i in range(min(3, len(sentences), len(preds))):
    print(f"\n=== Example {i+1} ===")
    print(f"Sentence: {sentences[i][:80]}...")
    print(f"Prediction: {preds[i][:100]}...")
EOF
```

**Expected:** Sentence về "nhớ lời" → Prediction có `(n / nhớ` ✅

---

### 2. Check extraction quality

```bash
python3 << 'EOF'
with open('evaluation_results/mtup_predictions_ALIGNED.txt', 'r') as f:
    preds = f.read().strip().split('\n\n')

import re
for i, p in enumerate(preds[:5], 1):
    lines = len(p.split('\n'))
    rels = len(re.findall(r':[\w_\-]+', p))
    print(f"Pred {i}: {lines} lines, {rels} relations")
EOF
```

**Expected:** 6-10 lines, 5-10 relations (not 1-2 lines!)

---

### 3. Verify unique predictions

```bash
python3 << 'EOF'
with open('evaluation_results/mtup_predictions_ALIGNED.txt', 'r') as f:
    preds = f.read().strip().split('\n\n')

total = len(preds)
unique = len(set(preds))
print(f"Total: {total}")
print(f"Unique: {unique}")
print(f"Unique rate: {unique/total*100:.1f}%")

if unique < total * 0.8:
    print("\n⚠️  WARNING: Low unique rate suggests overfitting!")
else:
    print("\n✅ Good unique rate")
EOF
```

**Expected:** Unique rate > 80%

---

## 🚀 QUICK START

### Trên server, chạy ngay:

```bash
cd /mnt/nghiepth/giangha/visempar/ViSemPar_new1

# Pull code mới
git pull origin main

# Extract correct test sentences
grep "^#::snt" data/public_test_ground_truth.txt | \
    sed 's/^#::snt //' > data/public_test_sentences_CORRECT.txt

# Verify
echo "=== First 3 test sentences ==="
head -3 data/public_test_sentences_CORRECT.txt
echo ""
echo "Expected: 'tôi nhớ lời anh chủ tịch xã...'"
echo ""

# Run prediction
python3 predict_mtup_fixed.py \
    --model outputs/mtup_fixed_20260104_105638/checkpoint-148 \
    --test-file data/public_test_sentences_CORRECT.txt \
    --output evaluation_results/mtup_predictions_ALIGNED.txt \
    --verbose 2>&1 | tee prediction_aligned.log

# Compare
python3 compare_predictions.py \
    evaluation_results/mtup_predictions_ALIGNED.txt \
    data/public_test_ground_truth.txt | head -150
```

---

## 📋 VERIFICATION STEPS

Sau khi chạy xong, verify:

### Step 1: Check alignment
```bash
echo "=== Sentence 1 ==="
head -1 data/public_test_sentences_CORRECT.txt

echo -e "\n=== Prediction 1 ==="
head -10 evaluation_results/mtup_predictions_ALIGNED.txt
```

**Expected:** Both about "tôi nhớ lời..."

---

### Step 2: Check F-score

```bash
python3 filter_valid_amrs.py \
    --predictions evaluation_results/mtup_predictions_ALIGNED.txt \
    --ground-truth data/public_test_ground_truth.txt \
    --output-pred evaluation_results/mtup_valid.txt \
    --output-gold evaluation_results/gold_valid.txt

python -m smatch -f \
    evaluation_results/mtup_valid.txt \
    evaluation_results/gold_valid.txt \
    --significant 4
```

**Expected F-score:** 0.45-0.50 (close to Baseline's 0.47)

---

## 🎯 SUMMARY

### Issues:
1. ❌ Wrong test file used (predictions về "em sẽ ngưỡng mộ" thay vì "tôi nhớ lời")
2. ❌ Data misalignment (Pred #1 ≠ Test #1)
3. ❌ High duplication (51% duplicate predictions)
4. ❌ Ground truth format not handled

### Fixes:
1. ✅ Extract correct sentences from ground truth
2. ✅ Use `public_test_sentences_CORRECT.txt`
3. ✅ Update `compare_predictions.py` to handle `#::snt`
4. ✅ Re-run prediction with correct alignment

### Expected outcome:
- F-score: **0.45-0.50** (up from 0.10)
- Alignment: **100% correct**
- Unique predictions: **>90%**

---

**Hãy chạy lại prediction với file test đúng và cho tôi biết kết quả!** 🚀
