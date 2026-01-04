# 🔍 Debug: Prediction Process Stopped

## Vấn đề / Problem

Prediction process đã dừng lại khi đang chạy trên server.

## Các bước kiểm tra / Debugging Steps

### 1. Kiểm tra process có còn chạy không

```bash
ssh islabworker2@islab-server2

cd /mnt/nghiepth/giangha/visempar/ViSemPar_new1

# Check if still running
ps aux | grep predict_mtup_fixed.py
```

**Nếu còn chạy:**
- Process vẫn đang hoạt động, chỉ là chậm
- Đợi thêm (inference takes ~1 hour for 150 sentences)

**Nếu không chạy nữa:**
- Process đã crash hoặc kết thúc
- Tiếp tục bước 2

---

### 2. Kiểm tra file output có được tạo không

```bash
ls -lh evaluation_results/mtup_predictions_FIXED.txt

# Count predictions
wc -l evaluation_results/mtup_predictions_FIXED.txt

# View last few predictions
tail -50 evaluation_results/mtup_predictions_FIXED.txt
```

**Nếu file tồn tại:**
- Check xem có bao nhiêu predictions đã được tạo
- Expected: 150 AMRs (separated by blank lines)

**Nếu file không tồn tại hoặc trống:**
- Prediction crashed trước khi save
- Tiếp tục bước 3

---

### 3. Kiểm tra lỗi trong stdout/stderr

Nếu bạn chạy command trong screen/tmux:

```bash
# If using screen
screen -r  # Resume screen session

# If using tmux
tmux attach  # Resume tmux session
```

Nếu chạy trực tiếp, check terminal output để tìm error message.

**Common errors:**

**A. CUDA OOM (Out of Memory)**
```
RuntimeError: CUDA out of memory
```

**Giải pháp:**
```bash
# Use smaller batch or reduce max_new_tokens
python3 predict_mtup_fixed.py \
    --model outputs/mtup_fixed_20260104_082506/checkpoint-148 \
    --test-file data/public_test.txt \
    --output evaluation_results/mtup_predictions_FIXED.txt \
    --batch-size 1 \
    --verbose
```

**B. Hung on specific sentence**

Model generation stuck on một câu cụ thể.

**Giải pháp:** Add timeout
```python
# In predict_mtup_fixed.py, add timeout to generate():
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        num_beams=1,
        do_sample=False,
        timeout=60.0  # 60 second timeout
    )
```

**C. Extraction error**

Warning về invalid AMR và process crash.

**Giải pháp:** Predictions vẫn nên được save ngay cả khi invalid. Check code.

---

### 4. Kiểm tra sentence nào gây lỗi

```bash
# Check how many sentences processed
grep -c "Processing sentence" <logfile_if_exists>

# Or check output file
python3 << 'EOF'
with open('evaluation_results/mtup_predictions_FIXED.txt', 'r') as f:
    content = f.read()
    preds = content.strip().split('\n\n')
    print(f"Predictions generated: {len(preds)}")
    print(f"Expected: 150")

with open('data/public_test.txt', 'r') as f:
    sentences = [line.strip() for line in f if line.strip()]
    print(f"Total sentences: {len(sentences)}")

print(f"\nStopped at sentence: {len(preds) + 1}")
EOF
```

---

## Giải pháp / Solutions

### Solution 1: Restart với verbose logging

```bash
cd /mnt/nghiepth/giangha/visempar/ViSemPar_new1

python3 predict_mtup_fixed.py \
    --model outputs/mtup_fixed_20260104_082506/checkpoint-148 \
    --test-file data/public_test.txt \
    --output evaluation_results/mtup_predictions_FIXED.txt \
    --verbose 2>&1 | tee prediction.log
```

Điều này sẽ:
- Print mọi bước processing
- Save log to `prediction.log`
- Dễ debug nếu crash lại

---

### Solution 2: Test với 10 sentences trước

```bash
# Create small test file
head -10 data/public_test.txt > data/test_small.txt

# Test prediction
python3 predict_mtup_fixed.py \
    --model outputs/mtup_fixed_20260104_082506/checkpoint-148 \
    --test-file data/test_small.txt \
    --output evaluation_results/mtup_test_small.txt \
    --verbose
```

Nếu 10 sentences work → Problem là timeout/memory với full dataset

Nếu 10 sentences cũng crash → Problem là model/code logic

---

### Solution 3: Resume từ checkpoint (nếu có partial output)

```python
# Add to predict_mtup_fixed.py
# Check if output file exists and skip processed sentences

if os.path.exists(args.output):
    with open(args.output, 'r') as f:
        existing_preds = f.read().strip().split('\n\n')
        skip_count = len(existing_preds)
        print(f"Found {skip_count} existing predictions, resuming...")
else:
    skip_count = 0

# Then in loop:
for i, sentence in enumerate(sentences):
    if i < skip_count:
        continue  # Skip already processed
    # ... rest of prediction
```

---

### Solution 4: Generate in smaller batches với explicit save

```python
# Modify to save after every N predictions
SAVE_INTERVAL = 10

predictions = []
for i, sentence in enumerate(sentences):
    pred = self.predict(sentence)
    predictions.append(pred)

    # Save checkpoint every 10 predictions
    if (i + 1) % SAVE_INTERVAL == 0:
        with open(args.output, 'w') as f:
            f.write('\n\n'.join(predictions))
        print(f"Saved checkpoint at {i+1} predictions")

# Final save
with open(args.output, 'w') as f:
    f.write('\n\n'.join(predictions))
```

---

## Quick Fix - Run ngay bây giờ

**Option A: Nếu bạn muốn chạy lại ngay (recommended)**

```bash
ssh islabworker2@islab-server2
cd /mnt/nghiepth/giangha/visempar/ViSemPar_new1

# Test small first (5 minutes)
head -10 data/public_test.txt > data/test_small.txt

python3 predict_mtup_fixed.py \
    --model outputs/mtup_fixed_20260104_082506/checkpoint-148 \
    --test-file data/test_small.txt \
    --output evaluation_results/mtup_test_small.txt \
    --verbose

# If successful, run full (1 hour)
python3 predict_mtup_fixed.py \
    --model outputs/mtup_fixed_20260104_082506/checkpoint-148 \
    --test-file data/public_test.txt \
    --output evaluation_results/mtup_predictions_FIXED.txt \
    --verbose 2>&1 | tee prediction.log
```

**Option B: Nếu bạn muốn tôi debug code trước**

Tell me:
1. Error message cuối cùng bạn thấy (nếu có)
2. File `evaluation_results/mtup_predictions_FIXED.txt` có tồn tại không?
3. Nếu có, có bao nhiêu predictions? (`wc -l evaluation_results/mtup_predictions_FIXED.txt`)

---

## Expected timeline

- **Small test (10 sentences):** ~2-3 minutes
- **Full test (150 sentences):** ~30-60 minutes (2-stage generation is slower)
- **SMATCH calculation:** ~5 minutes

---

## What to expect when successful

```
Processing sentence 1/150: Tôi nhớ lời...
  Stage 1: (nhớ :pivot (tôi) :theme (lời...))
  Stage 2: (n / nhớ :pivot (t / tôi) :theme (l / lời...))
  ✓ Valid AMR

Processing sentence 2/150: ...
...

Processing sentence 150/150: ...
  ✓ Valid AMR

================================================================================
PREDICTION COMPLETE
================================================================================

Total predictions: 150
Valid AMRs: 137 (91.3%)
Invalid AMRs: 13 (8.7%)

Saved to: evaluation_results/mtup_predictions_FIXED.txt
```

Sau đó:

```bash
# Calculate SMATCH
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

**Expected F1:** 0.50-0.55 (hypothesis: better than Baseline's 0.47)

---

## Hãy cho tôi biết / Please tell me:

1. **Bạn có thấy error message gì không?** (Any error messages?)
2. **File output có tồn tại không?** (Does output file exist?)
3. **Bạn muốn tôi sửa code hay bạn sẽ chạy lại?** (Want me to fix code or will you rerun?)
