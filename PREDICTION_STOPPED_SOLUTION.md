# ✅ Giải pháp: Prediction Process Stopped

## 🔍 Vấn đề (Problem)

Prediction process đã dừng lại khi chạy trên server. Có thể do:
1. Model generation bị treo (hung) ở một câu cụ thể
2. OOM (Out of Memory)
3. Không có checkpoint nên mất hết progress khi crash

## ✅ Đã sửa (Fixed)

### 1. **Thêm Checkpoint Saving** (Quan trọng nhất!)

File `predict_mtup_fixed.py` giờ tự động lưu checkpoint mỗi 10 predictions:

```python
# Save checkpoint every 10 predictions
if checkpoint_file and (i % save_interval == 0):
    logger.info(f"💾 Saving checkpoint at {i}/{len(sentences)}...")
    with open(checkpoint_file, 'w') as f:
        f.write('\n\n'.join(results))
```

**Lợi ích:**
- Nếu process dừng ở sentence 47, bạn vẫn có 40 predictions đã lưu
- Không mất công việc đã làm
- Có thể monitor progress

### 2. **Thêm Error Handling**

Mỗi sentence giờ được wrap trong try-catch:

```python
try:
    result = self.predict(sentence, verbose=verbose)
    results.append(result)
except Exception as e:
    logger.error(f"❌ Error processing sentence {i}: {e}")
    results.append("(error / processing)")  # Placeholder
    # Still save checkpoint even on error
```

**Lợi ích:**
- Process không crash khi gặp 1 câu lỗi
- Tiếp tục với câu tiếp theo
- Vẫn lưu checkpoint

### 3. **Progress Logging**

Show progress mỗi 10 sentences:

```python
if verbose or (i % 10 == 0):
    logger.info(f"Processing {i}/{len(sentences)}: {sentence[:50]}...")
```

**Lợi ích:**
- Biết được đang ở đâu
- Estimate thời gian còn lại
- Dễ debug nếu bị treo

---

## 🚀 Cách sử dụng (How to Use)

### Option 1: Test với 10 sentences trước (Recommended)

```bash
# On server
ssh islabworker2@islab-server2
cd /mnt/nghiepth/giangha/visempar/ViSemPar_new1

# Pull latest code
git pull origin main

# Run quick test (2-3 phút)
bash TEST_PREDICTION_SMALL.sh
```

**Kết quả mong đợi:**
```
✅ TEST COMPLETE
📊 Results:
  Predictions: 10 / 10

✅ All 10 predictions generated successfully!

🎯 Next step: Run full prediction
   bash RESUME_PREDICTION.sh
```

### Option 2: Chạy full 150 sentences (Nếu test OK)

```bash
bash RESUME_PREDICTION.sh
```

**Timeline:**
- Start: Loading model (~2 min)
- Progress: Every 10 sentences shows update
- Saves: Checkpoint every 10 predictions
- Total time: ~30-60 minutes

**Example output:**
```
Processing 10/150: Tôi nhớ lời anh chủ tịch...
💾 Saving checkpoint at 10/150...
✅ Checkpoint saved (10 predictions)

Processing 20/150: ...
💾 Saving checkpoint at 20/150...
✅ Checkpoint saved (20 predictions)

...

Processing 150/150: ...
💾 Saving checkpoint at 150/150...
✅ Checkpoint saved (150 predictions)

✅ PREDICTION COMPLETE

📊 Results:
  Predictions generated: ~150
  Expected: 150
  Output: evaluation_results/mtup_predictions_FIXED.txt

✅ All predictions complete!

🎯 Next steps:
   Calculate SMATCH...
```

---

## 📊 Sau khi prediction xong (After Completion)

### 1. Validate AMRs

```bash
python3 filter_valid_amrs.py \
    --predictions evaluation_results/mtup_predictions_FIXED.txt \
    --ground-truth data/public_test_ground_truth.txt \
    --output-pred evaluation_results/mtup_valid.txt \
    --output-gold evaluation_results/gold_valid.txt
```

**Expected:**
```
Filtering predictions...
  Valid AMRs: 137/150 (91.3%)  # Hypothesis: similar to Baseline
  Invalid AMRs: 13/150 (8.7%)

Saved:
  - evaluation_results/mtup_valid.txt (predictions)
  - evaluation_results/gold_valid.txt (ground truth)
```

### 2. Calculate SMATCH

```bash
python -m smatch -f \
    evaluation_results/mtup_valid.txt \
    evaluation_results/gold_valid.txt \
    --significant 4
```

**Expected:**
```
F-score: 0.51
Precision: 0.53
Recall: 0.49
```

**Hypothesis:** MTUP should score **>0.50** (better than Baseline's 0.47)

### 3. So sánh với Baseline (Comparison)

| Metric | Baseline | MTUP (Expected) | Winner |
|--------|----------|-----------------|--------|
| **F1** | 0.47 | 0.50-0.52 | MTUP? |
| **Validity** | 91.3% | ~91% | Similar |
| **Training Time** | 2.5h | 4h | Baseline |
| **Inference Speed** | 5/sec | 2.5/sec | Baseline |
| **Method** | Direct | Two-stage | - |

---

## 🐛 Nếu vẫn gặp vấn đề (If Still Having Issues)

### Issue 1: Process vẫn bị treo (Still hanging)

**Kiểm tra:**
```bash
# Check if process is stuck
ps aux | grep predict_mtup_fixed.py

# Check GPU memory
nvidia-smi

# Check which sentence it stopped at
tail -50 prediction_*.log
```

**Giải pháp:**
```bash
# Kill process
pkill -f predict_mtup_fixed.py

# Check checkpoint
ls -lh evaluation_results/mtup_predictions_FIXED.txt
grep -c '^(' evaluation_results/mtup_predictions_FIXED.txt

# Resume (sẽ bắt đầu lại từ đầu nhưng có checkpoint)
bash RESUME_PREDICTION.sh
```

### Issue 2: OOM (Out of Memory)

**Triệu chứng:**
```
RuntimeError: CUDA out of memory
```

**Giải pháp:** Reduce max_new_tokens

Edit `config/config_mtup_fixed.py`:
```python
INFERENCE_CONFIG = {
    'max_new_tokens': 256,  # Reduce from 512
    # ...
}
```

Then rerun.

### Issue 3: Invalid AMRs quá nhiều (>20%)

**Kiểm tra checkpoint khác:**

```bash
# Evaluate other checkpoints
bash EVALUATE_MTUP_CHECKPOINTS.sh outputs/mtup_fixed_20260104_082506

# Try best checkpoint (usually not the last one)
python3 predict_mtup_fixed.py \
    --model outputs/mtup_fixed_20260104_082506/checkpoint-100 \
    --test-file data/public_test.txt \
    --output evaluation_results/mtup_predictions_ckpt100.txt \
    --verbose
```

---

## 📁 Files đã tạo (Created Files)

1. **RESUME_PREDICTION.sh** - Script chính để chạy prediction
2. **TEST_PREDICTION_SMALL.sh** - Test nhanh 10 sentences
3. **DEBUG_PREDICTION_STOPPED.md** - Hướng dẫn debug chi tiết
4. **test_single_prediction.py** - Test 1 câu (for debugging)

---

## ✅ Checklist để hoàn thành (To Complete)

- [ ] Pull latest code: `git pull origin main`
- [ ] Test 10 sentences: `bash TEST_PREDICTION_SMALL.sh` (2-3 min)
- [ ] ✅ Test success → Run full: `bash RESUME_PREDICTION.sh` (30-60 min)
- [ ] Filter valid AMRs: `python3 filter_valid_amrs.py ...` (5 min)
- [ ] Calculate SMATCH: `python -m smatch -f ...` (5 min)
- [ ] Compare with Baseline (F1=0.47)
- [ ] Document results in thesis

---

## 🎯 Expected Final Results

### Best Case (Hypothesis confirmed)

```
MTUP Results:
  F1: 0.51 (vs Baseline 0.47) → +8.5% improvement ✅
  Validity: 91.3% (same as Baseline)
  Conclusion: Two-stage decomposition helps!
```

### Neutral Case

```
MTUP Results:
  F1: 0.47 (same as Baseline)
  Validity: 91.3%
  Conclusion: No significant difference, but good validation
```

### Worst Case (Unlikely)

```
MTUP Results:
  F1: 0.43 (worse than Baseline)
  Validity: <85%
  Problem: Likely wrong checkpoint or inference prompt mismatch
  Solution: Try different checkpoint
```

---

## 📝 Summary cho User

### Tiếng Việt:

Tôi đã fix vấn đề prediction bị dừng:

**Các cải tiến:**
1. ✅ **Auto-save checkpoint** mỗi 10 predictions → Không mất dữ liệu nếu crash
2. ✅ **Error handling** cho từng câu → Process không crash khi gặp lỗi
3. ✅ **Progress logging** mỗi 10 câu → Biết được đang ở đâu
4. ✅ **2 scripts mới:**
   - `TEST_PREDICTION_SMALL.sh` - Test 10 sentences (2-3 phút)
   - `RESUME_PREDICTION.sh` - Chạy full 150 sentences (30-60 phút)

**Bước tiếp theo (trên server):**
```bash
git pull origin main
bash TEST_PREDICTION_SMALL.sh    # Test trước
bash RESUME_PREDICTION.sh        # Nếu test OK, chạy full
```

**Kết quả mong đợi:**
- F1 > 0.50 (better than Baseline's 0.47)
- Validity ~91%
- Hoàn thành trong 30-60 phút

**All code đã push lên GitHub!**

---

## 📧 Files để đọc

- **Quick start:** [DEBUG_PREDICTION_STOPPED.md](DEBUG_PREDICTION_STOPPED.md)
- **Technical details:** [MTUP_IMPLEMENTATION_COMPLETE.md](MTUP_IMPLEMENTATION_COMPLETE.md)
- **Comparison:** [BASELINE_VS_MTUP_COMPARISON.md](BASELINE_VS_MTUP_COMPARISON.md)

---

**Ready to go! 🚀**

Hãy pull code và chạy test nhé!
