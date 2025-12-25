# 🚀 How to Run Full Evaluation

## Quick Start (Recommended)

### Trên Server:

```bash
cd ~/ViSemPar_new1
git pull origin main

# Chạy full evaluation trong tmux (tránh disconnect)
bash RUN_FULL_EVALUATION_TMUX.sh
```

Xong! Script sẽ tự động:
- Tìm checkpoint mới nhất
- Chạy evaluation trên TẤT CẢ test samples
- Lưu kết quả vào `outputs/evaluation_results_full_TIMESTAMP.json`

---

## Monitor Progress

### Xem live progress:
```bash
tmux attach -t mtup_eval
# Press Ctrl+B then D to detach
```

### Check status nhanh:
```bash
bash CHECK_EVALUATION_STATUS.sh
```

### Xem log real-time:
```bash
tail -f outputs/evaluation_full_*.log
```

---

## Expected Timeline

Dựa trên test với 10 samples (~200 seconds):
- **10 samples**: ~3 minutes
- **50 samples**: ~17 minutes
- **200 samples**: ~67 minutes (~1 hour)
- **500 samples**: ~2.8 hours

Công thức: `samples * 20 seconds / 60 = minutes`

---

## Results

### Khi hoàn thành, bạn sẽ thấy:

```
================================================================================
EVALUATION RESULTS
================================================================================

Processed: XXX/XXX examples
Errors:    YYY

================================================================================
SMATCH SCORES
================================================================================
  Precision: 0.XXXX
  Recall:    0.XXXX
  F1:        0.XXXX
================================================================================
```

### View kết quả:

```bash
# Find latest results file
ls -t outputs/evaluation_results_full_*.json | head -1

# View formatted
cat outputs/evaluation_results_full_TIMESTAMP.json | python3 -m json.tool
```

---

## Manual Run (Without tmux)

Nếu không muốn dùng tmux:

```bash
bash RUN_FULL_EVALUATION.sh
```

**⚠️ Lưu ý**: Nếu SSH bị disconnect, quá trình sẽ dừng!

---

## Troubleshooting

### Evaluation bị stuck?

```bash
# Check nếu process còn chạy
ps aux | grep evaluate_mtup_model.py

# Check GPU usage
nvidia-smi

# Check log
tail -30 outputs/evaluation_full_*.log
```

### Stop evaluation:

```bash
# Kill tmux session
tmux kill-session -t mtup_eval

# OR kill process directly
pkill -f evaluate_mtup_model.py
```

### Restart evaluation:

```bash
# Kill existing session first
tmux kill-session -t mtup_eval

# Then restart
bash RUN_FULL_EVALUATION_TMUX.sh
```

---

## File Outputs

Sau khi chạy, bạn sẽ có:

```
outputs/
├── evaluation_results_full_TIMESTAMP.json  ← Results (precision, recall, F1)
├── evaluation_full_TIMESTAMP.log           ← Full log
```

Timestamp format: `YYYYMMDD_HHMMSS` (ví dụ: `20231225_143052`)

---

## Current Status

Quick test (10 samples) results:
- ✅ Processed: 7/10 examples (70%)
- ✅ F1 Score: **0.4933** (~49%)
- ✅ Precision: 0.4978
- ✅ Recall: 0.5002

**Next**: Run full evaluation để có F1 chính xác trên toàn bộ test set!

---

## Scripts Available

| Script | Purpose |
|--------|---------|
| `RUN_FULL_EVALUATION.sh` | Run evaluation (foreground) |
| `RUN_FULL_EVALUATION_TMUX.sh` | Run in tmux (recommended) |
| `CHECK_EVALUATION_STATUS.sh` | Check progress |
| `evaluate_mtup_model.py` | Main evaluation code |

---

## After Evaluation

Sau khi có F1 score trên full test set:

1. **Nếu F1 > 0.55**: Tốt! Model đã học tốt
2. **Nếu F1 = 0.45-0.55**: Acceptable, có thể improve
3. **Nếu F1 < 0.45**: Cần train thêm hoặc tune hyperparameters

### Next Steps sau evaluation:
- Phân tích errors (duplicate nodes, unmatched parens)
- Train với epochs/batch size lớn hơn
- Thử template khác (v5_cot)
- Compare với baseline models
