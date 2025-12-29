# Quick Restart Guide - Sau Khi Pull Code

## 1. Pull Code Mới
```bash
cd ~/ViSemPar_new1
git pull
```

## 2. Xóa Model Lỗi (Giải Phóng ~2-3GB)
```bash
bash CLEANUP_FAILED_MODEL.sh
```

Nhấn `y` để confirm xóa.

## 3. Kiểm Tra Template Đã Fix
```bash
# Kiểm tra không còn placeholder text
grep -n "Format.*biến.*khái_niệm" config/prompt_templates.py
grep -n "AMR cuối cùng:" config/prompt_templates.py
```

Cả 2 lệnh phải trả về **không có kết quả** (đã fix ✅)

## 4. Start Training Lại (MTUP)
```bash
# Tạo tmux session
tmux new -s mtup_train

# Trong tmux, chạy training
bash scripts/run_training_mtup.sh

# Detach khỏi tmux (training vẫn chạy background)
# Nhấn: Ctrl+B rồi D
```

## 5. Kiểm Tra Training Progress
```bash
# Attach lại vào tmux session
tmux attach -t mtup_train

# Hoặc check log file
tail -f models/checkpoints/mtup_reentrancy/training.log
```

## 6. Sau Khi Training Xong (~9 giờ)
```bash
# Chạy evaluation
python scripts/evaluate_model.py \
  --model_path models/mtup_reentrancy_final \
  --test_file data/processed/vlsp_amr_v2_reentrancy_test.json \
  --output_file results/evaluation/mtup_reentrancy_eval.json

# Xem kết quả F1
grep "f1" results/evaluation/mtup_reentrancy_eval.json
```

## 7. Nếu Evaluation Thành Công → Train Baseline
```bash
tmux new -s baseline_train
bash scripts/run_training_baseline.sh
# Ctrl+B, D để detach
```

## Quick Tmux Commands

| Command | Action |
|---------|--------|
| `tmux new -s name` | Tạo session mới |
| `Ctrl+B, D` | Detach (training vẫn chạy) |
| `tmux attach -t name` | Attach lại vào session |
| `tmux ls` | List tất cả sessions |
| `tmux kill-session -t name` | Kill session (nếu cần) |

## Files Changed in This Fix

1. **config/prompt_templates.py**
   - Line 45-51: Removed placeholder text từ V2_NATURAL
   - Line 117-125: Removed placeholder text từ V5_COT

2. **CLEANUP_FAILED_MODEL.sh** (NEW)
   - Script xóa model lỗi

3. **TEMPLATE_LEAKAGE_FIX.md** (NEW)
   - Chi tiết về lỗi và cách fix

4. **DEMO_STRATEGY.md** (NEW)
   - So sánh News Analysis vs Book Search

5. **demo_examples.json** (NEW)
   - Test cases từ VLSP corpus

6. **system_architecture_clean.drawio** (NEW)
   - Diagram cho thesis (black/white, simple)

## What Was Fixed

**Before**: Model output
```
(biến / khái_niệm :quan_hệ ...) AMR cuối cùng: (n / nhớ ...)
❌ Parsing error on 100% of test cases
```

**After**: Model should output
```
(n / nhớ :pivot(t / tôi) :theme(l / lời ...))
✅ Clean AMR without template text
```

## Expected Results After Re-training

- **F1 score**: 0.47-0.50 (same as before template fix was attempted)
- **Parsing success**: 100% (no template leakage)
- **Evaluation errors**: ~47/150 (semantic errors, not parsing errors)

## Troubleshooting

### If training fails with "out of memory"
```bash
# Giảm batch size trong config/config_mtup.py
per_device_train_batch_size: 4 → 2
```

### If evaluation still shows template text
```bash
# Verify template file
cat config/prompt_templates.py | grep -A 10 "BƯỚC 2"
# Không được thấy "Format:" hoặc "AMR cuối cùng:"
```

### If git pull conflicts
```bash
git stash
git pull
git stash pop
```

## Time Estimates

- Pull code: 10 giây
- Cleanup model: 30 giây
- Training MTUP: **~9 giờ**
- Evaluation: 5 phút
- Training Baseline: **~6 giờ**

**Total**: ~15 giờ cho cả 2 models

## Contact/Support

Nếu có lỗi, check:
1. `TEMPLATE_LEAKAGE_FIX.md` - Chi tiết về fix
2. Training logs trong tmux session
3. `models/checkpoints/mtup_reentrancy/training.log`
