# Hướng dẫn chẩn đoán MTUP v2 training thất bại

## Bước 1: Chạy script chẩn đoán

```bash
python mtup_v2/scripts/diagnose_model.py \
    --data_path data/train_mtup_unified.txt \
    --adapter_path outputs/mtup_260110/mtup_v2/final_adapter
```

Script này sẽ kiểm tra:
- ✅ Data có bị corrupt không
- ✅ Training loss có giảm không
- ✅ Model có học được gì không

## Bước 2: Kiểm tra training loss

Nếu có file `trainer_state.json`, xem loss progression:

```bash
# Xem loss đầu và cuối
python -c "
import json
with open('outputs/mtup_260110/mtup_v2/trainer_state.json', 'r') as f:
    state = json.load(f)
    logs = [log for log in state['log_history'] if 'loss' in log]
    print(f'First loss: {logs[0][\"loss\"]:.4f}')
    print(f'Last loss: {logs[-1][\"loss\"]:.4f}')
    print(f'Reduction: {((logs[0][\"loss\"] - logs[-1][\"loss\"]) / logs[0][\"loss\"] * 100):.1f}%')
"
```

### Phân tích kết quả:
- **Loss giảm < 30%**: Model chưa converge → cần train thêm hoặc tăng capacity
- **Loss giảm 30-60%**: Model đang học nhưng chưa đủ → train thêm 10-15 epochs nữa
- **Loss giảm > 60%**: Model đã converge tốt → vấn đề có thể ở prediction script

## Bước 3: Dựa trên kết quả chẩn đoán

### Nếu data bị corrupt:
```bash
# Kiểm tra encoding
file -i data/train_mtup_unified.txt

# Nếu không phải UTF-8, cần fix:
iconv -f ISO-8859-1 -t UTF-8 data/train_mtup_unified.txt > data/train_mtup_unified_fixed.txt
```

### Nếu model chưa converge (loss giảm < 60%):

**RECOMMENDED: Tăng capacity và train lại từ đầu**

Vì model hiện tại có vẻ không học được gì (generate "Task 5" thay vì "Task 2"),
nên nên xóa và train lại với capacity cao hơn:

```bash
# Xóa model cũ
rm -rf outputs/mtup_260110/mtup_v2

# Train lại với LoRA rank 64 (cao hơn 32), 20 epochs (nhiều hơn 15)
python mtup_v2/scripts/train_mtup_higher_capacity.py \
    --data_path data/train_mtup_unified.txt \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --output_dir outputs/mtup_260110/mtup_v2_rank64 \
    --epochs 20
```

Thay đổi quan trọng:
- LoRA rank: 64 (tăng từ 32) → More parameters to learn complex patterns
- LoRA alpha: 32 (tăng từ 16) → Stronger adaptation
- Learning rate: 3e-5 (giảm từ 5e-5) → More stable training
- Epochs: 20 (tăng từ 15) → More training time

### Nếu model đã converge tốt (loss giảm > 60%):
Vấn đề có thể ở prediction script - cần kiểm tra lại cách extract Task 2.

## Bước 4: Gửi kết quả chẩn đoán

Chạy lệnh ở Bước 1 và gửi output cho mình để quyết định hướng xử lý.
