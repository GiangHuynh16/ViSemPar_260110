# 🚀 FULL MTUP TRAINING GUIDE

## ✅ Đã Verify

Training đã chạy thành công với 25 samples (minimal mode). Bây giờ có thể chạy full training!

**Settings đã verify không OOM:**
- ✅ Batch size: 1
- ✅ Gradient accumulation: 1
- ✅ CPU offload: enabled
- ✅ Bitsandbytes: uninstalled

---

## 🎯 OPTION 1: Full Training Trong Tmux (Khuyến nghị)

### Bước 1: Tạo tmux session

```bash
cd ~/ViSemPar_new1
git pull origin main

# Tạo tmux session mới
tmux new -s mtup_full
```

### Bước 2: Chạy full training

Trong tmux session:

```bash
conda activate lora_py310
bash RUN_FULL_TRAINING.sh
```

### Bước 3: Detach khỏi tmux

Khi training đã bắt đầu, nhấn:
- `Ctrl+B` rồi nhấn `D`

Training sẽ tiếp tục chạy background.

### Bước 4: Reattach để xem progress

```bash
tmux attach -t mtup_full
```

### Bước 5: Theo dõi từ xa (optional)

Mở terminal khác và xem logs real-time:

```bash
# Xem training logs
tail -f outputs/logs/mtup_*/events.out.tfevents.*

# Hoặc TensorBoard
tensorboard --logdir outputs/logs --port 6006
# Mở browser: http://server_ip:6006
```

---

## 🎯 OPTION 2: Chạy Trực Tiếp (Không tmux)

Nếu kết nối SSH stable:

```bash
cd ~/ViSemPar_new1
conda activate lora_py310
git pull origin main
bash RUN_FULL_TRAINING.sh
```

**Lưu ý**: Nếu SSH disconnect, training sẽ dừng!

---

## 📊 Training Info

### Dataset
- **Training samples**: ~1200 (full ViAMR dataset)
- **Validation samples**: ~150
- **Epochs**: 10

### Time Estimate
- **Per epoch**: ~20-30 phút
- **Total time**: 3-6 giờ
- **GPU usage**: ~20-21 GB

### Checkpoints
Saved every 250 steps tại:
```
outputs/checkpoints_mtup/
├── checkpoint-250/
├── checkpoint-500/
├── checkpoint-750/
└── ...
```

### Logs
TensorBoard logs tại:
```
outputs/logs/mtup_YYYYMMDD_HHMMSS/
```

---

## 🔍 Monitor Progress

### Xem logs trong tmux
```bash
tmux attach -t mtup_full
```

### Xem GPU usage
```bash
# Terminal khác
watch -n 1 nvidia-smi
```

### Xem TensorBoard
```bash
tensorboard --logdir outputs/logs --bind_all --port 6006
```

---

## 🛑 Dừng Training (Nếu cần)

### Dừng tạm (trong tmux)
- `Ctrl+C` trong tmux session

### Kill tmux session
```bash
tmux kill-session -t mtup_full
```

### Resume từ checkpoint
```bash
# Training sẽ tự động resume từ checkpoint mới nhất
bash RUN_FULL_TRAINING.sh
```

---

## ✅ Sau Khi Training Xong

### 1. Tìm checkpoint tốt nhất

```bash
# List checkpoints
ls -lh outputs/checkpoints_mtup/

# Training args sẽ chọn checkpoint có lowest eval loss
# Check file: outputs/checkpoints_mtup/checkpoint-XXXX/
```

### 2. Evaluate trên test set

```bash
python3 evaluate_test_data.py
```

### 3. Inference thử

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load model
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-3B-Instruct",
    device_map="auto",
    torch_dtype="auto"
)

# Load LoRA adapter
model = PeftModel.from_pretrained(
    base_model,
    "outputs/checkpoints_mtup/checkpoint-BEST"  # Thay BEST bằng số
)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")

# Test
text = "Sentence: Tôi ăn cơm\n\nTask 1: Generate AMR structure without variables"
inputs = tokenizer(text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_length=500)
print(tokenizer.decode(outputs[0]))
```

---

## 🆘 Troubleshooting

### OOM Error
Nếu vẫn OOM, giảm batch size xuống 1:
```bash
# Already using batch_size=1, grad_accum=1
# Nếu vẫn OOM, chuyển sang model 1.5B:
python3 train_mtup.py --use-case full_training --no-quantize \
  --model-name Qwen/Qwen2.5-1.5B-Instruct \
  --batch-size 1 --grad-accum 1
```

### Bitsandbytes Error
```bash
# Uninstall again
pip uninstall -y bitsandbytes
conda uninstall -y bitsandbytes
```

### Training quá chậm
- Kiểm tra GPU usage: `nvidia-smi`
- Nếu GPU < 80%, CPU offload đang bottleneck
- Giảm CPU offload: edit train_mtup.py, tăng max_memory từ "20GB" lên "22GB"

---

## 📝 TÓM TẮT

**CHẠY NGAY:**

```bash
cd ~/ViSemPar_new1
git pull origin main
tmux new -s mtup_full
conda activate lora_py310
bash RUN_FULL_TRAINING.sh
# Ctrl+B, D để detach
```

**XEM PROGRESS:**

```bash
tmux attach -t mtup_full
```

**Estimated completion time: 3-6 giờ**
