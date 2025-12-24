# 🔧 FIX MÔI TRƯỜNG - COPY PASTE VÀO SERVER

## Cách 1: Copy Script (Khuyến nghị)

**Bước 1:** Trên server, chạy lệnh này:

```bash
cd ~/ViSemPar_new1
```

**Bước 2:** Mở file [FIX_FINAL.sh](FIX_FINAL.sh), copy TOÀN BỘ nội dung

**Bước 3:** Trên server, tạo file:

```bash
cat > FIX_FINAL.sh << 'END_OF_SCRIPT'
```

**Bước 4:** Paste toàn bộ nội dung script vào, rồi nhấn Enter và gõ:

```bash
END_OF_SCRIPT
```

**Bước 5:** Chạy script:

```bash
chmod +x FIX_FINAL.sh
bash FIX_FINAL.sh
```

---

## Cách 2: Chạy Trực Tiếp (Nếu Cách 1 Không Được)

Copy-paste TOÀN BỘ đoạn sau vào terminal server và nhấn Enter:

```bash
#!/bin/bash
cd ~/ViSemPar_new1
source ~/anaconda3/etc/profile.d/conda.sh
conda activate lora_py310

echo "🗑️  Xóa packages cũ..."
pip uninstall -y torch torchvision torchaudio numpy pandas scikit-learn transformers huggingface-hub peft accelerate bitsandbytes 2>/dev/null
conda uninstall -y pytorch torchvision torchaudio 2>/dev/null

echo "📦 Cài NumPy 1.26.4..."
pip install "numpy==1.26.4" --no-cache-dir

echo "📦 Cài PyTorch 2.3.0 CUDA 11.8..."
conda install -y pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=11.8 -c pytorch -c nvidia

echo "📦 Cài pandas, sklearn..."
pip install pandas scikit-learn --no-cache-dir

echo "📦 Cài HuggingFace..."
pip install "huggingface-hub>=0.24.0,<1.0" transformers==4.46.3 accelerate==1.2.1 peft==0.13.2 bitsandbytes==0.44.1 --no-cache-dir

echo "📦 Cài packages khác..."
pip install datasets tqdm penman smatch python-dotenv tensorboard --no-cache-dir

echo ""
echo "✅ Kiểm tra imports..."
python << 'PYEOF'
import torch, numpy, pandas, sklearn, transformers, peft
print(f"✓ PyTorch: {torch.__version__} (CUDA {torch.version.cuda})")
print(f"✓ NumPy: {numpy.__version__}")
print(f"✓ Pandas: {pandas.__version__}")
print(f"✓ Transformers: {transformers.__version__}")
print(f"✓ PEFT: {peft.__version__}")
print("\n✅ XONG! Chạy training:")
print("  python train_mtup.py --use-case quick_test --show-sample")
PYEOF
```

---

## Cách 3: Từng Lệnh Một (Nếu Cả 2 Cách Trên Không Được)

```bash
# 1. Activate environment
cd ~/ViSemPar_new1
conda activate lora_py310

# 2. Xóa PyTorch cũ
pip uninstall -y torch torchvision torchaudio
conda uninstall -y pytorch torchvision torchaudio

# 3. Xóa NumPy cũ
pip uninstall -y numpy

# 4. Xóa các packages khác
pip uninstall -y pandas scikit-learn transformers huggingface-hub peft accelerate bitsandbytes

# 5. Cài NumPy
pip install "numpy==1.26.4" --no-cache-dir

# 6. Cài PyTorch từ conda (FIX NCCL)
conda install -y pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=11.8 -c pytorch -c nvidia

# 7. Cài pandas, sklearn
pip install pandas scikit-learn --no-cache-dir

# 8. Cài HuggingFace
pip install "huggingface-hub>=0.24.0,<1.0" --no-cache-dir
pip install transformers==4.46.3 --no-cache-dir
pip install accelerate==1.2.1 peft==0.13.2 bitsandbytes==0.44.1 --no-cache-dir

# 9. Cài packages khác
pip install datasets tqdm penman smatch python-dotenv tensorboard --no-cache-dir

# 10. Kiểm tra
python -c "import torch; print(f'PyTorch {torch.__version__} CUDA {torch.version.cuda}')"
python -c "import transformers; print(f'Transformers {transformers.__version__}')"

# 11. Chạy training
python train_mtup.py --use-case quick_test --show-sample
```

---

## Kết Quả Mong Đợi

Sau khi chạy xong, bạn sẽ thấy:

```
✓ PyTorch: 2.3.0+cu118 (CUDA 11.8)
✓ NumPy: 1.26.4
✓ Pandas: 2.2.x
✓ Transformers: 4.46.3
✓ PEFT: 0.13.2

✅ XONG! Chạy training:
  python train_mtup.py --use-case quick_test --show-sample
```

Và training sẽ chạy KHÔNG có lỗi `ncclCommRegister` hoặc NumPy nữa!

---

## Lưu Ý Quan Trọng

1. **PHẢI** activate environment `lora_py310` trước:
   ```bash
   conda activate lora_py310
   ```

2. **QUAN TRỌNG:** Cài PyTorch từ **conda** thay vì pip để tránh NCCL conflict:
   ```bash
   conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
   ```

3. Nếu conda install PyTorch chậm quá (>5 phút), nhấn Ctrl+C và dùng pip:
   ```bash
   pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 \
       --index-url https://download.pytorch.org/whl/cu118 --no-cache-dir
   ```

---

## Nếu Vẫn Lỗi

Ping tôi với output lỗi cụ thể, tôi sẽ fix ngay!
