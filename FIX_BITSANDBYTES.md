# 🔧 FIX BITSANDBYTES - GIẢI PHÁP CUỐI CÙNG

## 🎯 ROOT CAUSE ĐÃ TÌM RA

**BitsAndBytes** không tương thích với CUDA 11.8 trên server:

```
WARNING: Could not find bitsandbytes CUDA binary at libbitsandbytes_cuda128.so
ModuleNotFoundError: No module named 'triton.ops'
```

---

## ✅ GIẢI PHÁP NHANH NHẤT (KHUYẾN NGHỊ)

### Chạy training KHÔNG dùng quantization

Tôi đã thêm flag `--no-quantize` vào code. Bây giờ bạn chỉ cần:

**Trên server:**

```bash
cd ~/ViSemPar_new1
conda activate lora_py310

# Pull code mới nhất (có flag --no-quantize)
git pull origin main

# Chạy training KHÔNG dùng quantization
python train_mtup.py --use-case quick_test --show-sample --no-quantize
```

**Lưu ý:**
- Model sẽ load FP16 (float16) thay vì 4-bit
- GPU memory sẽ tăng từ ~6GB lên ~10GB (vẫn OK với RTX 6000 23GB)
- Training vẫn nhanh vì dùng LoRA

---

## 📊 So Sánh

| Mode | GPU Memory | Speed | Accuracy |
|------|-----------|-------|----------|
| **4-bit (bitsandbytes)** | ~6GB | Nhanh nhất | Tốt |
| **FP16 (--no-quantize)** | ~10GB | Nhanh | Tốt nhất |

Với GPU của bạn (23GB), **FP16 mode hoàn toàn OK** và thậm chí có thể **chính xác hơn**!

---

## 🚀 LỆNH ĐẦY ĐỦ

```bash
# 1. SSH vào server
ssh your_server

# 2. Activate environment
conda activate lora_py310

# 3. Vào thư mục project
cd ~/ViSemPar_new1

# 4. Pull code mới
git pull origin main

# 5. Quick test (100 samples, 1 epoch)
python train_mtup.py --use-case quick_test --show-sample --no-quantize

# 6. Full training (trong tmux)
tmux new -s amr
python train_mtup.py --use-case full_training --no-quantize
# Ctrl+B, D để detach
```

---

## 🔍 OUTPUT MONG ĐỢI

```
Using 4-bit quantization: False
⚠️  Quantization DISABLED by --no-quantize flag
   Training will use more GPU memory

Loading model...
✓ Model loaded
✓ Tokenizer loaded

Applying LoRA...
trainable params: 7.08M || all params: 3.09B || trainable%: 0.23%

Training...
```

**KHÔNG còn lỗi bitsandbytes!**

---

## 🆚 NẾU MUỐN FIX BITSANDBYTES (Tùy chọn)

Nếu bạn vẫn muốn dùng 4-bit quantization:

```bash
conda activate lora_py310

# Cài từ conda thay vì pip
conda uninstall -y bitsandbytes
conda install -y bitsandbytes -c conda-forge

# Hoặc cài từ source
pip uninstall -y bitsandbytes
pip install bitsandbytes==0.44.1 --no-build-isolation

# Verify
python -c "import bitsandbytes as bnb; print(bnb.__version__)"
```

Nhưng thật sự **KHÔNG cần thiết** vì:
- FP16 mode đã đủ nhanh với LoRA
- GPU 23GB đủ cho model 3B
- Accuracy thậm chí tốt hơn

---

## 📝 TÓM TẮT

**NGAY BÂY GIỜ:**

```bash
cd ~/ViSemPar_new1
conda activate lora_py310
git pull origin main
python train_mtup.py --use-case quick_test --no-quantize
```

**Xong!** Training sẽ chạy không lỗi.
