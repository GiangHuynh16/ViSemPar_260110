# 🚨 HƯỚNG DẪN NHANH: Fix Model Training

Model hiện tại **KHÔNG học được gì** (generate "Task 5" thay vì "Task 2").

## TL;DR - Làm gì bây giờ?

**Bước 1: Chẩn đoán**
```bash
python mtup_v2/scripts/diagnose_model.py \
    --data_path data/train_mtup_unified.txt \
    --adapter_path outputs/mtup_260110/mtup_v2/final_adapter
```

**Bước 2: Train lại với capacity cao hơn (RECOMMENDED)**
```bash
# Xóa model cũ
rm -rf outputs/mtup_260110/mtup_v2

# Train lại
python mtup_v2/scripts/train_mtup_higher_capacity.py \
    --data_path data/train_mtup_unified.txt \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --output_dir outputs/mtup_260110/mtup_v2_rank64 \
    --epochs 20
```

Training sẽ mất khoảng 3-4 giờ.

**Bước 3: Test lại**
```bash
python mtup_v2/scripts/debug_prediction.py \
    --adapter_path outputs/mtup_260110/mtup_v2_rank64/final_adapter \
    --test_sentence "bi kịch là ở chỗ đó !"
```

---

## Chi tiết: Tại sao model thất bại?

### Hiện tượng:
Model generate:
```
Task  1: (k ? / k :compound(chziaÅ) :mode interrogative)
Task ï¼': (ï½‹ / k ? :compound(ï½ƒï½ˆï¼'áº¥ï½: interrogative)
```

### Vấn đề:
1. ❌ "Task ï¼'" thay vì "Task 2" → Model không học được task structure
2. ❌ Parse sai hoàn toàn → Model không hiểu AMR parsing
3. ❌ Có thể do LoRA rank quá thấp (32) cho task phức tạp này

### Giải pháp:
**Tăng model capacity** để học tốt hơn:

| Parameter | Old Value | New Value | Lý do |
|-----------|-----------|-----------|-------|
| LoRA rank | 32 | 64 | More parameters to learn complex patterns |
| LoRA alpha | 16 | 32 | Stronger LoRA adaptation |
| Learning rate | 5e-5 | 3e-5 | More stable, less likely to overshoot |
| Epochs | 15 | 20 | More training iterations |

---

## Checklist sau khi train xong

Sau khi train xong với `train_mtup_higher_capacity.py`, test bằng debug script:

```bash
python mtup_v2/scripts/debug_prediction.py \
    --adapter_path outputs/mtup_260110/mtup_v2_rank64/final_adapter \
    --test_sentence "bi kịch là ở chỗ đó !"
```

### ✅ Output đúng phải có dạng:
```
Task 1: (bi_kịch :domain(chỗ :mod(đó)))
Task 2: (b / bi_kịch :domain(c / chỗ :mod(đ / đó)))
```

### Kiểm tra:
- [ ] Có "Task 1:" và "Task 2:" (KHÔNG phải "Task 5" hay "Task ï¼'")
- [ ] Task 1 có structure đúng: `(bi_kịch ...)`
- [ ] Task 2 có variables: `(b / bi_kịch ...)`
- [ ] Số lượng `(` bằng số lượng `)`

---

## Nếu vẫn thất bại sau khi train với rank 64?

Có 3 khả năng:

### 1. Data bị corrupt
```bash
# Kiểm tra encoding
file -i data/train_mtup_unified.txt

# Phải là: charset=utf-8
# Nếu không, re-download hoặc convert
```

### 2. Dataset quá nhỏ (1840 samples)
Cần augment thêm data hoặc thử pre-train approach:
- Train base model trên large corpus trước
- Fine-tune trên task cụ thể sau

### 3. Base model không phù hợp
Thử model khác:
- `Qwen/Qwen2.5-14B-Instruct` (nếu có đủ VRAM)
- `meta-llama/Llama-3.1-8B-Instruct`

---

## Gửi kết quả nếu cần hỗ trợ

Sau khi chạy diagnosis, gửi:
1. Output của `diagnose_model.py`
2. Loss progression (first và last loss)
3. Output của `debug_prediction.py` với model mới
