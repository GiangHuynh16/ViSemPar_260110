# Tóm tắt MTUP v2 - Tiếng Việt

## ✅ Đã hoàn thành

### 1. Dọn dẹp và tổ chức lại
- ✅ Di chuyển 92 file markdown cũ vào `archive/mtup_v1/`
- ✅ Di chuyển 6 scripts training/prediction cũ vào archive
- ✅ Root directory giờ sạch sẽ, chỉ có files mới

### 2. Tạo MTUP v2 mới hoàn toàn
- ✅ 4 Python scripts (1,264 dòng code)
- ✅ 10 files documentation đầy đủ
- ✅ Data preprocessing script (đã test thành công)
- ✅ Training script với unified prompt
- ✅ Prediction script (extract Task 2)
- ✅ Evaluation script (SMATCH)

### 3. Preprocessing data
- ✅ Chạy thành công: 1,262 samples valid
- ✅ File output: `data/train_mtup_unified.txt` (1.5 MB)
- ✅ Format: Unified prompt cho cả 2 tasks

## 🎯 Điểm quan trọng nhất

### MTUP = Multi-Task Unified Prompting

**SAI (v1 cũ):** Train 2 models riêng
- Model 1: Sentence → Skeleton
- Model 2: Skeleton → Full AMR

**ĐÚNG (v2 mới):** Train 1 model duy nhất với 1 prompt chung
- Model: Sentence → [Task 1: Skeleton, Task 2: Full AMR]
- Lợi ích: Model học cả 2 tasks cùng lúc, shared knowledge, F1 cao hơn

### Co-reference (QUAN TRỌNG!)

Đây là yếu tố quyết định F1 score cao!

**Quy tắc:** Định nghĩa biến 1 lần, sau đó tái sử dụng

```
✅ ĐÚNG:
(b / bác_sĩ :domain(t / tôi))  ← Định nghĩa 't'
(l / làm :ARG0 t ...)           ← Tái sử dụng 't'

❌ SAI:
(b / bác_sĩ :domain(t / tôi))  ← Định nghĩa 't'
(l / làm :ARG0(t / tôi) ...)   ← Định nghĩa lại 't' → LỖI!
```

## 📂 Cấu trúc files

```
ViSemPar_new1/
├── START_HERE.md              ← Bắt đầu từ đây
├── README.md                  ← Tổng quan
├── MTUP_V2_QUICKSTART.md      ← Hướng dẫn nhanh
├── COPY_TO_SERVER.md          ← Cách copy lên server
├── FINAL_SUMMARY.txt          ← Tóm tắt chi tiết
│
├── mtup_v2/                   ← Implementation mới
│   ├── scripts/
│   │   ├── train_mtup_unified.py
│   │   ├── predict_mtup_unified.py
│   │   └── evaluate.py
│   ├── preprocessing/
│   │   └── create_mtup_data.py
│   └── docs/
│       ├── MTUP_CONCEPT.md
│       ├── TRAINING_GUIDE.md
│       └── COREFERENCE_EXAMPLES.md
│
├── data/
│   └── train_mtup_unified.txt ← ✅ SẴN SÀNG (1,262 samples)
│
└── archive/
    └── mtup_v1/               ← Files cũ (98 files)
```

## 🚀 Các bước tiếp theo

### Bước 1: Copy lên server (5 phút)
```bash
# Tạo tarball
tar -czf mtup_v2.tar.gz mtup_v2/ data/train_mtup_unified.txt

# Copy lên server
scp mtup_v2.tar.gz user@server:/path/to/ViSemPar_new1/

# Giải nén trên server
ssh user@server
cd /path/to/ViSemPar_new1
tar -xzf mtup_v2.tar.gz
```

### Bước 2: Training trên server (2-3 giờ)
```bash
# Kích hoạt environment
conda activate amr

# Chạy training
nohup python3 mtup_v2/scripts/train_mtup_unified.py \
    --data_path data/train_mtup_unified.txt \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --output_dir outputs/mtup_v2 \
    --epochs 5 \
    > logs/train.log 2>&1 &

# Monitor
tail -f logs/train.log
nvidia-smi -l 1
```

### Bước 3: Prediction (10 phút)
```bash
python3 mtup_v2/scripts/predict_mtup_unified.py \
    --adapter_path outputs/mtup_v2/final_adapter \
    --input_file data/public_test.txt \
    --output_file outputs/predictions.txt
```

### Bước 4: Evaluation (1 phút)
```bash
pip install smatch  # Nếu chưa có

python3 mtup_v2/scripts/evaluate.py \
    --predictions outputs/predictions.txt \
    --ground_truth data/public_test_ground_truth.txt
```

## 📊 Kết quả mong đợi

### Training:
- Thời gian: ~2-3 giờ (5 epochs trên RTX 4090)
- Loss: Bắt đầu ~2.5 → Kết thúc ~1.0
- VRAM: ~20-22GB

### Evaluation:
- **Baseline F1:** 0.47
- **Mục tiêu:** F1 > 0.47
- **Tốt:** F1 > 0.50 (+6%)
- **Xuất sắc:** F1 > 0.52 (+10%)

## 📚 Đọc gì trước?

### Đọc nhanh (30 phút):
1. [START_HERE.md](START_HERE.md) - 5 phút
2. [MTUP_V2_QUICKSTART.md](MTUP_V2_QUICKSTART.md) - 15 phút
3. [COPY_TO_SERVER.md](COPY_TO_SERVER.md) - 10 phút
4. Bắt đầu training!

### Đọc đầy đủ (1 giờ):
1. [README.md](README.md) - 10 phút
2. [mtup_v2/docs/MTUP_CONCEPT.md](mtup_v2/docs/MTUP_CONCEPT.md) - 15 phút
3. [MTUP_V2_QUICKSTART.md](MTUP_V2_QUICKSTART.md) - 15 phút
4. [mtup_v2/docs/TRAINING_GUIDE.md](mtup_v2/docs/TRAINING_GUIDE.md) - 20 phút

## 🔧 Xử lý lỗi thường gặp

### Out of Memory (OOM):
Edit file `train_mtup_unified.py`:
```python
per_device_train_batch_size=1  # Giảm từ 2
gradient_accumulation_steps=32  # Tăng từ 16
```

### Predictions không có biến (không có dấu `/`):
```bash
# Model cần train lâu hơn
--epochs 10
```

### Lỗi Duplicate node:
Model chưa học tốt co-reference. Cần train lâu hơn hoặc check training data.

## ✅ Checklist

### Đã xong:
- [x] Architecture sạch sẽ
- [x] Code hoàn chỉnh (1,264 dòng)
- [x] Documentation đầy đủ (10 files)
- [x] Data preprocessing (1,262 samples)
- [x] Test validation: 100% pass
- [x] Sẵn sàng training

### Cần làm:
- [ ] Copy lên server
- [ ] Training (~3 giờ)
- [ ] Prediction
- [ ] Evaluation
- [ ] So sánh với baseline

## 🎉 Khi nào thành công?

Bạn thành công khi:
1. ✅ Training chạy xong không lỗi
2. ✅ F1 > 0.47 (cao hơn baseline)
3. ✅ Predictions có format PENMAN đúng
4. ✅ Có variables trong output
5. ✅ Ngoặc cân bằng

## 💡 Lưu ý quan trọng

1. **Co-reference là QUAN TRỌNG NHẤT** - Đây là yếu tố quyết định F1 cao
2. **Monitor GPU** - Chạy `nvidia-smi -l 1` để theo dõi
3. **Save checkpoints** - Mặc định save mỗi epoch
4. **Test sớm** - Sau epoch 1 thử test vài samples
5. **Backup adapter** - Copy `final_adapter/` trước khi train lại

## 📞 Cần hỗ trợ?

- **Quick questions:** [MTUP_V2_QUICKSTART.md](MTUP_V2_QUICKSTART.md)
- **Training issues:** [mtup_v2/docs/TRAINING_GUIDE.md](mtup_v2/docs/TRAINING_GUIDE.md)
- **All commands:** [RUN_COMMANDS.sh](RUN_COMMANDS.sh)
- **Chi tiết kỹ thuật:** [FINAL_SUMMARY.txt](FINAL_SUMMARY.txt)

## 🏆 Mục tiêu

**Đánh bại Baseline F1: 0.47 → Mục tiêu: >0.47 → Tốt nhất: >0.50**

---

**Trạng thái:** ✅ SẴN SÀNG TRAINING
**Phiên bản:** 2.0
**Ngày:** 2026-01-10

🚀 **Bắt đầu thôi! Chúc may mắn!** 🚀

---

## Hành động tiếp theo

1. Đọc [START_HERE.md](START_HERE.md) hoặc [MTUP_V2_QUICKSTART.md](MTUP_V2_QUICKSTART.md)
2. Copy files lên server theo [COPY_TO_SERVER.md](COPY_TO_SERVER.md)
3. Chạy training
4. Đánh giá kết quả
5. 🎉 Hy vọng F1 > 0.47!
