# Optimization Applied - MTUP Training

**Tối ưu hóa đã áp dụng trước khi deploy lên server**

---

## 🎯 **TÓM TẮT OPTIMIZATION**

Ngày áp dụng: 2024-12-24
Mục tiêu: Tối ưu hóa training performance và accuracy cho MTUP strategy

---

## ✅ **CÁC THAY ĐỔI ĐÃ ÁP DỤNG**

### 1. **Training Configuration (config/config_mtup.py)**

#### **Learning Rate: 3e-4 → 2e-4**
```python
# TRƯỚC:
"learning_rate": 3e-4,              # Slightly higher for smaller models

# SAU (OPTIMIZED):
"learning_rate": 2e-4,              # OPTIMIZED: Lower for stable training
```

**Lý do:**
- 3e-4 quá cao cho 3B model → có thể overshooting
- 2e-4 stable hơn, convergence mượt mà hơn
- Best practice cho LoRA fine-tuning với small models

**Kết quả mong đợi:**
- Training loss ổn định hơn
- Tránh divergence
- Better final performance

---

#### **Number of Epochs: 15 → 10**
```python
# TRƯỚC:
"num_train_epochs": 15,              # Fewer epochs (MTUP learns faster)

# SAU (OPTIMIZED):
"num_train_epochs": 10,              # OPTIMIZED: MTUP converges faster, 10 epochs sufficient
```

**Lý do:**
- MTUP với explicit supervision → converge nhanh hơn
- 15 epochs có thể overfitting trên ~2500 examples
- 10 epochs đủ cho MTUP strategy

**Kết quả mong đợi:**
- Tiết kiệm ~33% training time
- Tránh overfitting
- Better generalization

---

#### **Validation Split: 5% → 10%**
```python
# TRƯỚC:
"validation_split": 0.05,        # 5% for validation

# SAU (OPTIMIZED):
"validation_split": 0.1,         # OPTIMIZED: 10% for better validation monitoring
```

**Lý do:**
- 5% (~125 examples) quá ít để đánh giá reliable
- 10% (~250 examples) cho validation signal tốt hơn
- Vẫn giữ đủ training data (90% = ~2250 examples)

**Kết quả mong đợi:**
- Validation metrics reliable hơn
- Better early stopping decisions
- Monitor overfitting tốt hơn

---

### 2. **Training Use Cases (train_mtup.py)**

#### **Quick Test**
```bash
# Không đổi - vẫn 100 samples, 1 epoch
# Thêm: --lr 2e-4 default
```

#### **Fast Iteration: 2 epochs → 3 epochs**
```python
# TRƯỚC:
# Fast iteration: 500 samples, 2 epochs

# SAU (OPTIMIZED):
# Fast iteration: 500 samples, 3 epochs
args.epochs = args.epochs or 3
args.lr = args.lr or 2e-4
```

**Lý do:**
- 2 epochs chưa đủ để model học tốt
- 3 epochs vẫn nhanh (~30 min) nhưng accuracy tốt hơn

---

#### **Full Training: 3 epochs → 10 epochs**
```python
# TRƯỚC:
# Full training: all data, 3 epochs

# SAU (OPTIMIZED):
# Full training: all data, 10 epochs
args.epochs = args.epochs or 10
args.lr = args.lr or 2e-4
```

**Lý do:**
- 3 epochs quá ít cho production model
- 10 epochs optimal cho MTUP (không quá nhiều, không quá ít)
- Consistent với config default

---

### 3. **Documentation Updates**

#### **QUICK_COMMANDS.md**
- Cập nhật training commands với optimized values
- Thêm recommended flag (⭐) cho full training
- Thêm example cho 7B model training

---

## 📊 **SO SÁNH TRƯỚC VÀ SAU**

| Metric | TRƯỚC | SAU (OPTIMIZED) | Improvement |
|--------|-------|-----------------|-------------|
| Learning Rate | 3e-4 | 2e-4 | More stable |
| Epochs (full) | 3 | 10 | Better learning |
| Validation Split | 5% | 10% | Better monitoring |
| Fast Iteration Epochs | 2 | 3 | Better convergence |
| Expected Training Time | ~1h (underfit) | ~2.5h (optimal) | Quality over speed |

---

## 🎯 **KẾT QUẢ MONG ĐỢI**

### **Training Metrics:**
- ✅ Training loss: Ổn định hơn, giảm đều
- ✅ Validation loss: Không fluctuate nhiều
- ✅ No divergence/explosion
- ✅ Smooth learning curve

### **Model Performance:**
- **Target SMATCH F1**: 70-80%
- **Realistic với 3B model**: 68-75%
- **Với 7B model**: 75-82%

### **Training Time:**
- **Quick test**: ~10 minutes (không đổi)
- **Fast iteration**: ~30-40 minutes (tăng 50% nhưng quality tốt hơn)
- **Full training (3B)**: ~2.5 hours (tăng từ 1h nhưng quality tốt hơn nhiều)
- **Full training (7B)**: ~6-7 hours

---

## 💡 **KHUYẾN NGHỊ SỬ DỤNG**

### **Lần đầu training:**
```bash
# Step 1: Verify pipeline
python3 train_mtup.py --use-case quick_test --show-sample

# Step 2: Fast iteration để test
python3 train_mtup.py --use-case fast_iteration

# Step 3: Nếu fast_iteration OK, chạy full training
tmux new -s amr-training
python3 train_mtup.py --use-case full_training
```

### **Production training (recommended):**
```bash
# 3B model - Fast & Good
python3 train_mtup.py \
  --model qwen2.5-3b \
  --epochs 10 \
  --batch-size 4 \
  --grad-accum 4 \
  --lr 2e-4 \
  --val-split 0.1

# 7B model - Best Accuracy (nếu có GPU mạnh)
python3 train_mtup.py \
  --model qwen2.5-7b \
  --epochs 15 \
  --batch-size 2 \
  --grad-accum 8 \
  --lr 1e-4 \
  --val-split 0.1
```

---

## 🔍 **MONITORING CHECKLIST**

Khi training, check các metrics này:

### **During Training:**
- [ ] Training loss giảm đều (không có spikes)
- [ ] Validation loss track với training loss
- [ ] No divergence (loss không tăng đột ngột)
- [ ] GPU utilization ~80-95%
- [ ] No OOM errors

### **After Training:**
- [ ] Final validation loss < 1.0
- [ ] SMATCH F1 > 65% (acceptable)
- [ ] SMATCH F1 > 70% (good)
- [ ] SMATCH F1 > 75% (excellent)
- [ ] Task 1 accuracy > Task 2 accuracy (expected)

---

## 🚨 **TROUBLESHOOTING**

### **Nếu validation loss tăng:**
- Giảm learning rate: `--lr 1e-4`
- Tăng weight decay: Edit config → `weight_decay: 0.02`
- Giảm epochs: `--epochs 8`

### **Nếu training quá chậm:**
- Dùng model nhỏ hơn: `--model qwen2.5-1.5b`
- Tăng batch size (nếu GPU cho phép): `--batch-size 8`

### **Nếu SMATCH < 65%:**
- Tăng epochs: `--epochs 15`
- Thử template khác: Edit config → `template_name: "v5_cot"`
- Dùng model lớn hơn: `--model qwen2.5-7b`

---

## 📝 **CHANGELOG**

### v1.1 (2024-12-24) - Optimization Applied
- ✅ Learning rate: 3e-4 → 2e-4
- ✅ Epochs (full): 3 → 10
- ✅ Validation split: 5% → 10%
- ✅ Fast iteration: 2 → 3 epochs
- ✅ Documentation updated
- ✅ Use case presets optimized

### v1.0 (2024-12-24) - Initial MTUP Implementation
- ✅ MTUP strategy implementation
- ✅ 5 Vietnamese templates
- ✅ Vietnamese character support
- ✅ Multi-model support

---

## 🎓 **LƯU Ý QUAN TRỌNG**

1. **Không cần train lâu với MTUP:**
   - MTUP với explicit supervision → converge nhanh
   - 10 epochs đủ, không cần 20-30 epochs như standard approach

2. **Validation split quan trọng:**
   - 10% validation cho reliable metrics
   - Monitor validation loss để early stopping

3. **Learning rate thấp = stable:**
   - 2e-4 cho 3B model
   - 1e-4 cho 7B model
   - Không nên > 3e-4

4. **MTUP benefits:**
   - Model nhỏ (3B) có thể đạt performance gần model lớn (7B)
   - Training nhanh hơn 2-3x
   - Easier subtasks → better learning

---

**Tất cả optimizations đã được apply vào code.**
**Sẵn sàng để pull về server và training!** 🚀