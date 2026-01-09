# MTUP v2 - Training Guide

## Tổng quan

Hướng dẫn này sẽ giúp bạn train model MTUP v2 từ đầu đến cuối.

## Yêu cầu hệ thống

- GPU: ≥24GB VRAM (RTX 3090, 4090, A100, etc.)
- RAM: ≥32GB
- Disk: ≥50GB free space
- CUDA: ≥11.8
- Python: 3.8+

## Cài đặt

```bash
# Clone repo (if needed)
cd ViSemPar_new1

# Install dependencies
pip install torch transformers datasets peft accelerate bitsandbytes
pip install sentencepiece protobuf
pip install smatch  # For evaluation

# Optional: Install flash-attention-2 for faster training
pip install flash-attn --no-build-isolation
```

## Workflow hoàn chỉnh

### Bước 1: Preprocessing - Tạo Unified Data

```bash
python mtup_v2/preprocessing/create_mtup_data.py
```

**Output:** `data/train_mtup_unified.txt`

**Kiểm tra:**
```bash
# Xem số samples
wc -l data/train_mtup_unified.txt

# Xem ví dụ đầu tiên
head -50 data/train_mtup_unified.txt
```

**Format mong đợi:**
```
<|im_start|>system
Bạn là chuyên gia phân tích AMR...
<|im_end|>
<|im_start|>user
Câu: bi kịch là ở chỗ đó !
<|im_end|>
<|im_start|>assistant
Task 1: (bi_kịch :domain(chỗ :mod(đó)))
Task 2: (b / bi_kịch :domain(c / chỗ :mod(đ / đó)))
<|im_end|>

<|im_start|>system
...
```

### Bước 2: Training

#### Trên local machine (test)

```bash
python mtup_v2/scripts/train_mtup_unified.py \
    --data_path data/train_mtup_unified.txt \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --output_dir outputs/mtup_v2_unified_test \
    --epochs 1
```

#### Trên server (production)

```bash
# SSH vào server
ssh user@server

# Navigate to project
cd /path/to/ViSemPar_new1

# Activate environment
conda activate amr  # or your env name

# Run training
nohup python mtup_v2/scripts/train_mtup_unified.py \
    --data_path data/train_mtup_unified.txt \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --output_dir outputs/mtup_v2_unified_production \
    --epochs 5 \
    > logs/train_mtup_v2.log 2>&1 &

# Monitor progress
tail -f logs/train_mtup_v2.log

# Check GPU usage
nvidia-smi -l 1
```

**Training time estimate:**
- ~3000 samples
- 5 epochs
- Batch size 2, grad accumulation 16
- ~2-3 hours on RTX 4090

**Expected output:**
```
outputs/mtup_v2_unified_production/
├── checkpoint-epoch-1/
├── checkpoint-epoch-2/
├── ...
└── final_adapter/
    ├── adapter_config.json
    ├── adapter_model.safetensors
    └── ...
```

### Bước 3: Prediction

```bash
python mtup_v2/scripts/predict_mtup_unified.py \
    --base_model Qwen/Qwen2.5-7B-Instruct \
    --adapter_path outputs/mtup_v2_unified_production/final_adapter \
    --input_file data/public_test.txt \
    --output_file outputs/predictions_mtup_v2.txt
```

**Kiểm tra output:**
```bash
# Xem số predictions
wc -l outputs/predictions_mtup_v2.txt

# Xem 5 predictions đầu
head -5 outputs/predictions_mtup_v2.txt

# Check format (should have variables)
head -1 outputs/predictions_mtup_v2.txt | grep -o "/" | wc -l
# Should return > 0 (có dấu /)
```

### Bước 4: Evaluation

```bash
python mtup_v2/scripts/evaluate.py \
    --predictions outputs/predictions_mtup_v2.txt \
    --ground_truth data/public_test_ground_truth.txt \
    --output_comparison outputs/comparison_mtup_v2.txt
```

**Output:**
```
📊 OFFICIAL SMATCH SCORES
==========================
   Precision: 0.5234 (52.34%)
   Recall:    0.4987 (49.87%)
   F1 Score:  0.5108 (51.08%)
==========================
```

**So sánh với baseline:**
- Baseline F1: 0.47 (47%)
- MTUP v2 F1: ? (target > 0.47)

## Troubleshooting

### Issue 1: OOM (Out of Memory)

**Triệu chứng:**
```
RuntimeError: CUDA out of memory
```

**Giải pháp:**
1. Giảm batch size:
```python
# In train_mtup_unified.py, line ~195
per_device_train_batch_size=1,  # Giảm từ 2 xuống 1
gradient_accumulation_steps=32,  # Tăng từ 16 lên 32
```

2. Giảm max_length:
```python
# In train_mtup_unified.py, line ~151
max_length=1536,  # Giảm từ 2048 xuống 1536
```

3. Enable gradient checkpointing (đã bật mặc định):
```python
gradient_checkpointing=True,
```

### Issue 2: Loss NaN

**Triệu chứng:**
```
Step 50: loss=nan
```

**Giải pháp:**
1. Giảm learning rate:
```python
learning_rate=5e-5,  # Giảm từ 1e-4
```

2. Kiểm tra data:
```bash
# Check for empty/invalid AMRs
grep -n "(a / amr-empty)" data/train_mtup_unified.txt
```

### Issue 3: Model không học Task 2

**Triệu chứng:**
Predictions không có biến (không có dấu `/`)

**Giải pháp:**
1. Kiểm tra prompt masking:
```python
# In train_mtup_unified.py, line ~149
# Make sure labels are masked correctly
```

2. Tăng số epochs:
```bash
--epochs 10  # Tăng từ 5 lên 10
```

3. Kiểm tra training data:
```bash
# Should have both Task 1 and Task 2
grep "Task 2:" data/train_mtup_unified.txt | head -5
```

### Issue 4: Duplicate Node Error trong predictions

**Triệu chứng:**
```
Error: Duplicate node definition 't'
```

**Giải pháp:**
Đây là vấn đề model chưa học tốt co-reference. Cần:
1. Train lâu hơn (more epochs)
2. Tăng LoRA rank:
```python
r=128,  # Tăng từ 64
```
3. Thêm nhiều examples về co-reference vào training data

## Monitoring Training

### Metrics quan trọng

1. **Loss giảm dần:**
```
Epoch 1: loss=2.5
Epoch 2: loss=1.8
Epoch 3: loss=1.3
...
```

2. **GPU Utilization:**
```bash
nvidia-smi
# Should see ~90-95% GPU usage
```

3. **Sample predictions during training:**
Thỉnh thoảng test 1 sample để xem model học như thế nào:
```bash
# After each epoch
python mtup_v2/scripts/predict_mtup_unified.py \
    --adapter_path outputs/.../checkpoint-epoch-X \
    --input_file data/test_sample.txt \
    --output_file outputs/test_epoch_X.txt

# Compare outputs
cat outputs/test_epoch_*.txt
```

## Best Practices

### 1. Incremental Training
Train theo bước:
- Epoch 1-2: Học cấu trúc cơ bản
- Epoch 3-4: Học variable assignment
- Epoch 5+: Fine-tune co-reference

### 2. Data Validation
Trước khi train, luôn validate:
```bash
python mtup_v2/preprocessing/create_mtup_data.py
# Should show: "✅ Valid samples: X/Y"
# X should be close to Y
```

### 3. Checkpoint Management
Giữ checkpoints quan trọng:
```bash
# Copy best checkpoint
cp -r outputs/.../checkpoint-epoch-3 outputs/best_checkpoint
```

### 4. Experiment Tracking
Log tất cả experiments:
```bash
# Create experiment log
echo "Experiment: MTUP_v2_run1" > logs/experiments.txt
echo "Date: $(date)" >> logs/experiments.txt
echo "Config: epochs=5, lr=1e-4, r=64" >> logs/experiments.txt
echo "Result: F1=0.XX" >> logs/experiments.txt
echo "---" >> logs/experiments.txt
```

## Next Steps

Sau khi đạt F1 > baseline:
1. Thử các model size khác (1.5B, 14B)
2. Thử LoRA rank khác (32, 128, 256)
3. Thử learning rate khác (5e-5, 2e-4)
4. Ensemble nhiều models
5. Post-processing rules

## Liên hệ

Nếu gặp vấn đề, check:
1. Logs: `logs/train_mtup_v2.log`
2. Archive: `archive/mtup_v1/` (old approaches)
3. Documentation: `mtup_v2/docs/`
