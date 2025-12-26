# 🚀 Training Commands - Quick Reference

## 🎯 Train MTUP (Today)

### Option 1: Full Training (Recommended)
```bash
cd ~/ViSemPar_new1
git pull origin main

# Train MTUP with Qwen 2.5 7B
python3 train_mtup.py \
  --use-case full_training \
  --model qwen2.5-7b \
  --epochs 10
```

### Option 2: With Custom Output Path
```bash
python3 train_mtup.py \
  --use-case full_training \
  --model qwen2.5-7b \
  --epochs 10 \
  --output-dir outputs/models/mtup_two_task_7b
```

### Option 3: In Tmux (For Long Training)
```bash
# Start tmux session
tmux new -s mtup_training

# Train
python3 train_mtup.py \
  --use-case full_training \
  --model qwen2.5-7b \
  --epochs 10

# Detach: Ctrl+B, then D
# Re-attach: tmux attach -t mtup_training
```

## ⏰ Expected Time

- **full_training**: ~4-6 hours (full dataset, 10 epochs)
- **fast_iteration**: ~2-3 hours (full dataset, fewer epochs)
- **quick_test**: ~30 min (500 samples, 5 epochs)

## 📊 Training Options Explained

### Use Cases
| Use Case | Samples | Epochs | Time | Purpose |
|----------|---------|--------|------|---------|
| `quick_test` | 500 | 5 | ~30 min | Test if training works |
| `fast_iteration` | All | 7-8 | ~2-3h | Quick experiments |
| `full_training` | All | 10-15 | ~4-6h | **Best results** ✅ |

### Models Available
| Model | Size | Memory | Speed | F1 (Expected) |
|-------|------|--------|-------|---------------|
| `qwen2.5-1.5b` | 1.5B | ~8GB | Fast | ~0.40-0.44 |
| `qwen2.5-3b` | 3B | ~12GB | Medium | ~0.45-0.48 |
| `qwen2.5-7b` | 7B | ~18GB | Slower | **~0.49-0.53** ✅ |

## 📝 All Training Arguments

```bash
python3 train_mtup.py \
  --use-case full_training \        # quick_test | fast_iteration | full_training
  --model qwen2.5-7b \               # qwen2.5-1.5b | qwen2.5-3b | qwen2.5-7b
  --epochs 10 \                      # Number of epochs
  --batch-size 4 \                   # Per-device batch size
  --grad-accum 4 \                   # Gradient accumulation steps
  --lr 2e-4 \                        # Learning rate
  --max-length 2048 \                # Max sequence length
  --val-split 0.1 \                  # Validation split
  --log-steps 10 \                   # Log every N steps
  --save-steps 250 \                 # Save checkpoint every N steps
  --output-dir outputs/models/mtup_two_task_7b
```

## 🔍 Monitor Training

### Check Progress
```bash
# Watch log file
tail -f logs/training_mtup.log

# Check GPU usage
watch -n 1 nvidia-smi

# If using tmux
tmux attach -t mtup_training
```

### Expected Output
```
Loading model Qwen/Qwen2.5-7B-Instruct...
✓ Model loaded (7.62B parameters)
✓ LoRA configured (r=64, trainable: 67.1M params)

Preparing training data...
✓ Loaded 2500 training examples
✓ Validation split: 250 examples

Training:
Epoch 1/10: 100%|███████| 625/625 [45:23<00:00, loss=1.234]
Epoch 2/10: 100%|███████| 625/625 [45:18<00:00, loss=0.987]
...

✓ Training complete!
✓ Model saved to: outputs/checkpoints_mtup/mtup_full_training_final
```

## 🎯 Tomorrow: Train Baseline

### Baseline Command
```bash
# Train baseline with same 7B model
python3 src/train.py \
  --config config/config.py \
  --num_epochs 10 \
  --output_dir outputs/models/baseline_single_task_7b
```

**Note**: Check if `src/train.py` exists. If not, we may need to create it or use different script.

## 🐛 Common Issues

### Issue: OOM (Out of Memory)
```bash
# Solution: Reduce batch size
python3 train_mtup.py \
  --use-case full_training \
  --model qwen2.5-7b \
  --batch-size 2 \        # Reduce from 4
  --grad-accum 8          # Increase to keep effective batch=16
```

### Issue: Model download slow
```bash
# Use HuggingFace mirror (if in Asia)
export HF_ENDPOINT=https://hf-mirror.com
python3 train_mtup.py --use-case full_training --model qwen2.5-7b
```

### Issue: Training interrupted
```bash
# Resume from checkpoint (if auto-save enabled)
python3 train_mtup.py \
  --use-case full_training \
  --model qwen2.5-7b \
  --resume-from outputs/checkpoints_mtup/checkpoint-500
```

## 📋 Quick Reference

### Today (MTUP)
```bash
python3 train_mtup.py --use-case full_training --model qwen2.5-7b --epochs 10
```

### After Training
```bash
# Push to HuggingFace
python3 push_to_hf_simple.py --model-type mtup
```

### Tomorrow (Baseline)
```bash
# TBD - need to check baseline training script
```

---

**Current Status**: Ready to train MTUP! 🚀
**Command**: `python3 train_mtup.py --use-case full_training --model qwen2.5-7b --epochs 10`
**Expected Time**: 4-6 hours
