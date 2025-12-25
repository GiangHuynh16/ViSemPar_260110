# 🧹 Model Cleanup Guide

## Overview

Clean up old/unused models to free disk space while keeping the important ones.

---

## 📊 Current Model Status

| Model | Size | Status | Action |
|-------|------|--------|--------|
| **3B MTUP** | ~457 MB | ✅ Active | **KEEP** - Primary model (F1=0.49) |
| **7B** | ~14 GB | 🟡 Backup | **KEEP** - Good balance, future use |
| **14B** | ~28 GB | ❌ Unused | **DELETE** - Causes OOM, not used |

---

## 🎯 Recommendation

**Delete 14B model** because:
- ❌ Causes OOM on 24GB GPU
- ❌ Slower than 3B for inference
- ❌ 3B MTUP performs well enough (F1=0.49)
- ❌ Not using it anymore

**Keep 7B model** because:
- ✅ Smaller than 14B (~14GB vs 28GB)
- ✅ May be useful for future experiments
- ✅ Good balance of size/performance
- ✅ Fits comfortably in 24GB GPU

**Keep 3B MTUP** because:
- ✅ Primary working model
- ✅ Tested and proven (F1=0.49)
- ✅ Fast inference
- ✅ Uses Vietnamese prompts
- ✅ LoRA adapter (very small)

---

## 🚀 Quick Start

### Step 1: Check Disk Usage

```bash
bash CHECK_DISK_USAGE.sh
```

Shows:
- Size of each model
- Total disk usage
- Recommendations

### Step 2: Clean Up (Delete 14B)

```bash
bash CLEANUP_MODELS.sh
```

This will:
1. Show what will be deleted (14B)
2. Show what will be kept (7B, 3B)
3. Ask for confirmation
4. Delete 14B model
5. Create deletion log

**Expected space freed**: ~28 GB

---

## 📁 Model Locations

```
outputs/
├── checkpoints/                    ← Large base models
│   ├── qwen2.5-14b-fine-tuned/    (~28 GB) ← DELETE
│   └── qwen2.5-7b-fine-tuned/     (~14 GB) ← KEEP
│
└── checkpoints_mtup/               ← MTUP LoRA adapters
    └── mtup_full_training_final/  (~457 MB) ← KEEP (PRIMARY!)
```

---

## ⚠️ Safety Features

### Confirmation Required

Script asks for confirmation before deleting:
```
Type 'yes' to confirm deletion: yes
```

Must type exactly `yes` (not `y` or `YES`).

### Deletion Log

Creates `outputs/deleted_14b_model.log` with:
- List of deleted files
- Deletion timestamp
- Can reference if needed later

### No Auto-Delete

Script will NOT delete without your explicit confirmation.

---

## 🔍 Manual Cleanup (Advanced)

If you prefer manual control:

### Check what exists:

```bash
ls -lh outputs/checkpoints/
ls -lh outputs/checkpoints_mtup/
```

### Check sizes:

```bash
du -sh outputs/checkpoints/*
du -sh outputs/checkpoints_mtup/*
```

### Delete 14B manually:

```bash
# Backup file list first
ls -lR outputs/checkpoints/qwen2.5-14b-fine-tuned/ > deleted_14b.log

# Delete
rm -rf outputs/checkpoints/qwen2.5-14b-fine-tuned/

# Verify
ls outputs/checkpoints/
```

---

## 📊 Expected Results

### Before Cleanup:

```
outputs/checkpoints/qwen2.5-14b-fine-tuned/  28 GB
outputs/checkpoints/qwen2.5-7b-fine-tuned/   14 GB
outputs/checkpoints_mtup/                    457 MB

Total: ~42 GB
```

### After Cleanup:

```
outputs/checkpoints/qwen2.5-7b-fine-tuned/   14 GB
outputs/checkpoints_mtup/                    457 MB

Total: ~14.5 GB

Space freed: ~28 GB ✅
```

---

## 🛡️ What If I Need 14B Later?

You can always:

1. **Re-download** from Hugging Face (if you uploaded it)
2. **Re-train** using your training scripts
3. **Use 7B instead** - similar quality, less memory
4. **Use 3B MTUP** - proven to work well

But realistically, you won't need it because:
- 3B MTUP works great for your use case
- 14B doesn't fit in 24GB GPU for inference
- Training 14B is very slow

---

## 🎯 Post-Cleanup Checklist

After running cleanup:

- [ ] 14B model deleted
- [ ] 7B model still present
- [ ] 3B MTUP model still present
- [ ] Deletion log created
- [ ] Disk space freed (~28 GB)
- [ ] Run evaluation to confirm 3B MTUP still works:
  ```bash
  bash RUN_FULL_EVALUATION_TMUX.sh
  ```

---

## 📖 Related Documentation

- [FIX_14B_OOM.md](FIX_14B_OOM.md) - Why 14B causes OOM
- [EVALUATION_SUMMARY.md](EVALUATION_SUMMARY.md) - 3B MTUP results
- [README.md](README.md) - Project overview

---

## 💡 Tips

### Keep Your Workspace Clean

```bash
# Regular cleanup
bash CHECK_DISK_USAGE.sh  # Weekly check

# Remove old logs
find outputs/ -name "*.log" -mtime +30 -delete  # Logs older than 30 days

# Remove old evaluation results
find outputs/ -name "evaluation_*" -mtime +30  # Review before deleting
```

### Organize Models

```bash
# Move unused models to archive (instead of deleting)
mkdir -p archive/models
mv outputs/checkpoints/qwen2.5-14b-fine-tuned/ archive/models/

# Compress if needed
tar -czf archive/14b_model.tar.gz archive/models/qwen2.5-14b-fine-tuned/
rm -rf archive/models/qwen2.5-14b-fine-tuned/
```

---

## ✅ Recommended Cleanup Schedule

| When | Action | Command |
|------|--------|---------|
| Now | Delete 14B | `bash CLEANUP_MODELS.sh` |
| Weekly | Check disk | `bash CHECK_DISK_USAGE.sh` |
| Monthly | Clean logs | Review old logs in `outputs/` |
| As needed | Remove old results | Keep only recent evaluations |

---

_This guide helps maintain a clean and efficient workspace_
