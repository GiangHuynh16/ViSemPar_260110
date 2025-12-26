# 🔧 Fix Git Conflict on Server

## ⚠️ Vấn Đề

Khi pull code từ GitHub về server, gặp conflict:

```
error: Your local changes to the following files would be overwritten by merge:
        config/config_mtup.py
Please commit your changes or stash them before you merge.
```

## ✅ Giải Pháp

### Option 1: Stash Changes (Recommended)

Cách này **giữ lại changes** của bạn trên server, nhưng tạm thời "cất đi" để pull code mới:

```bash
# 1. Lưu changes hiện tại
git stash save "My local changes on server"

# 2. Pull code mới
git pull origin main

# 3. Xem changes của bạn đã stash
git stash show -p

# 4. Quyết định:
# - Nếu changes của bạn quan trọng: git stash pop
# - Nếu không cần nữa: git stash drop
```

### Option 2: Overwrite Local Changes

Cách này **XÓA changes** của bạn trên server, dùng code mới từ GitHub:

```bash
# ⚠️ WARNING: This will DELETE your local changes!

# 1. Discard local changes
git checkout -- config/config_mtup.py

# 2. Pull code mới
git pull origin main
```

### Option 3: Commit Local Changes First

Cách này **commit changes** của bạn trước, sau đó merge với code mới:

```bash
# 1. Commit changes của bạn
git add config/config_mtup.py
git commit -m "Local changes on server"

# 2. Pull và merge
git pull origin main

# 3. Nếu có conflict, resolve manually
# Edit config/config_mtup.py to fix conflicts
git add config/config_mtup.py
git commit -m "Merge with remote changes"
```

## 🎯 Khuyến Nghị

**Dùng Option 1 (Stash)** vì:
- ✅ An toàn (không mất changes)
- ✅ Có thể review changes sau
- ✅ Dễ quay lại nếu cần

## 📋 Chi Tiết Các Bước

### Bước 1: Check What Changed

```bash
# Xem bạn đã thay đổi gì
git diff config/config_mtup.py
```

**Likely changes**:
- Model name (3B → 7B)
- LoRA config
- Training parameters

### Bước 2: Stash Changes

```bash
cd ~/ViSemPar_new1

# Save your changes
git stash save "Server config changes before pull"

# Verify stash
git stash list
# Should show: stash@{0}: On main: Server config changes before pull
```

### Bước 3: Pull New Code

```bash
# Now pull will work
git pull origin main

# Verify
git log --oneline -5
# Should show latest commits including unified pipeline changes
```

### Bước 4: Review Changes

```bash
# See what you had stashed
git stash show -p stash@{0}

# Compare with new code
cat config/config_mtup.py | grep MODEL_NAME
# Should show: MODEL_NAME = MODELS['qwen2.5-7b']
```

### Bước 5: Decide What to Keep

**If your stashed changes are important**:
```bash
# Apply your changes on top
git stash pop

# If conflict, resolve manually
# Then:
git add config/config_mtup.py
git commit -m "Merge server changes with new config"
```

**If you want to use the new code** (recommended):
```bash
# Just drop your stashed changes
git stash drop

# Verify you have latest
python3 -c "
import sys
sys.path.insert(0, 'config')
from config_mtup import MODEL_NAME
print(f'Model: {MODEL_NAME}')
"
# Should print: Model: Qwen/Qwen2.5-7B-Instruct
```

## 🔍 What Changed in New Code

The unified pipeline changes:

### config/config_mtup.py
```python
# OLD (your server version - likely 3B):
MODEL_NAME = MODELS['qwen2.5-3b']

# NEW (from GitHub - 7B):
MODEL_NAME = MODELS['qwen2.5-7b']
```

### config/prompt_templates.py
```python
# NEW: Cleaner formatting
MTUP_TEMPLATE_V2_NATURAL = """### NHIỆM VỤ
Chuyển đổi câu tiếng Việt sang AMR (2 bước)

### CÂU ĐẦU VÀO
{sentence}
...
```

### evaluate_mtup_model.py
```python
# NEW: No post-processing
# OLD had: final_amr = post_process_amr_conservative(final_amr)
# NEW: Removed that line
```

## ✅ Quick Fix (One Command)

If you just want to **use the new code** and don't care about local changes:

```bash
cd ~/ViSemPar_new1
git reset --hard origin/main
```

⚠️ **Warning**: This **DELETES ALL local changes**!

## 🎯 Recommended Solution

```bash
# 1. Stash (save but set aside)
git stash

# 2. Pull new code
git pull origin main

# 3. Drop stash (don't need old changes)
git stash drop

# 4. Verify
python3 -c "
import sys
sys.path.insert(0, 'config')
from config_mtup import MODEL_NAME
print(f'✅ Model: {MODEL_NAME}')
"
```

## 📞 After Fixing

Once conflict is resolved:

```bash
# Verify you have all new files
ls -la *.md | grep -E "PIPELINE|TRAINING|READY"

# Should see:
# PIPELINE_SUMMARY.md
# PIPELINE_UNIFIED.md
# TRAINING_GUIDE_UNIFIED.md
# READY_FOR_TRAINING.md
# etc.

# Now you can train
python3 train_mtup.py --use-case best_accuracy --epochs 10
```

## 🐛 If Still Have Issues

```bash
# Nuclear option: Delete and re-clone
cd ~
mv ViSemPar_new1 ViSemPar_new1.backup
git clone https://github.com/GiangHuynh16/ViSemPar_new1.git
cd ViSemPar_new1

# Copy over any important local files from backup
cp ~/ViSemPar_new1.backup/outputs/*.pth outputs/ 2>/dev/null || true
```

---

**TL;DR**: Run this on server:
```bash
cd ~/ViSemPar_new1
git stash
git pull origin main
git stash drop
```
