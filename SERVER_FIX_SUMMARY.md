# Server Dependency Fix Summary

## Issues Found on Server

### ✅ Issue 1: Missing Checkpoints Directory
**Status:** FIXED (commit a763767)

**Error:**
```
⚠️ Checkpoints directory missing: /mnt/nghiepth/giang/ViSemPar/outputs/checkpoints_mtup
❌ Environment check failed
```

**Fix:** Updated `train_mtup.py` to auto-create missing directories

---

### ✅ Issue 2: NumPy 2.x Incompatibility
**Status:** FIXED (commit 738a585)

**Error:**
```
AttributeError: _ARRAY_API not found
ValueError: numpy.dtype size changed, may indicate binary incompatibility
```

**Root Cause:** Server has NumPy 2.4.0, but pandas/sklearn compiled with NumPy 1.x

**Fix:**
- Pin `numpy>=1.24.0,<2.0.0` in requirements.txt
- Created automated fix script: `fix_numpy_server.sh`

---

### ✅ Issue 3: HuggingFace Hub Version Conflict
**Status:** FIXED (commit 738a585)

**Error:**
```
ImportError: huggingface-hub>=0.24.0,<1.0 is required for a normal functioning of this module,
but found huggingface-hub==1.2.3
```

**Root Cause:** Server has huggingface-hub 1.2.3, but transformers 4.46.3 requires <1.0

**Fix:**
- Pin `huggingface-hub>=0.24.0,<1.0` in requirements.txt
- Updated fix script to downgrade hub version

---

## What to Do on Server

### Step 1: Pull Latest Code

```bash
cd ~/ViSemPar_new1
git pull origin main
```

**Expected output:**
```
From https://github.com/GiangHuynh16/ViSemPar_new1
   a763767..c1527bc  main       -> origin/main
Updating a763767..c1527bc
```

### Step 2: Run Automated Fix

```bash
bash fix_numpy_server.sh
```

**This will:**
1. ✅ Downgrade NumPy to 1.x
2. ✅ Downgrade huggingface-hub to <1.0
3. ✅ Reinstall pandas and scikit-learn
4. ✅ Verify all installations

**Expected output:**
```
✓ NumPy 1.26.4
✓ pandas OK
✓ sklearn OK
✓ huggingface-hub 0.26.2
```

### Step 3: Verify Training Works

```bash
python3 train_mtup.py --use-case quick_test --show-sample
```

**Expected output:**
```
✓ Checkpoints directory: /mnt/nghiepth/giang/ViSemPar/outputs/checkpoints_mtup
✓ Training file found: train_amr_1.txt
✓ Training file found: train_amr_2.txt
✅ Environment check passed

Loading MTUP data...
✓ Total raw examples: 1842
✓ Limited to 100 samples
✓ Processed 100/100 examples

SAMPLE MTUP EXAMPLE
...
Training will start...
```

---

## Files Modified

1. **train_mtup.py** - Auto-create directories
2. **requirements.txt** - Pin NumPy <2.0, huggingface-hub <1.0
3. **fix_numpy_server.sh** - Automated fix script (NEW)
4. **NUMPY_FIX_GUIDE.md** - Detailed troubleshooting (NEW)
5. **DEPLOYMENT_GUIDE.md** - Added NumPy fix section

---

## Git Commits

```
c1527bc - Update NUMPY_FIX_GUIDE with huggingface-hub fix
738a585 - Fix: Add huggingface-hub version constraint
683880d - Update DEPLOYMENT_GUIDE with NumPy fix section (no secrets)
efd9995 - Fix: Add NumPy 1.x compatibility fix for server
a763767 - Fix: Auto-create missing directories in training script
```

---

## Quick Commands

```bash
# Pull and fix
cd ~/ViSemPar_new1
git pull origin main
bash fix_numpy_server.sh

# Test training
python3 train_mtup.py --use-case quick_test

# Full training (in tmux)
tmux new -s amr-training
python3 train_mtup.py --use-case full_training
# Ctrl+B, D to detach
```

---

## Manual Fix (Alternative)

If the automated script fails:

```bash
# Fix NumPy
pip install "numpy<2.0.0" --upgrade

# Fix HuggingFace Hub
pip install "huggingface-hub>=0.24.0,<1.0" --force-reinstall

# Reinstall pandas/sklearn
pip install --force-reinstall pandas scikit-learn

# Verify
python3 -c "import numpy, pandas, sklearn, huggingface_hub; print('All OK')"
```

---

## Status: READY FOR TRAINING

All issues have been identified and fixed. The code is ready to pull and run on the server.
