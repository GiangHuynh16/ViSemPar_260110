# NumPy Compatibility Fix for Server

## Problems

### Problem 1: NumPy Version Incompatibility

The server has **NumPy 2.x**, but dependencies like `pandas`, `scikit-learn`, `bottleneck`, and `numexpr` were compiled with **NumPy 1.x**, causing:

```
AttributeError: _ARRAY_API not found
ValueError: numpy.dtype size changed, may indicate binary incompatibility
```

### Problem 2: HuggingFace Hub Version Conflict

The server has **huggingface-hub 1.2.3**, but `transformers 4.46.3` requires **<1.0**, causing:

```
ImportError: huggingface-hub>=0.24.0,<1.0 is required for a normal functioning of this module, but found huggingface-hub==1.2.3.
```

## Quick Fix (Method 1 - Recommended)

Run the automated fix script:

```bash
cd ~/ViSemPar_new1
bash fix_numpy_server.sh
```

This will:
1. Downgrade NumPy to 1.x
2. Reinstall pandas and scikit-learn for compatibility
3. Downgrade huggingface-hub to <1.0
4. Verify all fixes

## Manual Fix (Method 2)

If the script doesn't work, fix manually:

```bash
# Fix NumPy
pip install "numpy<2.0.0" --upgrade

# Fix huggingface-hub
pip install "huggingface-hub>=0.24.0,<1.0" --force-reinstall

# Reinstall dependencies
pip install --force-reinstall pandas scikit-learn

# Verify
python3 -c "import numpy; print(f'NumPy {numpy.__version__}')"
python3 -c "import pandas; print('pandas OK')"
python3 -c "import sklearn; print('sklearn OK')"
python3 -c "import huggingface_hub; print(f'HF Hub {huggingface_hub.__version__}')"
```

## Expected Result

After the fix:

```
✓ NumPy 1.26.4 (or similar 1.x version)
✓ pandas OK
✓ sklearn OK
✓ huggingface-hub 0.26.x (or similar <1.0 version)
```

## Verify Training Works

```bash
python3 train_mtup.py --use-case quick_test --show-sample
```

You should see:
```
✓ Created: /mnt/nghiepth/giang/ViSemPar/outputs/checkpoints_mtup
✓ Training file found: train_amr_1.txt
✓ Training file found: train_amr_2.txt
✅ Environment check passed

Loading MTUP data...
```

## Why This Happens

- **NumPy 2.x** (released 2024) changed internal C API
- Packages compiled with NumPy 1.x are **not compatible** with NumPy 2.x
- Solution: Use NumPy 1.x until all dependencies update

## Long-term Solution

Wait for these packages to release NumPy 2.x compatible versions:
- pandas (working on 2.x support)
- scikit-learn (working on 2.x support)
- bottleneck, numexpr (pending updates)

For now, **NumPy 1.x is the stable choice** for ML/NLP projects.

---

## Alternative: Virtual Environment

If you want to avoid affecting other projects:

```bash
# Create new environment
conda create -n amr-mtup python=3.10 -y
conda activate amr-mtup

# Install compatible versions
pip install -r requirements.txt

# Run training
python3 train_mtup.py --use-case quick_test
```

This isolates the fix to this project only.
