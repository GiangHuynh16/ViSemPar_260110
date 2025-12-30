# Environment Setup Guide

**Project**: Vietnamese AMR Parser (ViSemPar)
**Updated**: 2025-12-30

---

## System Requirements

### Hardware
- **GPU**: NVIDIA GPU with >= 20GB VRAM (tested on Quadro RTX 6000 24GB)
- **RAM**: >= 32GB
- **Storage**: >= 50GB free space (for models and checkpoints)

### Software
- **OS**: Linux (Ubuntu 20.04+ recommended)
- **Python**: 3.10 or 3.11
- **CUDA**: 12.1+ (or 11.8+)
- **Git**: For version control

---

## Quick Setup (Recommended)

### Option 1: Conda Environment

```bash
# Create conda environment
conda create -n viamr python=3.10 -y
conda activate viamr

# Install PyTorch with CUDA 12.1
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Install other packages
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

### Option 2: venv + pip

```bash
# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Install PyTorch with CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other packages
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

---

## Package Versions

### Core Dependencies

```
Python: 3.10+
PyTorch: 2.3.0+
Transformers: 4.46.3
PEFT: 0.13.2
Accelerate: 1.2.1
```

### Important Notes

**⚠️ bitsandbytes NOT included**:
- Reason: Triton compatibility issues on server
- Solution: Use `adamw_torch` optimizer instead of `adamw_8bit`
- Config: `USE_4BIT_QUANTIZATION = False`

**⚠️ NumPy version locked to 1.x**:
- Reason: NumPy 2.x breaks pandas/sklearn
- Version: `numpy>=1.24.0,<2.0.0`

**⚠️ HuggingFace Hub version**:
- Constraint: `<1.0` (required by transformers 4.46.3)
- Version: `huggingface-hub>=0.24.0,<1.0`

---

## Installation Steps (Detailed)

### Step 1: System Preparation

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install system dependencies
sudo apt install -y git build-essential python3.10 python3.10-venv python3-pip

# Verify CUDA installation
nvidia-smi
nvcc --version
```

### Step 2: Clone Repository

```bash
# Clone project
git clone https://github.com/your-username/ViSemPar_new1.git
cd ViSemPar_new1

# Check current branch
git branch
```

### Step 3: Create Environment

**Using Conda** (Recommended):
```bash
# Create environment
conda create -n viamr python=3.10 -y
conda activate viamr

# Install PyTorch
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

**Using venv**:
```bash
# Create environment
python3.10 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

### Step 4: Install PyTorch

**For CUDA 12.1**:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**For CUDA 11.8**:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Step 5: Install Dependencies

```bash
# Install from requirements.txt
pip install -r requirements.txt

# Verify critical packages
python -c "from transformers import AutoModelForCausalLM; print('✓ Transformers OK')"
python -c "from peft import LoraConfig; print('✓ PEFT OK')"
python -c "import penman; print('✓ Penman OK')"
python -c "import smatch; print('✓ SMATCH OK')"
```

### Step 6: Setup Directories

```bash
# Create necessary directories
mkdir -p data outputs logs outputs/checkpoints results

# Verify structure
ls -la
```

### Step 7: HuggingFace Login (Optional)

```bash
# Login to HuggingFace (for model download and upload)
huggingface-cli login

# Paste your token from: https://huggingface.co/settings/tokens
# Choose: Write permission (for model upload)
```

---

## Verification Checklist

Run these commands to verify setup:

```bash
# 1. Python version
python --version
# Expected: Python 3.10.x or 3.11.x

# 2. PyTorch and CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
# Expected: PyTorch 2.3.0+, CUDA available: True, CUDA version: 12.1

# 3. Transformers
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
# Expected: 4.46.3

# 4. PEFT
python -c "import peft; print(f'PEFT: {peft.__version__}')"
# Expected: 0.13.2

# 5. GPU memory
nvidia-smi --query-gpu=memory.total,memory.free --format=csv
# Expected: >= 20GB free

# 6. Project structure
ls -la data/ outputs/ config/
# Expected: Directories exist

# 7. HuggingFace login
huggingface-cli whoami
# Expected: Your username (if logged in)
```

---

## Common Issues

### Issue 1: CUDA not available

**Symptoms**:
```python
>>> torch.cuda.is_available()
False
```

**Solutions**:
1. Check NVIDIA driver: `nvidia-smi`
2. Reinstall PyTorch with correct CUDA version
3. Verify CUDA installation: `nvcc --version`

### Issue 2: Out of Memory (OOM)

**Symptoms**:
```
RuntimeError: CUDA out of memory
```

**Solutions**:
1. Reduce batch size: `--batch-size 1`
2. Increase gradient accumulation: `--grad-accum 16`
3. Check other GPU processes: `nvidia-smi`
4. Kill unused processes: `pkill -f python`

### Issue 3: bitsandbytes import error

**Symptoms**:
```
ModuleNotFoundError: No module named 'triton.ops'
```

**Solution**:
- This is expected! We don't use bitsandbytes
- Config already set: `USE_4BIT_QUANTIZATION = False`
- Optimizer already set: `adamw_torch`

### Issue 4: NumPy version conflict

**Symptoms**:
```
ValueError: numpy.dtype size changed
```

**Solution**:
```bash
pip install "numpy>=1.24.0,<2.0.0" --force-reinstall
```

### Issue 5: HuggingFace token invalid

**Symptoms**:
```
401 Client Error: Unauthorized
```

**Solution**:
```bash
# Re-login with new token
huggingface-cli login

# Get token from: https://huggingface.co/settings/tokens
# Make sure to select "Write" permission
```

---

## Environment Variables

Create `.env` file (optional):

```bash
# .env file
CUDA_VISIBLE_DEVICES=0
TOKENIZERS_PARALLELISM=false
HF_HOME=/path/to/huggingface/cache
TRANSFORMERS_CACHE=/path/to/transformers/cache
```

---

## Conda Environment Export/Import

### Export environment:

```bash
# Export to YAML
conda env export > environment_export.yml

# Export to requirements
pip list --format=freeze > requirements_freeze.txt
```

### Import environment:

```bash
# From YAML
conda env create -f environment.yml

# From requirements
pip install -r requirements.txt
```

---

## Testing Installation

### Quick Test

```bash
# Test imports
python -c "
import torch
import transformers
from peft import LoraConfig
import penman
import smatch
print('✅ All imports successful!')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
print(f'Transformers: {transformers.__version__}')
"
```

### Full Test (Load Model)

```bash
# Test model loading
python -c "
import torch
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct')
print(f'✅ Tokenizer loaded: {len(tokenizer)} tokens')
"
```

---

## GPU Memory Estimation

### Baseline 7B Training:

```
Base model: ~14GB (FP16)
LoRA adapters: ~0.5GB
Activations: ~2-3GB
Gradients: ~2-3GB
Optimizer states: ~1-2GB
Buffer: ~2GB
---
Total: ~20-22GB
```

**Recommendation**: >= 24GB VRAM for safe training

---

## Cleanup Commands

### Clear cache:

```bash
# PyTorch cache
rm -rf ~/.cache/torch

# HuggingFace cache
rm -rf ~/.cache/huggingface

# Python cache
find . -type d -name __pycache__ -exec rm -r {} +
find . -type f -name "*.pyc" -delete
```

### Remove environment:

```bash
# Conda
conda env remove -n viamr

# venv
rm -rf venv/
```

---

## Summary

**Recommended Setup**:
```bash
# 1. Create conda env
conda create -n viamr python=3.10 -y
conda activate viamr

# 2. Install PyTorch
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# 5. Login to HF
huggingface-cli login

# 6. Ready to train!
bash START_BASELINE_7B_TRAINING.sh
```

**Key Points**:
- ✅ Python 3.10
- ✅ PyTorch 2.3.0+ with CUDA 12.1
- ✅ NO bitsandbytes
- ✅ adamw_torch optimizer
- ✅ >= 24GB VRAM recommended
