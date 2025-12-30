# Setup Training on New Server (islab-server2)

**Server**: `islabworker2@islab-server2:/mnt/nghiepth/giangha`
**Date**: 2025-12-30

---

## Step 1: SSH vào Server Mới

```bash
ssh islabworker2@islab-server2
```

Nhập password khi được hỏi.

---

## Step 2: Tạo Directory và Set Permissions

```bash
# Navigate to target directory
cd /mnt/nghiepth/giangha

# Check if you have write permission
touch test_permission.txt
rm test_permission.txt

# If permission denied, you need to request access from admin
# Otherwise, create project directory
mkdir -p ViSemPar
cd ViSemPar
```

### Nếu Bị Permission Denied:

Option A - Yêu cầu admin cấp quyền:
```bash
# Contact admin to run:
# sudo chown -R islabworker2:islabworker2 /mnt/nghiepth/giangha
# sudo chmod -R 755 /mnt/nghiepth/giangha
```

Option B - Dùng home directory thay thế:
```bash
cd ~
mkdir -p ViSemPar
cd ViSemPar
```

---

## Step 3: Setup Git SSH Key (Nếu Chưa Có)

### 3.1: Check existing SSH key
```bash
ls -la ~/.ssh/id_*.pub
```

### 3.2: Nếu chưa có, tạo SSH key mới
```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
# Press Enter 3 times (default location, no passphrase)

# Copy public key
cat ~/.ssh/id_ed25519.pub
```

### 3.3: Add SSH key to GitHub
1. Copy nội dung từ `cat ~/.ssh/id_ed25519.pub`
2. Vào GitHub.com → Settings → SSH and GPG keys → New SSH key
3. Paste key và save

### 3.4: Test SSH connection
```bash
ssh -T git@github.com
# Should see: "Hi USERNAME! You've successfully authenticated..."
```

---

## Step 4: Clone Repository

```bash
# Navigate to working directory
cd /mnt/nghiepth/giangha/ViSemPar  # or ~/ViSemPar if using home

# Clone via SSH
git clone git@github.com:GiangHuynh16/ViSemPar_new1.git

# Navigate into repo
cd ViSemPar_new1

# Verify files
ls -la
```

---

## Step 5: Check GPU Availability

```bash
# Check GPU info
nvidia-smi

# Expected output should show:
# - GPU model (e.g., RTX 3090, A100, etc.)
# - Total VRAM (ideally >= 40GB for comfortable training)
# - Current usage (should be low/free)
```

**Expected for strong server**:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.xx       Driver Version: 535.xx       CUDA Version: 12.x    |
|-------------------------------+----------------------+----------------------+
|   0  Tesla V100-SXM2    On   | 00000000:xx:xx.x Off |                    0 |
| N/A   35C    P0    40W / 300W |      0MiB / 32510MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
```

---

## Step 6: Setup Python Environment

### Option A: Check existing conda environments

```bash
# List existing environments
conda env list

# Check if lora_py310 exists
conda activate lora_py310

# Verify packages
python --version
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import peft; print(f'PEFT: {peft.__version__}')"

# If all packages are correct, you're done!
# Otherwise, continue to Option B
```

### Option B: Create new environment

```bash
# Create new conda environment
conda create -n baseline_7b python=3.10 -y

# Activate environment
conda activate baseline_7b

# Install PyTorch with CUDA support (check CUDA version first)
nvidia-smi  # Check CUDA version

# For CUDA 11.8
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# Install required packages
pip install transformers==4.46.3
pip install peft==0.13.2
pip install accelerate==0.34.2
pip install datasets==2.14.5
pip install smatch==1.0.4
pip install penman==1.3.0
pip install tqdm

# Verify installation
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA devices: {torch.cuda.device_count()}')"
```

---

## Step 7: Verify Code Configuration

```bash
cd /mnt/nghiepth/giangha/ViSemPar/ViSemPar_new1

# Run verification script
bash VERIFY_AND_START.sh
```

This will check:
- ✅ MAX_SEQ_LENGTH = 512
- ✅ batch_size = 1
- ✅ device_map = None
- ✅ GPU availability

**If verification passes**, you're ready to train!

---

## Step 8: Adjust Config for Stronger Server (Optional)

Nếu server mới có VRAM >= 40GB, bạn có thể tăng config để match MTUP:

```bash
# Edit config
nano config/config.py
```

### Recommended adjustments based on VRAM:

#### If VRAM >= 40GB (e.g., A100):
```python
MAX_SEQ_LENGTH = 2048  # Same as MTUP
batch_size = 2         # Same as MTUP
gradient_accumulation = 8  # Same as MTUP
```

#### If VRAM = 32GB (e.g., V100):
```python
MAX_SEQ_LENGTH = 1024  # Half of MTUP
batch_size = 1
gradient_accumulation = 16
```

#### If VRAM = 24GB (e.g., RTX 3090):
```python
MAX_SEQ_LENGTH = 512   # Current config
batch_size = 1
gradient_accumulation = 16
```

**IMPORTANT**: After editing, save and commit:
```bash
git add config/config.py
git commit -m "Adjust config for server GPU capacity"
```

---

## Step 9: Create Required Directories

```bash
cd /mnt/nghiepth/giangha/ViSemPar/ViSemPar_new1

# Create directories
mkdir -p outputs/checkpoints
mkdir -p logs
mkdir -p results

# Verify data directory exists
ls -la data/
# Should see: train_amr_1.txt, train_amr_2.txt, public_test.txt, etc.
```

---

## Step 10: Start Training in tmux

```bash
# Create new tmux session
tmux new -s baseline_7b

# Inside tmux, activate environment
conda activate baseline_7b  # or lora_py310

# Navigate to project
cd /mnt/nghiepth/giangha/ViSemPar/ViSemPar_new1

# Start training
bash START_BASELINE_7B_TRAINING.sh

# Or use verification script
bash VERIFY_AND_START.sh
```

### Tmux Commands:
- **Detach from tmux**: `Ctrl+B` then `D`
- **Reattach to session**: `tmux attach -t baseline_7b`
- **List sessions**: `tmux ls`
- **Kill session**: `tmux kill-session -t baseline_7b`

---

## Step 11: Monitor Training

### In another SSH session:

```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# Check training logs
tail -f logs/training_baseline*.log

# Check disk space
df -h /mnt/nghiepth/giangha
```

### Expected GPU usage:

For 24GB VRAM:
```
|   0  GPU              |  ~20000MiB / 24576MiB |     90%      |
```

For 40GB+ VRAM:
```
|   0  GPU              |  ~25000MiB / 40960MiB |     90%      |
```

---

## Troubleshooting

### Issue 1: Permission Denied in `/mnt/nghiepth/giangha`

**Solution A**: Request admin access
```bash
# Admin needs to run:
sudo chown -R islabworker2:islabworker2 /mnt/nghiepth/giangha
sudo chmod -R 755 /mnt/nghiepth/giangha
```

**Solution B**: Use home directory
```bash
cd ~
mkdir -p ViSemPar
cd ViSemPar
git clone git@github.com:GiangHuynh16/ViSemPar_new1.git
```

### Issue 2: SSH Key Already Exists on GitHub

```bash
# Generate new key with different name
ssh-keygen -t ed25519 -C "your_email@example.com" -f ~/.ssh/id_ed25519_server2

# Add to SSH config
echo "
Host github.com
  IdentityFile ~/.ssh/id_ed25519_server2
" >> ~/.ssh/config

# Test
ssh -T git@github.com
```

### Issue 3: Conda Not Found

```bash
# Check if conda is installed
which conda

# If not found, check common locations
ls /opt/conda/bin/conda
ls ~/anaconda3/bin/conda
ls ~/miniconda3/bin/conda

# Add to PATH if found
export PATH="/opt/conda/bin:$PATH"
# Or add to ~/.bashrc for permanent
echo 'export PATH="/opt/conda/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

### Issue 4: Still OOM on New Server

If training still fails with OOM even on stronger server:

```bash
# Check actual VRAM
nvidia-smi --query-gpu=memory.total --format=csv,noheader

# Adjust config based on VRAM
# See Step 8 above
```

---

## Expected Timeline

| VRAM | Config | Training Time |
|------|--------|---------------|
| 24GB | seq=512, batch=1 | ~15-18 hours |
| 32GB | seq=1024, batch=1 | ~12-15 hours |
| 40GB+ | seq=2048, batch=2 | ~10-12 hours (same as MTUP) |

---

## After Training Completes

```bash
# Evaluate model
python evaluate_baseline_model.py \
  --checkpoint outputs/checkpoints/baseline_7b_final \
  --test-file data/public_test_ground_truth.txt \
  --output results/baseline_7b_evaluation.json

# Check F1 score
cat results/baseline_7b_evaluation.json

# Compare with MTUP
echo "MTUP 7B F1: [from MTUP evaluation]"
echo "Baseline 7B F1: [from results]"
```

---

## Quick Command Reference

```bash
# SSH to server
ssh islabworker2@islab-server2

# Navigate to project
cd /mnt/nghiepth/giangha/ViSemPar/ViSemPar_new1

# Activate environment
conda activate baseline_7b

# Start training
tmux new -s baseline_7b
bash VERIFY_AND_START.sh

# Detach from tmux
Ctrl+B, then D

# Monitor GPU
watch -n 1 nvidia-smi

# Check logs
tail -f logs/training_baseline*.log

# Reattach to tmux
tmux attach -t baseline_7b
```

---

## Summary Checklist

Before starting training, ensure:

- [ ] SSH access to server works
- [ ] Directory created with write permission
- [ ] Git SSH key configured
- [ ] Repository cloned
- [ ] GPU available (nvidia-smi works)
- [ ] Python environment set up
- [ ] Required packages installed (torch, transformers, peft)
- [ ] Config verified (MAX_SEQ_LENGTH, batch_size, device_map)
- [ ] Required directories created (outputs, logs, results)
- [ ] Data files present in data/
- [ ] Running in tmux session

**All checked?** → `bash VERIFY_AND_START.sh` → Training begins! 🚀
