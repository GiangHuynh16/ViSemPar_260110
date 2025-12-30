#!/bin/bash
# Quick Setup Script for New Server (islab-server2)
# Run this after SSH-ing into the new server

set -e

echo "=========================================="
echo "BASELINE 7B - NEW SERVER SETUP"
echo "=========================================="
echo ""
echo "Server: islabworker2@islab-server2"
echo "Target: /mnt/nghiepth/giangha/ViSemPar"
echo ""

# Step 1: Check current location
echo "Step 1: Checking current directory..."
echo "  Current: $(pwd)"
echo ""

# Step 2: Navigate or create directory
echo "Step 2: Setting up project directory..."
TARGET_DIR="/mnt/nghiepth/giangha/ViSemPar"

if [ -w "/mnt/nghiepth/giangha" ]; then
    echo "  ✓ Have write permission to /mnt/nghiepth/giangha"
    mkdir -p "$TARGET_DIR"
    cd "$TARGET_DIR"
    echo "  ✓ Created and navigated to: $TARGET_DIR"
else
    echo "  ⚠️  No write permission to /mnt/nghiepth/giangha"
    echo "  Using home directory instead..."
    TARGET_DIR="$HOME/ViSemPar"
    mkdir -p "$TARGET_DIR"
    cd "$TARGET_DIR"
    echo "  ✓ Created and navigated to: $TARGET_DIR"
fi
echo ""

# Step 3: Check if repo already exists
echo "Step 3: Checking repository..."
if [ -d "ViSemPar_new1" ]; then
    echo "  ✓ Repository already exists"
    cd ViSemPar_new1
    echo "  Pulling latest changes..."
    git pull origin main
else
    echo "  Cloning repository..."

    # Try SSH first
    if ssh -T git@github.com 2>&1 | grep -q "successfully authenticated"; then
        echo "  ✓ SSH key configured"
        git clone git@github.com:GiangHuynh16/ViSemPar_new1.git
    else
        echo "  ⚠️  SSH key not configured"
        echo "  Please run the following to setup SSH key:"
        echo ""
        echo "    ssh-keygen -t ed25519 -C \"your_email@example.com\""
        echo "    cat ~/.ssh/id_ed25519.pub"
        echo ""
        echo "  Then add the public key to GitHub:"
        echo "    https://github.com/settings/keys"
        echo ""
        echo "  After that, run this script again."
        exit 1
    fi

    cd ViSemPar_new1
fi
echo ""

# Step 4: Check GPU
echo "Step 4: Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
    echo ""

    TOTAL_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    TOTAL_VRAM_GB=$(echo "scale=0; $TOTAL_VRAM/1024" | bc)
    echo "  Total VRAM: ${TOTAL_VRAM_GB} GB"

    if [ "$TOTAL_VRAM_GB" -ge 40 ]; then
        echo "  ✓ Excellent! >= 40GB VRAM - Can use full MTUP config"
        RECOMMENDED_SEQ=2048
        RECOMMENDED_BATCH=2
    elif [ "$TOTAL_VRAM_GB" -ge 32 ]; then
        echo "  ✓ Good! 32GB VRAM - Can use medium config"
        RECOMMENDED_SEQ=1024
        RECOMMENDED_BATCH=1
    else
        echo "  ✓ Acceptable. 24GB VRAM - Must use minimal config"
        RECOMMENDED_SEQ=512
        RECOMMENDED_BATCH=1
    fi
else
    echo "  ✗ nvidia-smi not found"
    echo "  Please check if CUDA drivers are installed"
    exit 1
fi
echo ""

# Step 5: Check Python environment
echo "Step 5: Checking Python environment..."
if command -v conda &> /dev/null; then
    echo "  ✓ Conda found: $(conda --version)"
    echo "  Current environment: ${CONDA_DEFAULT_ENV:-base}"
    echo ""

    # Check if lora_py310 exists
    if conda env list | grep -q "lora_py310"; then
        echo "  ✓ Found existing environment: lora_py310"
        echo "  To activate: conda activate lora_py310"
    elif conda env list | grep -q "baseline_7b"; then
        echo "  ✓ Found existing environment: baseline_7b"
        echo "  To activate: conda activate baseline_7b"
    else
        echo "  ⚠️  No suitable environment found"
        echo "  Create new environment with:"
        echo ""
        echo "    conda create -n baseline_7b python=3.10 -y"
        echo "    conda activate baseline_7b"
        echo "    pip install torch transformers peft accelerate datasets smatch penman"
    fi
else
    echo "  ⚠️  Conda not found"
    echo "  Python: $(python --version 2>&1 || echo 'Not found')"
fi
echo ""

# Step 6: Create required directories
echo "Step 6: Creating required directories..."
mkdir -p outputs/checkpoints
mkdir -p logs
mkdir -p results
echo "  ✓ Created: outputs/checkpoints, logs, results"
echo ""

# Step 7: Check data files
echo "Step 7: Checking data files..."
if [ -d "data" ]; then
    echo "  ✓ data/ directory exists"
    DATA_FILES=$(ls -1 data/*.txt 2>/dev/null | wc -l)
    echo "  Found $DATA_FILES .txt files in data/"

    if [ -f "data/train_amr_1.txt" ] && [ -f "data/train_amr_2.txt" ]; then
        echo "  ✓ Training files present"
    else
        echo "  ⚠️  Training files missing"
    fi
else
    echo "  ⚠️  data/ directory not found"
    echo "  You may need to copy data files to this server"
fi
echo ""

# Step 8: Suggest config based on VRAM
echo "Step 8: Configuration recommendation..."
echo ""
if [ ! -z "$RECOMMENDED_SEQ" ]; then
    CURRENT_SEQ=$(grep "^MAX_SEQ_LENGTH = " config/config.py | awk '{print $3}')
    CURRENT_BATCH=$(grep '"per_device_train_batch_size":' config/config.py | awk -F': ' '{print $2}' | awk '{print $1}' | tr -d ',')

    echo "  Current config:"
    echo "    MAX_SEQ_LENGTH: $CURRENT_SEQ"
    echo "    batch_size: $CURRENT_BATCH"
    echo ""
    echo "  Recommended for ${TOTAL_VRAM_GB}GB VRAM:"
    echo "    MAX_SEQ_LENGTH: $RECOMMENDED_SEQ"
    echo "    batch_size: $RECOMMENDED_BATCH"
    echo ""

    if [ "$CURRENT_SEQ" != "$RECOMMENDED_SEQ" ]; then
        echo "  ⚠️  Config mismatch!"
        echo ""
        echo "  To adjust config for this server's GPU:"
        echo "    nano config/config.py"
        echo ""
        echo "  Change:"
        echo "    MAX_SEQ_LENGTH = $RECOMMENDED_SEQ"
        echo "    \"per_device_train_batch_size\": $RECOMMENDED_BATCH"
        echo ""
    else
        echo "  ✓ Config matches recommended values"
    fi
fi
echo ""

# Summary
echo "=========================================="
echo "SETUP SUMMARY"
echo "=========================================="
echo ""
echo "Location: $(pwd)"
echo "GPU VRAM: ${TOTAL_VRAM_GB:-Unknown} GB"
echo "Recommended config:"
echo "  • MAX_SEQ_LENGTH: ${RECOMMENDED_SEQ:-512}"
echo "  • batch_size: ${RECOMMENDED_BATCH:-1}"
echo ""
echo "Next steps:"
echo ""
echo "1. Activate Python environment:"
echo "   conda activate lora_py310  # or baseline_7b"
echo ""
echo "2. (Optional) Adjust config if needed:"
echo "   nano config/config.py"
echo ""
echo "3. Start training in tmux:"
echo "   tmux new -s baseline_7b"
echo "   bash VERIFY_AND_START.sh"
echo ""
echo "4. Detach from tmux: Ctrl+B then D"
echo ""
echo "5. Monitor training:"
echo "   watch -n 1 nvidia-smi"
echo "   tail -f logs/training_baseline*.log"
echo ""
echo "=========================================="
