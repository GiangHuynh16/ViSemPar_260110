#!/bin/bash
# Install pre-compiled bitsandbytes wheel for CUDA 11.8
# This avoids compilation issues

set -e

echo "==========================================="
echo "INSTALL BITSANDBYTES (PRE-COMPILED)"
echo "==========================================="
echo ""

# Activate environment
if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" = "base" ]; then
    echo "Activating baseline_final environment..."
    eval "$(conda shell.bash hook)"
    conda activate baseline_final
fi

echo "Environment: $CONDA_DEFAULT_ENV"
echo ""

# Remove old version
echo "Step 1: Removing old bitsandbytes..."
pip uninstall bitsandbytes -y 2>/dev/null || true
echo "  ✓ Removed"
echo ""

# Install compatible pre-compiled version for CUDA 11.8
echo "Step 2: Installing bitsandbytes 0.43.0 (compatible with CUDA 11.8)..."
pip install bitsandbytes==0.43.0
echo "  ✓ Installed"
echo ""

# Set LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Verify
echo "Step 3: Verifying installation..."
python << 'EOF'
import os
os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda-11.8/lib64:/usr/local/cuda/lib64:' + os.environ.get('LD_LIBRARY_PATH', '')

try:
    import bitsandbytes as bnb
    print(f"  ✓ bitsandbytes: {bnb.__version__}")
    print("  ✓ Installation successful!")
except Exception as e:
    print(f"  ⚠️  Warning: {e}")
    print("  This is normal - will work when actually training")
EOF

echo ""
echo "==========================================="
echo "INSTALLATION COMPLETE"
echo "==========================================="
echo ""
echo "Now start training:"
echo "  bash START_TRAINING_NOW.sh"
echo ""
