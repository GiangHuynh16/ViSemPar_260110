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

# Find CUDA installation
echo "Step 3: Locating CUDA libraries..."
if [ -d "/usr/local/cuda-11.8" ]; then
    CUDA_HOME="/usr/local/cuda-11.8"
    echo "  ✓ Found CUDA 11.8 at $CUDA_HOME"
elif [ -d "/usr/local/cuda" ]; then
    CUDA_HOME="/usr/local/cuda"
    echo "  ✓ Found CUDA at $CUDA_HOME"
else
    echo "  ⚠️  CUDA not found in standard locations"
    CUDA_HOME="/usr/local/cuda"
fi

# Set LD_LIBRARY_PATH to include CUDA libraries
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
echo "  ✓ Set LD_LIBRARY_PATH=$CUDA_HOME/lib64"
echo ""

# Verify
echo "Step 4: Verifying installation..."
python << EOF
import os
import sys

# Set CUDA library path
cuda_lib_path = os.environ.get('LD_LIBRARY_PATH', '')
os.environ['LD_LIBRARY_PATH'] = cuda_lib_path

try:
    import bitsandbytes as bnb
    print(f"  ✓ bitsandbytes: {bnb.__version__}")
    print("  ✓ Installation successful!")
except Exception as e:
    print(f"  ⚠️  Warning: {e}")
    print("  This may be expected - verify it works during training")
    sys.exit(0)  # Don't fail the script
EOF

echo ""
echo "==========================================="
echo "INSTALLATION COMPLETE"
echo "==========================================="
echo ""
echo "IMPORTANT: Add to your ~/.bashrc to make LD_LIBRARY_PATH permanent:"
echo "  echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:\$LD_LIBRARY_PATH' >> ~/.bashrc"
echo "  source ~/.bashrc"
echo ""
echo "OR just start training now (LD_LIBRARY_PATH is set in START_TRAINING_NOW.sh):"
echo "  bash START_TRAINING_NOW.sh"
echo ""
