#!/bin/bash
# Install bitsandbytes for 4-bit quantization

set -e

echo "==========================================="
echo "INSTALL BITSANDBYTES"
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

# Install bitsandbytes
echo "Installing bitsandbytes..."
pip install bitsandbytes==0.41.1
echo "  ✓ bitsandbytes installed"
echo ""

# Verify
echo "Verifying installation..."
python << 'EOF'
try:
    import bitsandbytes
    print(f"  ✓ bitsandbytes: {bitsandbytes.__version__}")
except Exception as e:
    print(f"  ✗ Error: {e}")
    exit(1)
EOF

echo ""
echo "==========================================="
echo "INSTALLATION COMPLETE"
echo "==========================================="
echo ""
echo "Now you can start training:"
echo "  bash START_TRAINING_NOW.sh"
echo ""
