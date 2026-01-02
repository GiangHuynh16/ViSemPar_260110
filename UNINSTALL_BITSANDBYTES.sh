#!/bin/bash
# Uninstall bitsandbytes completely
# We don't need it since we're not using quantization

set -e

echo "==========================================="
echo "UNINSTALL BITSANDBYTES"
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

# Uninstall bitsandbytes
echo "Removing bitsandbytes..."
pip uninstall bitsandbytes -y 2>/dev/null || echo "  (not installed)"
echo "  ✓ Removed"
echo ""

# Verify it's gone
echo "Verifying removal..."
python << 'PYEOF'
try:
    import bitsandbytes
    print("  ⚠️  WARNING: bitsandbytes still installed!")
    exit(1)
except ImportError:
    print("  ✓ bitsandbytes successfully removed")
PYEOF

echo ""
echo "==========================================="
echo "UNINSTALL COMPLETE"
echo "==========================================="
echo ""
echo "Now start training:"
echo "  bash START_TRAINING_NOW.sh"
echo ""
