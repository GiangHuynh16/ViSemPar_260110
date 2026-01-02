#!/bin/bash
# Upgrade PEFT to newer version that doesn't require bitsandbytes

set -e

echo "==========================================="
echo "UPGRADE PEFT"
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

# Check current version
echo "Step 1: Current PEFT version..."
python -c "import peft; print(f'  Current: peft {peft.__version__}')" || echo "  Not installed"
echo ""

# Uninstall bitsandbytes
echo "Step 2: Removing bitsandbytes..."
pip uninstall bitsandbytes -y 2>/dev/null || echo "  (not installed)"
echo "  ✓ Removed"
echo ""

# Upgrade PEFT
echo "Step 3: Upgrading PEFT to latest version..."
pip install --upgrade peft
echo "  ✓ Upgraded"
echo ""

# Check new version
echo "Step 4: New PEFT version..."
python -c "import peft; print(f'  New: peft {peft.__version__}')"
echo ""

# Verify PEFT works without bitsandbytes
echo "Step 5: Verifying PEFT works without bitsandbytes..."
python << 'PYEOF'
try:
    from peft import LoraConfig, get_peft_model
    print("  ✓ PEFT imports successfully")
except Exception as e:
    print(f"  ✗ Error: {e}")
    exit(1)
PYEOF

echo ""
echo "==========================================="
echo "UPGRADE COMPLETE"
echo "==========================================="
echo ""
echo "Now start training:"
echo "  bash START_TRAINING_NOW.sh"
echo ""
