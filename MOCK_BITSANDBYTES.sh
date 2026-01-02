#!/bin/bash
# Create a mock bitsandbytes module so PEFT can import it
# But it won't actually be used since we're not using quantization

set -e

echo "==========================================="
echo "MOCK BITSANDBYTES MODULE"
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

# Uninstall real bitsandbytes first
echo "Step 1: Removing real bitsandbytes..."
pip uninstall bitsandbytes -y 2>/dev/null || echo "  (not installed)"
echo "  ✓ Removed"
echo ""

# Find site-packages directory
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
echo "Step 2: Site-packages directory: $SITE_PACKAGES"
echo ""

# Create mock bitsandbytes module
echo "Step 3: Creating mock bitsandbytes module..."
mkdir -p "$SITE_PACKAGES/bitsandbytes"

cat > "$SITE_PACKAGES/bitsandbytes/__init__.py" << 'PYEOF'
"""Mock bitsandbytes module - not actually used"""
__version__ = "0.0.0-mock"

class MockBnB:
    def __getattr__(self, name):
        return None

# Mock the imports that PEFT needs
import sys
sys.modules['bitsandbytes.nn'] = MockBnB()
sys.modules['bitsandbytes.functional'] = MockBnB()
sys.modules['bitsandbytes.optim'] = MockBnB()
PYEOF

echo "  ✓ Mock module created"
echo ""

# Verify
echo "Step 4: Verifying mock module..."
python << 'PYEOF'
import bitsandbytes as bnb
print(f"  ✓ bitsandbytes mock: {bnb.__version__}")
print("  ✓ Import successful (mock version)")
PYEOF

echo ""
echo "==========================================="
echo "MOCK INSTALLATION COMPLETE"
echo "==========================================="
echo ""
echo "Now start training:"
echo "  bash START_TRAINING_NOW.sh"
echo ""
