#!/bin/bash
# Check how PEFT imports bitsandbytes

set -e

echo "==========================================="
echo "CHECK PEFT BITSANDBYTES IMPORTS"
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

# Find PEFT installation
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
PEFT_DIR="$SITE_PACKAGES/peft"

echo "PEFT directory: $PEFT_DIR"
echo ""

echo "Searching for all bitsandbytes imports in PEFT..."
echo ""
grep -r "import bitsandbytes\|from bitsandbytes" "$PEFT_DIR" || echo "No imports found"
echo ""

echo "==========================================="
echo "Checking PEFT version and dependencies..."
echo "==========================================="
echo ""

python << 'PYEOF'
import peft
print(f"PEFT version: {peft.__version__}")

# Check if PEFT has optional dependencies list
import importlib.metadata as metadata
try:
    dist = metadata.distribution('peft')
    print(f"\nPEFT metadata:")
    print(f"  Requires: {dist.metadata.get_all('Requires-Dist')[:5]}")  # First 5 deps
except Exception as e:
    print(f"Could not get metadata: {e}")
PYEOF

echo ""
echo "==========================================="
echo "CHECK COMPLETE"
echo "==========================================="
