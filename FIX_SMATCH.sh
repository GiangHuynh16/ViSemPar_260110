#!/bin/bash
# Fix SMATCH installation and test

set -e

echo "==========================================="
echo "FIX SMATCH INSTALLATION"
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

# Uninstall old smatch
echo "Step 1: Removing old smatch..."
pip uninstall smatch -y 2>/dev/null || echo "  (not installed)"
echo ""

# Install correct smatch version
echo "Step 2: Installing smatch from source..."
pip install git+https://github.com/snowblink14/smatch.git
echo ""

# Test installation
echo "Step 3: Testing smatch installation..."
python << 'PYEOF'
import smatch

print("Available functions in smatch:")
for attr in dir(smatch):
    if not attr.startswith('_'):
        print(f"  - {attr}")

# Try to parse a simple AMR
try:
    test_amr = "(p / person)"
    print(f"\nTesting AMR parsing: {test_amr}")

    # Check which function to use
    if hasattr(smatch, 'AMR'):
        amr_obj = smatch.AMR.parse_AMR_line(test_amr)
        print("  ✓ Using smatch.AMR.parse_AMR_line()")
    elif hasattr(smatch, 'parse_amr_line'):
        amr_obj = smatch.parse_amr_line(test_amr)
        print("  ✓ Using smatch.parse_amr_line()")
    else:
        print("  ✗ Cannot find AMR parsing function")
        import sys
        sys.exit(1)

    print("  ✓ SMATCH installed correctly!")

except Exception as e:
    print(f"  ✗ Error: {e}")
    import sys
    sys.exit(1)
PYEOF

echo ""
echo "==========================================="
echo "SMATCH READY"
echo "==========================================="
echo ""
echo "Now re-run evaluation:"
echo "  python evaluate_baseline.py \\"
echo "      --predictions public_test_result_baseline_7b.txt \\"
echo "      --sentences data/public_test.txt \\"
echo "      --gold data/public_test_ground_truth.txt \\"
echo "      --formatted-output predictions_formatted.txt \\"
echo "      --results evaluation_results.txt"
echo ""
