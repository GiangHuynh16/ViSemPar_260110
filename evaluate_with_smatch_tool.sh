#!/bin/bash
# Calculate SMATCH using command-line smatch tool

set -e

echo "=========================================="
echo "SMATCH EVALUATION USING CLI TOOL"
echo "=========================================="
echo ""

PRED_FILE="${1:-predictions_formatted.txt}"
GOLD_FILE="${2:-data/public_test_ground_truth.txt}"

echo "Predictions: $PRED_FILE"
echo "Gold: $GOLD_FILE"
echo ""

# Check if files exist
if [ ! -f "$PRED_FILE" ]; then
    echo "ERROR: Predictions file not found: $PRED_FILE"
    exit 1
fi

if [ ! -f "$GOLD_FILE" ]; then
    echo "ERROR: Gold file not found: $GOLD_FILE"
    exit 1
fi

# Count AMRs
PRED_COUNT=$(grep -c "^#::snt" "$PRED_FILE" || echo 0)
GOLD_COUNT=$(grep -c "^#::snt" "$GOLD_FILE" || echo 0)

echo "Predictions: $PRED_COUNT AMRs"
echo "Gold: $GOLD_COUNT AMRs"
echo ""

# Try using smatch as command-line tool
if command -v smatch &> /dev/null; then
    echo "Using smatch command-line tool..."
    echo ""

    smatch -f "$PRED_FILE" "$GOLD_FILE" --pr

elif python -c "import smatch" 2>/dev/null; then
    echo "Using smatch Python module..."
    echo ""

    # Use Python to call smatch.main()
    python -m smatch -f "$PRED_FILE" "$GOLD_FILE" --pr

else
    echo "ERROR: smatch not found!"
    echo ""
    echo "Install with: pip install smatch"
    exit 1
fi

echo ""
echo "=========================================="
echo "EVALUATION COMPLETE"
echo "=========================================="
