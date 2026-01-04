#!/bin/bash
# Quick test on 10 sentences to verify prediction works

echo "================================================================================"
echo "MTUP QUICK TEST (10 sentences)"
echo "================================================================================"
echo ""

MODEL_PATH="outputs/mtup_fixed_20260104_082506/checkpoint-148"
SMALL_TEST="data/test_small.txt"
OUTPUT="evaluation_results/test_small_output.txt"

# Check model
if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ Error: Model not found at $MODEL_PATH"
    echo ""
    echo "Available models:"
    ls -ld outputs/*/checkpoint-* 2>/dev/null | tail -5
    exit 1
fi

# Create small test file
echo "📝 Creating small test file (10 sentences)..."
head -10 data/public_test.txt > "$SMALL_TEST"
echo "✅ Created: $SMALL_TEST"
echo ""

# Run prediction
echo "🚀 Running prediction on 10 sentences..."
echo "   (This should take ~2-3 minutes)"
echo ""

python3 predict_mtup_fixed.py \
    --model "$MODEL_PATH" \
    --test-file "$SMALL_TEST" \
    --output "$OUTPUT" \
    --verbose

EXIT_CODE=$?

echo ""
echo "================================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ TEST COMPLETE"
else
    echo "❌ TEST FAILED (exit code: $EXIT_CODE)"
fi
echo "================================================================================"
echo ""

# Show results
if [ -f "$OUTPUT" ]; then
    echo "📊 Results:"
    echo ""

    # Count predictions
    PRED_COUNT=$(grep -c '^(' "$OUTPUT" 2>/dev/null || echo "0")
    echo "  Predictions: $PRED_COUNT / 10"
    echo ""

    # Show first prediction
    echo "📝 First prediction:"
    echo "  ────────────────────────────────────────"
    head -15 "$OUTPUT"
    echo "  ────────────────────────────────────────"
    echo ""

    if [ "$PRED_COUNT" -eq 10 ]; then
        echo "✅ All 10 predictions generated successfully!"
        echo ""
        echo "🎯 Next step: Run full prediction"
        echo ""
        echo "   bash RESUME_PREDICTION.sh"
        echo ""
    else
        echo "⚠️  Only $PRED_COUNT/10 predictions generated"
        echo ""
        echo "Check the output for errors."
    fi
else
    echo "❌ No output file generated"
    echo ""
    echo "Prediction may have crashed. Check error messages above."
fi

# Cleanup
echo "🧹 Cleanup?"
read -p "Delete test files? (y/n): " cleanup
if [ "$cleanup" = "y" ]; then
    rm -f "$SMALL_TEST" "$OUTPUT"
    echo "✅ Deleted test files"
fi

echo ""
echo "================================================================================"
