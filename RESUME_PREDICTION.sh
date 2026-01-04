#!/bin/bash
# Resume MTUP prediction with improved error handling and checkpointing

echo "================================================================================"
echo "RESUME MTUP PREDICTION"
echo "================================================================================"
echo ""
echo "This script will:"
echo "  ✅ Save checkpoint every 10 predictions"
echo "  ✅ Handle errors gracefully"
echo "  ✅ Show progress every 10 sentences"
echo "  ✅ Prevent data loss if process stops"
echo ""

# Configuration
MODEL_PATH="outputs/mtup_fixed_20260104_082506/checkpoint-148"
TEST_FILE="data/public_test.txt"
OUTPUT_FILE="evaluation_results/mtup_predictions_FIXED.txt"

# Check if model exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ Error: Model not found at $MODEL_PATH"
    echo ""
    echo "Available models:"
    ls -ld outputs/*/checkpoint-* 2>/dev/null | tail -5
    echo ""
    echo "Please update MODEL_PATH in this script."
    exit 1
fi

# Create output directory
mkdir -p evaluation_results

echo "📊 Configuration:"
echo "  Model: $MODEL_PATH"
echo "  Test file: $TEST_FILE"
echo "  Output: $OUTPUT_FILE"
echo ""

# Count sentences
TOTAL_SENTENCES=$(grep -c . "$TEST_FILE")
echo "  Total sentences: $TOTAL_SENTENCES"
echo ""

# Check if output already exists
if [ -f "$OUTPUT_FILE" ]; then
    EXISTING_PREDS=$(grep -c '^(' "$OUTPUT_FILE" 2>/dev/null || echo "0")
    echo "⚠️  Warning: Output file already exists with ~$EXISTING_PREDS predictions"
    echo ""
    echo "Options:"
    echo "  1) Delete and start fresh"
    echo "  2) Keep and overwrite (predictions will restart from beginning)"
    echo "  3) Cancel"
    echo ""
    read -p "Choose (1/2/3): " choice

    case $choice in
        1)
            rm "$OUTPUT_FILE"
            echo "✅ Deleted existing file"
            ;;
        2)
            echo "✅ Will overwrite"
            ;;
        3)
            echo "Cancelled"
            exit 0
            ;;
        *)
            echo "Invalid choice, cancelling"
            exit 1
            ;;
    esac
    echo ""
fi

echo "🚀 Starting prediction..."
echo ""
echo "Monitor progress:"
echo "  - Saves checkpoint every 10 predictions"
echo "  - Shows progress every 10 sentences"
echo "  - Handles errors gracefully"
echo ""
echo "Press Ctrl+C to stop (progress will be saved)"
echo ""
sleep 2

# Run prediction with verbose output and save to log
python3 predict_mtup_fixed.py \
    --model "$MODEL_PATH" \
    --test-file "$TEST_FILE" \
    --output "$OUTPUT_FILE" \
    --verbose 2>&1 | tee "prediction_$(date +%Y%m%d_%H%M%S).log"

EXIT_CODE=$?

echo ""
echo "================================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ PREDICTION COMPLETE"
else
    echo "⚠️  PREDICTION STOPPED (exit code: $EXIT_CODE)"
fi
echo "================================================================================"
echo ""

# Check output
if [ -f "$OUTPUT_FILE" ]; then
    PRED_COUNT=$(grep -c '^(' "$OUTPUT_FILE" 2>/dev/null || echo "0")
    FILE_SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)

    echo "📊 Results:"
    echo "  Predictions generated: ~$PRED_COUNT"
    echo "  Expected: $TOTAL_SENTENCES"
    echo "  File size: $FILE_SIZE"
    echo "  Output: $OUTPUT_FILE"
    echo ""

    if [ "$PRED_COUNT" -ge "$TOTAL_SENTENCES" ]; then
        echo "✅ All predictions complete!"
        echo ""
        echo "🎯 Next steps:"
        echo ""
        echo "1. Calculate SMATCH:"
        echo ""
        echo "   python3 filter_valid_amrs.py \\"
        echo "       --predictions $OUTPUT_FILE \\"
        echo "       --ground-truth data/public_test_ground_truth.txt \\"
        echo "       --output-pred evaluation_results/mtup_valid.txt \\"
        echo "       --output-gold evaluation_results/gold_valid.txt"
        echo ""
        echo "   python -m smatch -f \\"
        echo "       evaluation_results/mtup_valid.txt \\"
        echo "       evaluation_results/gold_valid.txt \\"
        echo "       --significant 4"
        echo ""
        echo "2. Compare with Baseline (F1=0.47, Validity=91.3%)"
        echo ""
    else
        echo "⚠️  Incomplete: $PRED_COUNT/$TOTAL_SENTENCES predictions"
        echo ""
        echo "To resume, run this script again."
        echo ""
    fi
else
    echo "❌ Output file not found"
    echo ""
    echo "Check logs for errors:"
    echo "  ls -lt prediction_*.log | head -1"
    echo ""
fi

echo "================================================================================"
