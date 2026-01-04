#!/bin/bash
# Fix prediction issues NOW

echo "================================================================================"
echo "FIX MTUP PREDICTION ISSUES"
echo "================================================================================"
echo ""

cd /mnt/nghiepth/giangha/visempar/ViSemPar_new1 || exit 1

echo "Step 1: Verify test file format"
echo "--------------------------------------------------------------------------------"
echo "First 5 lines of public_test.txt:"
head -5 data/public_test.txt
echo ""

echo "Expected: Only sentences (no AMR)"
echo "If you see AMR graphs above, the file format is WRONG!"
echo ""
read -p "Is the test file format correct? (y/n): " correct

if [ "$correct" != "y" ]; then
    echo ""
    echo "ERROR: Test file has wrong format!"
    echo ""
    echo "Extracting sentences from ground truth..."

    # Extract sentences from ground truth
    grep "^#::snt" data/public_test_ground_truth.txt | sed 's/^#::snt //' > data/public_test_sentences_only.txt

    echo "Created: data/public_test_sentences_only.txt"
    echo ""
    echo "First 3 sentences:"
    head -3 data/public_test_sentences_only.txt
    echo ""

    TEST_FILE="data/public_test_sentences_only.txt"
else
    TEST_FILE="data/public_test.txt"
fi

echo ""
echo "Step 2: Choose model checkpoint"
echo "--------------------------------------------------------------------------------"
echo "Available checkpoints:"
ls -ld outputs/mtup_fixed_*/checkpoint-* 2>/dev/null | grep -v "^d" | head -10
echo ""

echo "We have:"
echo "  outputs/mtup_fixed_20260104_082506/checkpoint-148 (trained 08:25)"
echo "  outputs/mtup_fixed_20260104_105638/checkpoint-148 (trained 10:56)"
echo ""

read -p "Which one to use? Enter path (or press Enter for latest): " MODEL_PATH

if [ -z "$MODEL_PATH" ]; then
    # Use latest
    MODEL_PATH=$(ls -td outputs/mtup_fixed_*/checkpoint-148 2>/dev/null | head -1)
fi

echo ""
echo "Using model: $MODEL_PATH"
echo ""

if [ ! -f "$MODEL_PATH/adapter_config.json" ]; then
    echo "ERROR: Model not found at $MODEL_PATH"
    exit 1
fi

echo ""
echo "Step 3: Delete old predictions"
echo "--------------------------------------------------------------------------------"
rm -f evaluation_results/mtup_predictions_FIXED*.txt
echo "Deleted old predictions"
echo ""

echo ""
echo "Step 4: Run prediction with CORRECT alignment"
echo "--------------------------------------------------------------------------------"

OUTPUT_FILE="evaluation_results/mtup_predictions_REALIGNED_$(date +%Y%m%d_%H%M%S).txt"

echo "Command:"
echo "python3 predict_mtup_fixed.py \\"
echo "    --model $MODEL_PATH \\"
echo "    --test-file $TEST_FILE \\"
echo "    --output $OUTPUT_FILE \\"
echo "    --verbose"
echo ""

read -p "Run prediction? (y/n): " run

if [ "$run" = "y" ]; then
    python3 predict_mtup_fixed.py \
        --model "$MODEL_PATH" \
        --test-file "$TEST_FILE" \
        --output "$OUTPUT_FILE" \
        --verbose 2>&1 | tee "prediction_realigned_$(date +%Y%m%d_%H%M%S).log"

    echo ""
    echo "================================================================================"
    echo "PREDICTION COMPLETE"
    echo "================================================================================"
    echo ""

    # Check results
    if [ -f "$OUTPUT_FILE" ]; then
        PRED_COUNT=$(grep -c '^(' "$OUTPUT_FILE" 2>/dev/null || echo "0")
        echo "Predictions generated: $PRED_COUNT"

        # Count unique
        UNIQUE_COUNT=$(python3 << EOF
with open('$OUTPUT_FILE', 'r') as f:
    preds = f.read().strip().split('\n\n')
    print(len(set(preds)))
EOF
)
        echo "Unique predictions: $UNIQUE_COUNT"

        if [ "$UNIQUE_COUNT" -lt "$((PRED_COUNT / 2))" ]; then
            echo ""
            echo "⚠️  WARNING: Too many duplicate predictions ($UNIQUE_COUNT unique out of $PRED_COUNT)"
            echo "This suggests the model is overfitting or not trained properly."
            echo ""
        fi

        echo ""
        echo "First 3 predictions:"
        python3 << EOF
with open('$OUTPUT_FILE', 'r') as f:
    preds = f.read().strip().split('\n\n')
    for i, p in enumerate(preds[:3], 1):
        print(f"\n=== Prediction {i} ===")
        print(p[:200])
EOF

        echo ""
        echo "First 3 test sentences:"
        head -3 "$TEST_FILE"

        echo ""
        echo "================================================================================"
        echo "NEXT STEP: Calculate SMATCH"
        echo "================================================================================"
        echo ""
        echo "python3 compare_predictions.py \\"
        echo "    $OUTPUT_FILE \\"
        echo "    data/public_test_ground_truth.txt"
        echo ""
    else
        echo "ERROR: Output file not created!"
    fi
else
    echo "Cancelled"
fi

echo ""
echo "================================================================================"
