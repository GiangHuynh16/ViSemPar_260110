#!/bin/bash
# Test all checkpoints to find the best one
# Run: bash TEST_ALL_CHECKPOINTS.sh

echo "=========================================="
echo "TEST ALL CHECKPOINTS"
echo "=========================================="
echo ""

cd /mnt/nghiepth/giangha/visempar/ViSemPar_new1

# Activate conda
source ~/anaconda3/etc/profile.d/conda.sh
conda activate baseline_final

# Find latest model
LATEST_MODEL=$(ls -t outputs/ | grep baseline_fixed | head -1)

if [ -z "$LATEST_MODEL" ]; then
    echo "❌ No trained model found in outputs/"
    exit 1
fi

echo "Found model: $LATEST_MODEL"
echo ""

# Get all checkpoints
CHECKPOINTS=$(ls "outputs/$LATEST_MODEL/" | grep checkpoint | sort -V)

if [ -z "$CHECKPOINTS" ]; then
    echo "❌ No checkpoints found in outputs/$LATEST_MODEL"
    exit 1
fi

echo "Available checkpoints:"
echo "$CHECKPOINTS"
echo ""
echo "Total: $(echo "$CHECKPOINTS" | wc -l) checkpoints"
echo ""

# Ask which checkpoints to test
echo "Which checkpoints do you want to test?"
echo "  1. Test all checkpoints (may take a while)"
echo "  2. Test specific checkpoints (e.g., 100,200,300)"
echo "  3. Test early checkpoints only (100-500)"
echo "  4. Test every other checkpoint"
echo ""
read -p "Choice (1-4): " CHOICE

case $CHOICE in
    1)
        # Test all
        TEST_CHECKPOINTS="$CHECKPOINTS"
        ;;
    2)
        # Specific checkpoints
        read -p "Enter checkpoint numbers (comma-separated, e.g., 100,200,300): " NUMS
        TEST_CHECKPOINTS=""
        for num in $(echo $NUMS | tr ',' ' '); do
            ckpt="checkpoint-$num"
            if echo "$CHECKPOINTS" | grep -q "^$ckpt$"; then
                TEST_CHECKPOINTS="$TEST_CHECKPOINTS $ckpt"
            else
                echo "⚠️  Checkpoint $ckpt not found, skipping..."
            fi
        done
        ;;
    3)
        # Early checkpoints only (100-500)
        TEST_CHECKPOINTS=$(echo "$CHECKPOINTS" | grep -E 'checkpoint-[1-5][0-9][0-9]$')
        ;;
    4)
        # Every other checkpoint
        TEST_CHECKPOINTS=$(echo "$CHECKPOINTS" | awk 'NR % 2 == 1')
        ;;
    *)
        echo "Invalid choice!"
        exit 1
        ;;
esac

echo ""
echo "Will test these checkpoints:"
echo "$TEST_CHECKPOINTS"
echo ""

read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 1
fi

echo ""
echo "=========================================="
echo "TESTING CHECKPOINTS"
echo "=========================================="
echo ""

# Create results directory
mkdir -p evaluation_results/checkpoint_comparison

# Summary file
SUMMARY_FILE="evaluation_results/checkpoint_comparison/summary_$(date +%Y%m%d_%H%M%S).txt"

echo "Checkpoint Comparison Results" > "$SUMMARY_FILE"
echo "Generated: $(date)" >> "$SUMMARY_FILE"
echo "Model: $LATEST_MODEL" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"
echo "Checkpoint | Valid AMRs | Invalid AMRs | Valid %" >> "$SUMMARY_FILE"
echo "-----------|------------|--------------|----------" >> "$SUMMARY_FILE"

# Best checkpoint tracking
BEST_CKPT=""
BEST_VALID=0
BEST_PERCENT=0

# Test each checkpoint
for CKPT in $TEST_CHECKPOINTS; do
    echo ""
    echo "=========================================="
    echo "Testing: $CKPT"
    echo "=========================================="
    echo ""

    OUTPUT_FILE="evaluation_results/checkpoint_comparison/${CKPT}.txt"

    # Run prediction
    python predict_baseline_fixed.py \
        --model "outputs/$LATEST_MODEL/$CKPT" \
        --test-file data/public_test.txt \
        --output "$OUTPUT_FILE"

    if [ $? -ne 0 ]; then
        echo "❌ Prediction failed for $CKPT"
        echo "$CKPT | ERROR | ERROR | ERROR" >> "$SUMMARY_FILE"
        continue
    fi

    # Validate
    echo ""
    echo "Validating..."

    VALIDATION=$(python validate_vietnamese_output.py --file "$OUTPUT_FILE" 2>&1)

    # Extract metrics
    VALID=$(echo "$VALIDATION" | grep "Valid AMRs:" | grep -oP '\d+(?= \()')
    INVALID=$(echo "$VALIDATION" | grep "Invalid AMRs:" | grep -oP '\d+(?= \()')
    PERCENT=$(echo "$VALIDATION" | grep "Valid AMRs:" | grep -oP '\d+\.\d+(?=%)')

    if [ -z "$VALID" ]; then
        VALID="?"
        INVALID="?"
        PERCENT="0.0"
    fi

    echo "$CKPT | $VALID/150 | $INVALID/150 | $PERCENT%" >> "$SUMMARY_FILE"

    echo ""
    echo "Results for $CKPT:"
    echo "  Valid: $VALID/150"
    echo "  Invalid: $INVALID/150"
    echo "  Valid %: $PERCENT%"
    echo ""

    # Track best
    if [ -n "$VALID" ] && [ "$VALID" != "?" ]; then
        if [ "$VALID" -gt "$BEST_VALID" ]; then
            BEST_VALID=$VALID
            BEST_CKPT=$CKPT
            BEST_PERCENT=$PERCENT
        fi
    fi
done

echo ""
echo "=========================================="
echo "TESTING COMPLETE"
echo "=========================================="
echo ""
echo "Summary saved to: $SUMMARY_FILE"
echo ""

# Show summary
cat "$SUMMARY_FILE"

echo ""
echo "=========================================="
echo "BEST CHECKPOINT"
echo "=========================================="
echo ""

if [ -n "$BEST_CKPT" ]; then
    echo "✅ Best checkpoint: $BEST_CKPT"
    echo "   Valid AMRs: $BEST_VALID/150 ($BEST_PERCENT%)"
    echo ""
    echo "To use this checkpoint:"
    echo ""
    echo "  python predict_baseline_fixed.py \\"
    echo "      --model \"outputs/$LATEST_MODEL/$BEST_CKPT\" \\"
    echo "      --test-file data/public_test.txt \\"
    echo "      --output evaluation_results/baseline_7b_fixed/predictions.txt"
    echo ""
    echo "  python -m smatch -f \\"
    echo "      evaluation_results/baseline_7b_fixed/predictions.txt \\"
    echo "      data/public_test_ground_truth.txt \\"
    echo "      --significant 4"
    echo ""
else
    echo "❌ No valid checkpoint found!"
fi

echo ""
echo "All results saved in: evaluation_results/checkpoint_comparison/"
echo ""
