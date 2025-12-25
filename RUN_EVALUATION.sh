#!/bin/bash
################################################################################
# EVALUATE MTUP MODEL
# Chạy evaluation trên public test data
################################################################################

echo "========================================================================"
echo "📊 MTUP MODEL EVALUATION"
echo "========================================================================"
echo ""

# Activate environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate lora_py310

# Find latest checkpoint
echo "Finding latest checkpoint..."
latest_checkpoint=$(ls -d outputs/checkpoints_mtup/checkpoint-* 2>/dev/null | sort -V | tail -1)

if [ -z "$latest_checkpoint" ]; then
    echo "❌ No checkpoints found in outputs/checkpoints_mtup/"
    echo ""
    echo "Available checkpoints:"
    ls -lh outputs/checkpoints_mtup/ 2>/dev/null || echo "  (none)"
    exit 1
fi

echo "✓ Found checkpoint: $latest_checkpoint"
echo ""

# Check test file
test_file="data/public_test_ground_truth.txt"
if [ ! -f "$test_file" ]; then
    echo "⚠️  Test file not found: $test_file"
    echo "Looking for alternative test files..."

    # Try other locations
    if [ -f "data/test_amr.txt" ]; then
        test_file="data/test_amr.txt"
    elif [ -f "data/train_amr_1.txt" ]; then
        test_file="data/train_amr_1.txt"
        echo "⚠️  Using training data for testing (not ideal)"
    else
        echo "❌ No test data found"
        exit 1
    fi
fi

echo "✓ Test file: $test_file"
echo ""

# Ask for sample limit
echo "How many samples to evaluate?"
echo "  1) Quick test (10 samples, ~2 min)"
echo "  2) Medium test (50 samples, ~10 min)"
echo "  3) Full test (all samples, ~30-60 min)"
echo ""
read -p "Choose [1-3] or enter custom number: " choice

case $choice in
    1)
        max_samples=10
        ;;
    2)
        max_samples=50
        ;;
    3)
        max_samples=""
        ;;
    *)
        max_samples=$choice
        ;;
esac

echo ""
echo "========================================================================"
echo "Starting evaluation..."
echo "========================================================================"
echo "  Checkpoint: $latest_checkpoint"
echo "  Test file: $test_file"
if [ -n "$max_samples" ]; then
    echo "  Samples: $max_samples"
else
    echo "  Samples: ALL"
fi
echo "========================================================================"
echo ""

# Run evaluation
if [ -n "$max_samples" ]; then
    python3 evaluate_mtup_model.py \
        --checkpoint "$latest_checkpoint" \
        --test-file "$test_file" \
        --max-samples "$max_samples" \
        --output "outputs/evaluation_results.json"
else
    python3 evaluate_mtup_model.py \
        --checkpoint "$latest_checkpoint" \
        --test-file "$test_file" \
        --output "outputs/evaluation_results.json"
fi

echo ""
echo "========================================================================"
echo "✅ EVALUATION COMPLETE"
echo "========================================================================"
echo ""
echo "Results saved to: outputs/evaluation_results.json"
echo ""
echo "View results:"
echo "  cat outputs/evaluation_results.json"
echo ""
echo "========================================================================"
