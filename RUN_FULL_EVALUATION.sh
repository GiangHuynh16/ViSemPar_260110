#!/bin/bash
################################################################################
# RUN FULL EVALUATION - Không cần input, tự động chạy hết
# Chạy evaluation trên TOÀN BỘ public test set
################################################################################

echo "========================================================================"
echo "📊 FULL MTUP MODEL EVALUATION"
echo "========================================================================"
echo ""

# Activate environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate lora_py310

# Find latest checkpoint
echo "Finding checkpoints..."

# Try different checkpoint patterns
latest_checkpoint=$(ls -dt outputs/checkpoints_mtup/checkpoint-* 2>/dev/null | head -1)

if [ -z "$latest_checkpoint" ]; then
    # Try MTUP final checkpoints
    latest_checkpoint=$(ls -dt outputs/checkpoints_mtup/mtup_*_final 2>/dev/null | head -1)
fi

if [ -z "$latest_checkpoint" ]; then
    echo "❌ No checkpoints found in outputs/checkpoints_mtup/"
    echo ""
    echo "Available directories:"
    ls -lh outputs/checkpoints_mtup/ 2>/dev/null || echo "  (none)"
    exit 1
fi

echo "✓ Found checkpoint: $latest_checkpoint"
echo ""

# Check test file
test_file="data/public_test_ground_truth.txt"
if [ ! -f "$test_file" ]; then
    echo "❌ Test file not found: $test_file"
    exit 1
fi

echo "✓ Test file: $test_file"

# Count total samples
total_samples=$(grep -c "^# ::snt" "$test_file" 2>/dev/null || echo "unknown")
echo "✓ Total test samples: $total_samples"
echo ""

# Estimate time
if [ "$total_samples" != "unknown" ]; then
    # Assume ~20 seconds per sample (based on 10 samples in 200 seconds)
    estimated_minutes=$((total_samples * 20 / 60))
    echo "⏱️  Estimated time: ~$estimated_minutes minutes"
fi

echo ""
echo "========================================================================"
echo "Starting FULL evaluation..."
echo "========================================================================"
echo "  Checkpoint: $latest_checkpoint"
echo "  Test file: $test_file"
echo "  Samples: ALL ($total_samples)"
echo "  Output: outputs/evaluation_results_full.json"
echo "========================================================================"
echo ""

# Create timestamp
timestamp=$(date +"%Y%m%d_%H%M%S")

# Run evaluation
python3 evaluate_mtup_model.py \
    --checkpoint "$latest_checkpoint" \
    --test-file "$test_file" \
    --output "outputs/evaluation_results_full_${timestamp}.json" \
    2>&1 | tee "outputs/evaluation_full_${timestamp}.log"

echo ""
echo "========================================================================"
echo "✅ EVALUATION COMPLETE"
echo "========================================================================"
echo ""
echo "Results saved to: outputs/evaluation_results_full_${timestamp}.json"
echo "Log saved to: outputs/evaluation_full_${timestamp}.log"
echo ""
echo "View results:"
echo "  cat outputs/evaluation_results_full_${timestamp}.json"
echo ""
echo "View last 20 lines of log:"
echo "  tail -20 outputs/evaluation_full_${timestamp}.log"
echo ""
echo "========================================================================"
