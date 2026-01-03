#!/bin/bash
# Train baseline 7B model with ALL FIXES
# Run: bash TRAIN_BASELINE_FIXED.sh

echo "=========================================="
echo "TRAIN BASELINE 7B - ALL FIXES APPLIED"
echo "=========================================="
echo ""
echo "FIXES APPLIED:"
echo "  1. ✅ Instruction masking - No tokenization mismatch"
echo "  2. ✅ Simplified prompt - 3 lines instead of 135"
echo "  3. ✅ Reduced epochs - 2 instead of 15 (avoid overfitting)"
echo "  4. ✅ More checkpoints - Save every 100 steps"
echo "  5. ✅ Optimized inference - Better temperature and top_p"
echo ""
echo "EXPECTED RESULTS:"
echo "  - Target: 80-90% valid AMRs (up from 70%)"
echo "  - Training time: 2-3 hours (reduced from 4-5 hours)"
echo "  - Checkpoints: 100, 200, 300, 400, 500... for testing"
echo ""
echo "=========================================="
echo ""

cd /mnt/nghiepth/giangha/visempar/ViSemPar_new1

# Activate conda
source ~/anaconda3/etc/profile.d/conda.sh
conda activate baseline_final

# Check if in correct environment
if [ "$CONDA_DEFAULT_ENV" != "baseline_final" ]; then
    echo "❌ Not in baseline_final environment!"
    echo "Please run: conda activate baseline_final"
    exit 1
fi

echo "✓ Conda environment: $CONDA_DEFAULT_ENV"
echo ""

# Check GPU
echo "Checking GPU..."
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

# Confirm training
read -p "Start training? This will take 2-3 hours. (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Training cancelled."
    exit 1
fi

echo ""
echo "=========================================="
echo "STARTING TRAINING"
echo "=========================================="
echo ""

# Run training
python train_baseline_fixed.py \
    --show-sample

# Check if training succeeded
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ TRAINING COMPLETE"
    echo "=========================================="
    echo ""

    # Find latest model
    LATEST_MODEL=$(ls -t outputs/ | grep baseline_fixed | head -1)

    if [ -n "$LATEST_MODEL" ]; then
        echo "Model saved to: outputs/$LATEST_MODEL"
        echo ""
        echo "Available checkpoints:"
        ls -la "outputs/$LATEST_MODEL/" | grep checkpoint
        echo ""
        echo "=========================================="
        echo "NEXT STEPS"
        echo "=========================================="
        echo ""
        echo "1. Test multiple checkpoints to find the best one:"
        echo ""
        echo "   python predict_baseline_fixed.py \\"
        echo "       --model \"outputs/$LATEST_MODEL/checkpoint-100\" \\"
        echo "       --test-file data/public_test.txt \\"
        echo "       --output evaluation_results/test_ckpt100.txt"
        echo ""
        echo "   python validate_vietnamese_output.py \\"
        echo "       --file evaluation_results/test_ckpt100.txt"
        echo ""
        echo "2. Repeat for checkpoint-200, 300, 400..."
        echo ""
        echo "3. Find checkpoint with HIGHEST valid AMR %"
        echo ""
        echo "4. Calculate SMATCH for best checkpoint"
        echo ""
        echo "5. Compare with MTUP model results"
        echo ""
    fi
else
    echo ""
    echo "❌ Training failed!"
    echo ""
    echo "Check logs for errors:"
    echo "  tail -100 logs/training_*.log"
    echo ""
    exit 1
fi
