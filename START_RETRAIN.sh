#!/bin/bash
# Simple retrain script for server
# Run: bash START_RETRAIN.sh

echo "=========================================="
echo "RETRAIN BASELINE 7B - FIXED VERSION"
echo "=========================================="
echo ""

# Navigate to project directory
cd /mnt/nghiepth/giangha/visempar/ViSemPar_new1

echo "Step 1: Activate conda environment"
echo "=========================================="
source ~/miniconda3/etc/profile.d/conda.sh
conda activate baseline_final

echo "✓ Conda environment activated: baseline_final"
echo ""

echo "Step 2: Start training with fixes"
echo "=========================================="
echo ""
echo "Fixes applied:"
echo "  ✅ EOS token added"
echo "  ✅ Instruction masking enabled"
echo "  ✅ Clear Penman format prompt"
echo ""
echo "Training will take ~2-3 hours..."
echo "Press Ctrl+C to cancel, or wait 5 seconds to continue..."
sleep 5
echo ""

# Train
python train_baseline_fixed.py \
    --epochs 15 \
    --show-sample \
    --val-split 0.05

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Training failed! Check logs."
    exit 1
fi

echo ""
echo "=========================================="
echo "✅ TRAINING COMPLETE"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Find model in: outputs/baseline_fixed_*/"
echo "  2. Test with: python predict_baseline_fixed.py"
echo ""
