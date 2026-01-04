#!/bin/bash

# TRAIN_MTUP_FIXED.sh
# Training wrapper script for MTUP Fixed
# Applies all lessons learned from baseline success

set -e

echo "================================================================================"
echo "MTUP FIXED TRAINING"
echo "================================================================================"
echo ""
echo "🎯 Key Improvements from Baseline:"
echo "  ✅ Instruction masking (train only on final AMR)"
echo "  ✅ Minimal prompt with Penman examples"
echo "  ✅ 2 epochs (prevent overfitting)"
echo "  ✅ Save every 100 steps (early stopping)"
echo "  ✅ bfloat16 precision"
echo ""
echo "📊 Expected Results:"
echo "  - Structural validity: >90% (baseline achieved 91.3%)"
echo "  - SMATCH F1: ~0.50-0.55 (hypothesis: MTUP improves over baseline's 0.47)"
echo "  - Training time: ~4 hours (2-stage model, 2 epochs)"
echo ""

# Check if data exists
if [ ! -f "data/train_amr_mtup_preprocessed.txt" ]; then
    echo "⚠️  MTUP preprocessed data not found!"
    echo ""
    echo "Please preprocess the data first:"
    echo "  python3 preprocess_mtup.py"
    echo ""
    exit 1
fi

echo "📦 Training Data:"
ls -lh data/train_amr_mtup_preprocessed.txt
echo ""

# GPU check
if command -v nvidia-smi &> /dev/null; then
    echo "🖥️  GPU Status:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
    echo ""
else
    echo "⚠️  nvidia-smi not found, cannot check GPU status"
    echo ""
fi

# Confirm before starting
echo "⏰ Estimated training time: ~4 hours"
echo ""
read -p "🚀 Start training? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Training cancelled."
    exit 0
fi

echo ""
echo "================================================================================"
echo "STARTING TRAINING"
echo "================================================================================"
echo ""

# Create timestamp for output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="outputs/mtup_fixed_${TIMESTAMP}"

echo "📁 Output directory: $OUTPUT_DIR"
echo ""

# Run training
python3 train_mtup_fixed.py \
    --output_dir "$OUTPUT_DIR" \
    2>&1 | tee "logs/training_mtup_fixed_${TIMESTAMP}.log"

echo ""
echo "================================================================================"
echo "✅ TRAINING COMPLETE"
echo "================================================================================"
echo ""
echo "📊 Next Steps:"
echo ""
echo "1. Evaluate all checkpoints on validation set:"
echo "   bash EVALUATE_MTUP_CHECKPOINTS.sh $OUTPUT_DIR"
echo ""
echo "2. Test on public test set (150 samples):"
echo "   python3 predict_mtup_fixed.py \\"
echo "       --model $OUTPUT_DIR/checkpoint-XXXX \\"
echo "       --test-file data/public_test.txt \\"
echo "       --output evaluation_results/mtup_fixed_predictions.txt"
echo ""
echo "3. Calculate SMATCH:"
echo "   python3 filter_valid_amrs.py \\"
echo "       --predictions evaluation_results/mtup_fixed_predictions.txt \\"
echo "       --ground-truth data/public_test_ground_truth.txt \\"
echo "       --output-pred evaluation_results/mtup_valid.txt \\"
echo "       --output-gold evaluation_results/gold_valid.txt"
echo ""
echo "   python -m smatch -f \\"
echo "       evaluation_results/mtup_valid.txt \\"
echo "       evaluation_results/gold_valid.txt \\"
echo "       --significant 4"
echo ""
echo "4. Compare with Baseline:"
echo "   - Baseline F1: 0.47 (91.3% validity)"
echo "   - MTUP F1: ??? (target: >0.50)"
echo ""
