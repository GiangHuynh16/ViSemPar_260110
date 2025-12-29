#!/bin/bash
# Script to start MTUP training with fixed template
# Run this on the SERVER, not local machine

set -e

echo "=========================================="
echo "START MTUP TRAINING (FIXED TEMPLATE)"
echo "=========================================="
echo ""
echo "Template fixes applied:"
echo "  ✅ Removed placeholder text '(biến / khái_niệm :quan_hệ ...)'"
echo "  ✅ Removed 'AMR cuối cùng:' header"
echo "  ✅ Model will output clean AMR without template leakage"
echo ""
echo "Training configuration:"
echo "  - Model: Qwen/Qwen2.5-3B-Instruct"
echo "  - Template: MTUP_TEMPLATE_V2_NATURAL (fixed)"
echo "  - Epochs: 15"
echo "  - Batch size: 4"
echo "  - Gradient accumulation: 4"
echo "  - LoRA rank: 64"
echo "  - Estimated time: ~9 hours"
echo ""

# Check if in tmux
if [ -z "$TMUX" ]; then
    echo "⚠️  WARNING: You are NOT in a tmux session!"
    echo ""
    echo "Training will take ~9 hours. It's recommended to run in tmux:"
    echo ""
    echo "  tmux new -s mtup_train"
    echo "  bash START_MTUP_TRAINING.sh"
    echo ""
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Cancelled. Please start tmux first."
        exit 1
    fi
else
    echo "✅ Running in tmux session: $TMUX"
fi

echo ""
echo "🚀 Starting training..."
echo ""

# Run training
python train_mtup.py --use_case reentrancy

echo ""
echo "=========================================="
echo "TRAINING COMPLETED"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Run evaluation:"
echo "     python scripts/evaluate_model.py \\"
echo "       --model_path models/mtup_reentrancy_final \\"
echo "       --test_file data/processed/vlsp_amr_v2_reentrancy_test.json \\"
echo "       --output_file results/evaluation/mtup_reentrancy_eval.json"
echo ""
echo "  2. Check F1 score:"
echo "     grep '\"f1\"' results/evaluation/mtup_reentrancy_eval.json"
echo ""
echo "  3. If successful, train baseline:"
echo "     tmux new -s baseline_train"
echo "     python train_baseline.py --use_case reentrancy"
echo ""
