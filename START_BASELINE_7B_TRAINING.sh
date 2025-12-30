#!/bin/bash
# Start Baseline training with Qwen 2.5 7B + LoRA rank 128
# NOTE: Extreme memory optimization to fit 7B in 24GB VRAM

set -e

echo "=========================================="
echo "BASELINE 7B TRAINING - SINGLE-TASK"
echo "=========================================="
echo ""
echo "⚠️  NOTE: Using EXTREME memory optimization"
echo "   MAX_SEQ_LENGTH reduced to 512 (from 2048)"
echo "   max_memory limited to 14GB GPU + 50GB CPU offload"
echo ""
echo "Baseline approach (Single-Task):"
echo "  Model: Qwen 2.5 7B"
echo "  LoRA rank: 128"
echo "  Trainable params: ~100M"
echo "  Task: Direct Sentence → AMR mapping"
echo "  Purpose: Fair comparison with MTUP 7B"
echo ""
echo "Training configuration:"
echo "  - Epochs: 15"
echo "  - Batch size: 1 (per device)"
echo "  - Gradient accumulation: 16"
echo "  - Effective batch size: 16"
echo "  - Max sequence length: 512 (reduced for memory)"
echo "  - Learning rate: 2e-4"
echo "  - Estimated time: ~15-18 hours"
echo "  - Peak VRAM usage: ~14 GB (tight fit)"
echo ""

# Check VRAM
echo "Checking GPU availability..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "✗ nvidia-smi not found"
    exit 1
fi

free_vram=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
free_gb=$(echo "scale=1; $free_vram/1024" | bc)

echo "  Free VRAM: ${free_gb} GB"

if (( $(echo "$free_gb < 20" | bc -l) )); then
    echo "  ⚠️  WARNING: Less than 20 GB free VRAM"
    echo "  Current free: ${free_gb} GB"
    echo "  Training will use tight memory margins with CPU offload"
    echo ""
fi

echo "  ✓ VRAM check passed"
echo ""

# Check if in tmux
if [ -z "$TMUX" ]; then
    echo "⚠️  WARNING: You are NOT in a tmux session!"
    echo ""
    echo "Training will take ~12-15 hours. Recommended to run in tmux:"
    echo ""
    echo "  tmux new -s baseline_7b"
    echo "  bash START_BASELINE_7B_TRAINING.sh"
    echo ""
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Cancelled. Please start tmux first."
        exit 1
    fi
else
    echo "✓ Running in tmux session: $TMUX"
fi

echo ""
echo "🚀 Starting 7B baseline training..."
echo ""

# Set memory optimizations
echo "Applying memory optimizations..."
export PYTORCH_ALLOC_CONF="max_split_size_mb:128,expandable_segments:True"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
echo "  ✓ PYTORCH_ALLOC_CONF set"
echo "  ✓ TOKENIZERS_PARALLELISM=false"
echo ""

# Clear GPU cache before training
python3 << 'EOF'
import torch
import gc
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()
    print("  ✓ GPU cache cleared")
EOF
echo ""

# Run training
python train_baseline.py --epochs 15 --show-sample

echo ""
echo "=========================================="
echo "TRAINING COMPLETED"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Run evaluation:"
echo "     python evaluate_baseline_model.py \\"
echo "       --checkpoint outputs/checkpoints/baseline_7b_final \\"
echo "       --test-file data/public_test_ground_truth.txt \\"
echo "       --output results/baseline_7b_evaluation.json"
echo ""
echo "  2. Check F1 score:"
echo "     cat results/baseline_7b_evaluation.json"
echo ""
echo "  3. Compare with MTUP 7B:"
echo "     echo 'MTUP 7B: F1 = [from previous evaluation]'"
echo "     echo 'Baseline 7B: F1 = [check results]'"
echo ""
echo "  4. If satisfactory, push to HuggingFace:"
echo "     hf upload YOUR-USERNAME/vietnamese-amr-baseline-7b \\"
echo "       outputs/checkpoints/baseline_7b_final"
echo ""
