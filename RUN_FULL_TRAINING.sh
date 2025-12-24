#!/bin/bash
################################################################################
# FULL MTUP TRAINING - Chạy trong tmux
# Settings đã được verify: batch=1, grad_accum=1, no OOM
################################################################################

echo "========================================================================"
echo "🚀 FULL MTUP TRAINING"
echo "========================================================================"
echo ""
echo "⚠️  This will run FULL training on entire dataset"
echo "   Estimated time: 3-6 hours"
echo "   Settings: batch_size=1, grad_accum=1 (verified no OOM)"
echo ""
read -p "Press Enter to continue or Ctrl+C to cancel..."
echo ""

# Activate environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate lora_py310

# CRITICAL: Uninstall bitsandbytes
echo "🗑️  Checking bitsandbytes..."
python3 -c "import bitsandbytes" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "⚠️  bitsandbytes found, uninstalling..."
    pip uninstall -y bitsandbytes 2>/dev/null || true
    conda uninstall -y bitsandbytes 2>/dev/null || true
    echo "✓ bitsandbytes removed"
else
    echo "✓ bitsandbytes not installed (good)"
fi
echo ""

# Set memory optimization
export PYTORCH_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
export CUDA_LAUNCH_BLOCKING=0

# Clear caches
python3 << 'EOF'
import torch
import gc

if torch.cuda.is_available():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    print("✓ All caches cleared")
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")

    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"✓ Total GPU memory: {total:.2f} GB")
EOF

echo ""
echo "========================================================================"
echo "Starting FULL TRAINING"
echo "========================================================================"
echo ""
echo "Settings:"
echo "  - Use case: full_training"
echo "  - Model: Qwen/Qwen2.5-3B-Instruct"
echo "  - Batch size: 1"
echo "  - Gradient accumulation: 1"
echo "  - Epochs: 10"
echo "  - CPU offload: enabled (20GB GPU + 30GB CPU)"
echo ""
echo "Training will be saved to:"
echo "  - Checkpoints: outputs/checkpoints_mtup/"
echo "  - Logs: outputs/logs/"
echo ""
echo "Monitor progress:"
echo "  - tensorboard --logdir outputs/logs"
echo ""

# Run full training
python3 train_mtup.py --use-case full_training --no-quantize \
  --batch-size 1 \
  --grad-accum 1

echo ""
echo "========================================================================"
echo "✅ TRAINING COMPLETED"
echo "========================================================================"
echo ""
echo "Next steps:"
echo "  1. Evaluate on test set:"
echo "     python3 evaluate_test_data.py"
echo ""
echo "  2. Check TensorBoard logs:"
echo "     tensorboard --logdir outputs/logs"
echo ""
echo "  3. Find best checkpoint in:"
echo "     ls -lh outputs/checkpoints_mtup/"
echo ""
echo "========================================================================"
