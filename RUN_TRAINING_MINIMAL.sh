#!/bin/bash
################################################################################
# MINIMAL TRAINING - CHỈ 25 SAMPLES, BATCH=1, GRAD_ACCUM=1
# Nếu RUN_TRAINING_OOM_FIX.sh vẫn bị OOM, dùng script này
################################################################################

echo "========================================================================"
echo "🚀 MINIMAL TRAINING MODE - EMERGENCY OOM FIX"
echo "========================================================================"
echo ""
echo "⚠️  This uses MINIMAL settings to avoid OOM:"
echo "  - 25 samples only"
echo "  - Batch size: 1"
echo "  - Gradient accumulation: 1 (no accumulation)"
echo "  - CPU offload enabled"
echo ""

# Activate environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate lora_py310

# CRITICAL: Uninstall bitsandbytes to avoid import errors
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

# Clear ALL caches
python3 << 'EOF'
import torch
import gc

if torch.cuda.is_available():
    # Clear Python garbage
    gc.collect()

    # Clear CUDA cache
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    print("✓ All caches cleared")
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")

    # Show memory stats
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    reserved = torch.cuda.memory_reserved(0) / 1024**3
    allocated = torch.cuda.memory_allocated(0) / 1024**3
    free = total - reserved

    print(f"✓ Total: {total:.2f} GB")
    print(f"✓ Free: {free:.2f} GB")
    print(f"✓ Reserved: {reserved:.2f} GB")
    print(f"✓ Allocated: {allocated:.2f} GB")
EOF

echo ""
echo "Starting MINIMAL training..."
echo ""

# Run with absolute minimal settings
python3 train_mtup.py --use-case quick_test --show-sample --no-quantize \
  --batch-size 1 \
  --grad-accum 1 \
  --max-samples 25

echo ""
echo "========================================================================"
echo "Training completed!"
echo ""
echo "If this worked, you can gradually increase:"
echo "  1. max-samples to 50"
echo "  2. grad-accum to 2"
echo "  3. grad-accum to 4"
echo "========================================================================"
