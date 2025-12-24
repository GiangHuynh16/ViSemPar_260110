#!/bin/bash
################################################################################
# FIX OOM - RUN TRAINING WITH MEMORY OPTIMIZATION
# Copy-paste vào server và chạy
################################################################################

echo "========================================================================"
echo "🚀 RUNNING TRAINING WITH OOM FIX"
echo "========================================================================"
echo ""

# Activate environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate lora_py310

# Set memory optimization flags
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
export CUDA_LAUNCH_BLOCKING=0

# Clear GPU cache
python3 << 'EOF'
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print("✓ GPU cache cleared")
EOF

echo ""
echo "Memory optimization settings:"
echo "  - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128"
echo "  - Batch size: 1"
echo "  - Gradient accumulation: 4"
echo "  - Max samples: 50"
echo ""

# Run training with reduced settings
python3 train_mtup.py --use-case quick_test --show-sample --no-quantize \
  --batch-size 1 \
  --grad-accum 4 \
  --max-samples 50

echo ""
echo "========================================================================"
