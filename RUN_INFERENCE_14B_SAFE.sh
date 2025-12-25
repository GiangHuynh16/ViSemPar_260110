#!/bin/bash
################################################################################
# SAFE 14B MODEL INFERENCE
# Automatically handles OOM by killing old processes and optimizing memory
################################################################################

echo "========================================================================"
echo "🚀 SAFE 14B MODEL INFERENCE"
echo "========================================================================"
echo ""

# Step 1: Kill any existing Python processes on GPU
echo "Step 1: Clearing GPU..."
echo ""

# Find Python processes using GPU
gpu_processes=$(nvidia-smi | grep python | awk '{print $5}')

if [ -n "$gpu_processes" ]; then
    echo "Found existing processes on GPU:"
    nvidia-smi | grep python
    echo ""
    echo "Killing processes..."

    for pid in $gpu_processes; do
        echo "  Killing PID $pid"
        kill -9 $pid 2>/dev/null || true
    done

    sleep 2
    echo "✅ Processes killed"
else
    echo "✅ No existing GPU processes found"
fi

echo ""

# Step 2: Verify GPU is clear
echo "Step 2: Checking GPU status..."
nvidia-smi
echo ""

free_mem=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
echo "Free GPU memory: ${free_mem} MiB"

if [ "$free_mem" -lt 10000 ]; then
    echo "⚠️  Warning: Less than 10GB free!"
    echo "Trying to clear more memory..."

    # Clear cache
    python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
    sleep 1

    free_mem=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    echo "Free GPU memory after clear: ${free_mem} MiB"
fi

echo ""

# Step 3: Set memory optimization environment variables
echo "Step 3: Setting memory optimization..."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=0
echo "✅ Environment configured"
echo ""

# Step 4: Activate conda
echo "Step 4: Activating environment..."
source ~/anaconda3/etc/profile.d/conda.sh
conda activate amr-parser
echo "✅ Environment activated"
echo ""

# Step 5: Run inference with memory-optimized script
echo "========================================================================"
echo "Starting inference with memory optimization..."
echo "========================================================================"
echo ""

# Configuration
MODEL_PATH="outputs/checkpoints/qwen2.5-14b-fine-tuned"
INPUT_FILE="data/public_test_sentences.txt"  # Adjust if needed
OUTPUT_FILE="outputs/predictions_14b_safe.json"
MAX_SAMPLES=150  # Limit to avoid long runs

echo "Configuration:"
echo "  Model: $MODEL_PATH"
echo "  Input: $INPUT_FILE"
echo "  Output: $OUTPUT_FILE"
echo "  Max samples: $MAX_SAMPLES"
echo ""

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "❌ Input file not found: $INPUT_FILE"
    echo ""
    echo "Please provide correct input file path."
    exit 1
fi

# Run with error handling
python3 inference_14b_memory_optimized.py \
    --model "$MODEL_PATH" \
    --input "$INPUT_FILE" \
    --output "$OUTPUT_FILE" \
    --max-samples "$MAX_SAMPLES" \
    2>&1 | tee outputs/inference_14b_safe.log

exit_code=${PIPESTATUS[0]}

echo ""
echo "========================================================================"

if [ $exit_code -eq 0 ]; then
    echo "✅ INFERENCE COMPLETED SUCCESSFULLY"
    echo "========================================================================"
    echo ""
    echo "Results saved to: $OUTPUT_FILE"
    echo "Log saved to: outputs/inference_14b_safe.log"
    echo ""
    echo "View results:"
    echo "  cat $OUTPUT_FILE | python3 -m json.tool | head -50"
else
    echo "❌ INFERENCE FAILED"
    echo "========================================================================"
    echo ""
    echo "Check log for errors:"
    echo "  tail -50 outputs/inference_14b_safe.log"
    echo ""

    # If OOM, suggest using smaller model
    if grep -q "OutOfMemoryError" outputs/inference_14b_safe.log; then
        echo "💡 SUGGESTION: 14B model is too large for this GPU"
        echo ""
        echo "Options:"
        echo "  1. Use 3B MTUP model instead:"
        echo "     bash RUN_FULL_EVALUATION_TMUX.sh"
        echo ""
        echo "  2. Reduce max_samples further (try --max-samples 50)"
        echo ""
        echo "  3. Use quantization (8-bit or 4-bit)"
        echo ""
    fi
fi

echo "========================================================================"
