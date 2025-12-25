#!/bin/bash
################################################################################
# CHECK DISK USAGE
# Show disk usage for all model directories
################################################################################

echo "========================================================================"
echo "💾 DISK USAGE ANALYSIS"
echo "========================================================================"
echo ""

# Overall disk usage
echo "Overall disk space:"
df -h . | tail -1
echo ""

echo "========================================================================"
echo "MODEL DIRECTORIES"
echo "========================================================================"
echo ""

# Function to show directory size with details
show_dir_size() {
    local dir=$1
    local name=$2

    if [ -d "$dir" ]; then
        size=$(du -sh "$dir" 2>/dev/null | cut -f1)
        count=$(find "$dir" -type f 2>/dev/null | wc -l | tr -d ' ')
        echo "✓ $name"
        echo "  Path: $dir"
        echo "  Size: $size"
        echo "  Files: $count"
        echo ""
    else
        echo "✗ $name (not found)"
        echo "  Path: $dir"
        echo ""
    fi
}

# Check all model directories
show_dir_size "outputs/checkpoints/qwen2.5-14b-fine-tuned" "14B Model"
show_dir_size "outputs/checkpoints/qwen2.5-7b-fine-tuned" "7B Model"
show_dir_size "outputs/checkpoints_mtup" "3B MTUP Model"

echo "========================================================================"
echo "TOTAL BY CATEGORY"
echo "========================================================================"
echo ""

# Checkpoints total
if [ -d "outputs/checkpoints" ]; then
    checkpoint_size=$(du -sh outputs/checkpoints 2>/dev/null | cut -f1)
    echo "All checkpoints: $checkpoint_size"
    echo "  (includes 7B, 14B if present)"
fi

# MTUP total
if [ -d "outputs/checkpoints_mtup" ]; then
    mtup_size=$(du -sh outputs/checkpoints_mtup 2>/dev/null | cut -f1)
    echo "MTUP checkpoints: $mtup_size"
    echo "  (3B model - our primary)"
fi

# Other outputs
if [ -d "outputs" ]; then
    outputs_size=$(du -sh outputs 2>/dev/null | cut -f1)
    echo "Total outputs/: $outputs_size"
    echo "  (includes all models, logs, results)"
fi

echo ""

# Data directory
if [ -d "data" ]; then
    data_size=$(du -sh data 2>/dev/null | cut -f1)
    echo "Training data: $data_size"
fi

echo ""
echo "========================================================================"
echo "BREAKDOWN BY SIZE"
echo "========================================================================"
echo ""

# List largest directories in outputs/
if [ -d "outputs" ]; then
    echo "Largest directories in outputs/:"
    du -sh outputs/*/ 2>/dev/null | sort -hr | head -10
fi

echo ""
echo "========================================================================"
echo "RECOMMENDATION"
echo "========================================================================"
echo ""

# Check if 14B exists and recommend deletion
if [ -d "outputs/checkpoints/qwen2.5-14b-fine-tuned" ]; then
    size_14b=$(du -sh "outputs/checkpoints/qwen2.5-14b-fine-tuned" 2>/dev/null | cut -f1)
    echo "💡 14B model found ($size_14b)"
    echo ""
    echo "You can free up space by deleting it:"
    echo "  bash CLEANUP_MODELS.sh"
    echo ""
    echo "Safe to delete because:"
    echo "  - Causes OOM on 24GB GPU"
    echo "  - 3B MTUP model works better (F1=0.49)"
    echo "  - 7B model available if needed"
else
    echo "✓ 14B model not found (good - saves space)"
fi

echo ""
echo "========================================================================"
