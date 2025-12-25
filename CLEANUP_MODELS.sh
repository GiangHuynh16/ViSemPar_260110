#!/bin/bash
################################################################################
# CLEANUP OLD MODELS
# Remove 14B model, keep 7B for potential future use
################################################################################

echo "========================================================================"
echo "🧹 MODEL CLEANUP"
echo "========================================================================"
echo ""

# Define paths
MODEL_14B="outputs/checkpoints/qwen2.5-14b-fine-tuned"
MODEL_7B="outputs/checkpoints/qwen2.5-7b-fine-tuned"
MTUP_3B="outputs/checkpoints_mtup"

echo "Scanning for models..."
echo ""

# Check what exists
if [ -d "$MODEL_14B" ]; then
    size_14b=$(du -sh "$MODEL_14B" 2>/dev/null | cut -f1)
    echo "✓ Found 14B model: $MODEL_14B ($size_14b)"
    FOUND_14B=true
else
    echo "✗ 14B model not found: $MODEL_14B"
    FOUND_14B=false
fi

if [ -d "$MODEL_7B" ]; then
    size_7b=$(du -sh "$MODEL_7B" 2>/dev/null | cut -f1)
    echo "✓ Found 7B model: $MODEL_7B ($size_7b)"
    FOUND_7B=true
else
    echo "✗ 7B model not found: $MODEL_7B"
    FOUND_7B=false
fi

if [ -d "$MTUP_3B" ]; then
    size_3b=$(du -sh "$MTUP_3B" 2>/dev/null | cut -f1)
    echo "✓ Found 3B MTUP model: $MTUP_3B ($size_3b)"
    FOUND_3B=true
else
    echo "✗ 3B MTUP model not found: $MTUP_3B"
    FOUND_3B=false
fi

echo ""
echo "========================================================================"
echo "CLEANUP PLAN"
echo "========================================================================"
echo ""

if [ "$FOUND_14B" = true ]; then
    echo "Will DELETE:"
    echo "  ❌ 14B model ($size_14b)"
    echo "     Path: $MODEL_14B"
    echo "     Reason: Too large, causes OOM, not used"
else
    echo "No 14B model to delete"
fi

echo ""
echo "Will KEEP:"

if [ "$FOUND_7B" = true ]; then
    echo "  ✅ 7B model ($size_7b)"
    echo "     Path: $MODEL_7B"
    echo "     Reason: Good balance of size/performance"
else
    echo "  ⚠️  7B model not found (but would keep if existed)"
fi

if [ "$FOUND_3B" = true ]; then
    echo "  ✅ 3B MTUP model ($size_3b)"
    echo "     Path: $MTUP_3B"
    echo "     Reason: Primary model, tested, F1=0.49"
else
    echo "  ⚠️  3B MTUP model not found (CRITICAL - this is our main model!)"
fi

echo ""
echo "========================================================================"

# Calculate space to be freed
if [ "$FOUND_14B" = true ]; then
    space_freed=$(du -sb "$MODEL_14B" 2>/dev/null | cut -f1)
    space_freed_gb=$(echo "scale=2; $space_freed / 1024 / 1024 / 1024" | bc 2>/dev/null || echo "N/A")

    echo ""
    echo "Space to be freed: ~${space_freed_gb} GB"
    echo ""
fi

# Confirmation
if [ "$FOUND_14B" = true ]; then
    echo "⚠️  WARNING: This will permanently delete the 14B model!"
    echo ""
    read -p "Type 'yes' to confirm deletion: " confirmation

    if [ "$confirmation" = "yes" ]; then
        echo ""
        echo "Deleting 14B model..."

        # Create backup list of what's being deleted (just filenames for reference)
        echo "Creating deletion log..."
        ls -lh "$MODEL_14B" > "outputs/deleted_14b_model.log" 2>/dev/null

        # Delete
        rm -rf "$MODEL_14B"

        if [ ! -d "$MODEL_14B" ]; then
            echo "✅ 14B model deleted successfully!"
            echo ""
            echo "Deletion log saved to: outputs/deleted_14b_model.log"
        else
            echo "❌ Failed to delete 14B model"
            exit 1
        fi
    else
        echo ""
        echo "❌ Deletion cancelled (you did not type 'yes')"
        exit 0
    fi
else
    echo "Nothing to delete!"
fi

echo ""
echo "========================================================================"
echo "FINAL STATUS"
echo "========================================================================"
echo ""

# Show remaining models
echo "Remaining models in outputs/checkpoints/:"
ls -lh outputs/checkpoints/ 2>/dev/null || echo "(directory not found)"

echo ""
echo "Remaining models in outputs/checkpoints_mtup/:"
ls -lh outputs/checkpoints_mtup/ 2>/dev/null || echo "(directory not found)"

echo ""

# Show disk usage
echo "Total disk usage for model directories:"
du -sh outputs/checkpoints* 2>/dev/null

echo ""
echo "========================================================================"
echo "✅ CLEANUP COMPLETE"
echo "========================================================================"
echo ""

if [ "$FOUND_7B" = true ] && [ "$FOUND_3B" = true ]; then
    echo "Status: ✅ Good!"
    echo "  - 7B model: Available for future use"
    echo "  - 3B MTUP: Primary model (F1=0.49)"
elif [ "$FOUND_3B" = true ]; then
    echo "Status: ✅ OK"
    echo "  - 3B MTUP: Primary model (F1=0.49)"
    echo "  - 7B model: Not found (optional)"
else
    echo "Status: ⚠️  WARNING"
    echo "  - 3B MTUP model not found!"
    echo "  - This is your main working model"
    echo "  - Check: outputs/checkpoints_mtup/"
fi

echo ""
echo "========================================================================"
