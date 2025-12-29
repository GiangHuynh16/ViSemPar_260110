#!/bin/bash
# Script to remove failed MTUP model training outputs
# Model failed due to template leakage issue (outputting placeholder text)

set -e

echo "=========================================="
echo "CLEANUP FAILED MODEL TRAINING"
echo "=========================================="
echo ""
echo "This will remove:"
echo "  - models/mtup_reentrancy_final/"
echo "  - models/checkpoints/mtup_reentrancy/"
echo "  - results/evaluation/mtup_reentrancy_eval.json"
echo ""

read -p "Are you sure? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 1
fi

echo ""
echo "🗑️  Removing failed model outputs..."
echo ""

# Remove final model
if [ -d "models/mtup_reentrancy_final" ]; then
    SIZE=$(du -sh models/mtup_reentrancy_final | cut -f1)
    echo "  - Removing models/mtup_reentrancy_final/ ($SIZE)"
    rm -rf models/mtup_reentrancy_final
else
    echo "  - models/mtup_reentrancy_final/ not found (already removed)"
fi

# Remove checkpoints
if [ -d "models/checkpoints/mtup_reentrancy" ]; then
    SIZE=$(du -sh models/checkpoints/mtup_reentrancy | cut -f1)
    echo "  - Removing models/checkpoints/mtup_reentrancy/ ($SIZE)"
    rm -rf models/checkpoints/mtup_reentrancy
else
    echo "  - models/checkpoints/mtup_reentrancy/ not found (already removed)"
fi

# Remove evaluation results
if [ -f "results/evaluation/mtup_reentrancy_eval.json" ]; then
    echo "  - Removing results/evaluation/mtup_reentrancy_eval.json"
    rm -f results/evaluation/mtup_reentrancy_eval.json
else
    echo "  - results/evaluation/mtup_reentrancy_eval.json not found (already removed)"
fi

echo ""
echo "✅ Cleanup completed!"
echo ""
echo "📊 Disk space freed:"
du -sh models/ 2>/dev/null || echo "  No models directory found"
echo ""
echo "Next steps:"
echo "  1. Verify template fix in config/prompt_templates.py"
echo "  2. Re-run training: bash scripts/run_training_mtup.sh"
echo "  3. Training will take ~9 hours"
echo ""
