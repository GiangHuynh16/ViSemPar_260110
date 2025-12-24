#!/bin/bash
################################################################################
# UNINSTALL BITSANDBYTES COMPLETELY
# PEFT tries to import bitsandbytes even when not using quantization
################################################################################

echo "========================================================================"
echo "🗑️  UNINSTALLING BITSANDBYTES COMPLETELY"
echo "========================================================================"
echo ""

# Activate environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate lora_py310

echo "Uninstalling bitsandbytes..."
pip uninstall -y bitsandbytes 2>/dev/null || true
conda uninstall -y bitsandbytes 2>/dev/null || true

echo ""
echo "Verifying bitsandbytes is removed..."
python3 << 'EOF'
try:
    import bitsandbytes
    print("❌ bitsandbytes still installed")
    exit(1)
except ImportError:
    print("✓ bitsandbytes successfully removed")
    exit(0)
EOF

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================================================"
    echo "✅ BITSANDBYTES UNINSTALLED"
    echo "========================================================================"
    echo ""
    echo "Now you can run training with --no-quantize:"
    echo "  bash RUN_TRAINING_MINIMAL.sh"
    echo ""
else
    echo ""
    echo "❌ Failed to remove bitsandbytes. Manual cleanup needed."
    echo ""
    echo "Try:"
    echo "  pip list | grep bitsandbytes"
    echo "  pip uninstall -y bitsandbytes"
    echo "  conda list | grep bitsandbytes"
    echo "  conda uninstall -y bitsandbytes"
    exit 1
fi
