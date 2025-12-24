#!/bin/bash
################################################################################
# COMPLETE DEPENDENCY FIX FOR VIETNAMESE AMR PARSER
# Copy-paste this entire script to server and run directly
# No need to pull/push code
################################################################################

set -e  # Exit on error

echo "========================================================================"
echo "FIXING ALL DEPENDENCIES FOR VIETNAMESE AMR PARSER"
echo "========================================================================"
echo ""

# Function to check command success
check_success() {
    if [ $? -eq 0 ]; then
        echo "✅ $1"
    else
        echo "❌ $1 FAILED"
        exit 1
    fi
}

# ============================================================================
# STEP 1: FIX PYTORCH AND TORCHVISION CUDA MISMATCH
# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 1: Fixing PyTorch/Torchvision CUDA Version Mismatch"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "Current PyTorch version:"
python3 -c "import torch; print(f'  PyTorch: {torch.__version__}')" 2>/dev/null || echo "  PyTorch not found"
echo ""

echo "🔧 Reinstalling PyTorch and torchvision with matching CUDA 11.8..."
pip uninstall -y torch torchvision torchaudio 2>/dev/null || true
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu118
check_success "PyTorch CUDA 11.8 installation"

echo ""
echo "Verifying PyTorch installation:"
python3 -c "import torch; print(f'  ✓ PyTorch {torch.__version__}'); print(f'  ✓ CUDA Available: {torch.cuda.is_available()}'); print(f'  ✓ CUDA Version: {torch.version.cuda}')"
echo ""

# ============================================================================
# STEP 2: FIX NUMPY VERSION
# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 2: Downgrading NumPy to 1.x"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "Current NumPy version:"
python3 -c "import numpy; print(f'  NumPy: {numpy.__version__}')" 2>/dev/null || echo "  NumPy not found"
echo ""

echo "🔧 Downgrading NumPy to <2.0..."
pip install "numpy>=1.24.0,<2.0.0" --upgrade
check_success "NumPy downgrade"

echo ""
echo "New NumPy version:"
python3 -c "import numpy; print(f'  ✓ NumPy {numpy.__version__}')"
echo ""

# ============================================================================
# STEP 3: FIX HUGGINGFACE HUB VERSION
# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 3: Downgrading HuggingFace Hub to <1.0"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "Current huggingface-hub version:"
python3 -c "import huggingface_hub; print(f'  HF Hub: {huggingface_hub.__version__}')" 2>/dev/null || echo "  HF Hub not found"
echo ""

echo "🔧 Downgrading huggingface-hub to <1.0..."
pip install "huggingface-hub>=0.24.0,<1.0" --force-reinstall
check_success "HuggingFace Hub downgrade"

echo ""
echo "New huggingface-hub version:"
python3 -c "import huggingface_hub; print(f'  ✓ HF Hub {huggingface_hub.__version__}')"
echo ""

# ============================================================================
# STEP 4: REINSTALL PANDAS AND SCIKIT-LEARN
# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 4: Reinstalling Pandas and Scikit-learn"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "🔧 Reinstalling pandas and scikit-learn for NumPy 1.x compatibility..."
pip install --force-reinstall pandas scikit-learn
check_success "Pandas and scikit-learn reinstall"

echo ""

# ============================================================================
# STEP 5: VERIFY ALL INSTALLATIONS
# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 5: Verifying All Installations"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "Testing critical imports..."
python3 << 'EOF'
import sys

def test_import(module_name, display_name=None):
    display_name = display_name or module_name
    try:
        if module_name == "torch":
            import torch
            print(f"✓ {display_name}: {torch.__version__} (CUDA: {torch.version.cuda})")
        elif module_name == "torchvision":
            import torchvision
            print(f"✓ {display_name}: {torchvision.__version__}")
        elif module_name == "numpy":
            import numpy
            print(f"✓ {display_name}: {numpy.__version__}")
        elif module_name == "pandas":
            import pandas
            print(f"✓ {display_name}: {pandas.__version__}")
        elif module_name == "sklearn":
            import sklearn
            print(f"✓ {display_name}: {sklearn.__version__}")
        elif module_name == "huggingface_hub":
            import huggingface_hub
            print(f"✓ {display_name}: {huggingface_hub.__version__}")
        elif module_name == "transformers":
            import transformers
            print(f"✓ {display_name}: {transformers.__version__}")
        elif module_name == "peft":
            import peft
            print(f"✓ {display_name}: {peft.__version__}")
        else:
            __import__(module_name)
            print(f"✓ {display_name}: OK")
        return True
    except Exception as e:
        print(f"✗ {display_name}: FAILED - {str(e)[:50]}")
        return False

print("")
all_ok = True
all_ok &= test_import("torch", "PyTorch")
all_ok &= test_import("torchvision", "Torchvision")
all_ok &= test_import("numpy", "NumPy")
all_ok &= test_import("pandas", "Pandas")
all_ok &= test_import("sklearn", "Scikit-learn")
all_ok &= test_import("huggingface_hub", "HF Hub")
all_ok &= test_import("transformers", "Transformers")
all_ok &= test_import("peft", "PEFT")
all_ok &= test_import("accelerate", "Accelerate")
all_ok &= test_import("bitsandbytes", "BitsAndBytes")

print("")
if all_ok:
    print("✅ ALL IMPORTS SUCCESSFUL")
    sys.exit(0)
else:
    print("❌ SOME IMPORTS FAILED")
    sys.exit(1)
EOF

check_success "All imports verification"

echo ""
echo "========================================================================"
echo "✅ ALL DEPENDENCIES FIXED SUCCESSFULLY"
echo "========================================================================"
echo ""
echo "Summary of fixes applied:"
echo "  ✓ PyTorch 2.3.0 + Torchvision 0.18.0 (CUDA 11.8)"
echo "  ✓ NumPy <2.0 (compatible with pandas/sklearn)"
echo "  ✓ HuggingFace Hub <1.0 (compatible with transformers)"
echo "  ✓ Pandas and Scikit-learn reinstalled"
echo ""
echo "Next steps:"
echo "  1. Test training:"
echo "     python3 train_mtup.py --use-case quick_test --show-sample"
echo ""
echo "  2. Full training (in tmux):"
echo "     tmux new -s amr-training"
echo "     python3 train_mtup.py --use-case full_training"
echo "     # Press Ctrl+B, then D to detach"
echo ""
echo "========================================================================"
