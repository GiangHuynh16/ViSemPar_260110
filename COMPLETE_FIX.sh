#!/bin/bash
################################################################################
# COMPLETE DEPENDENCY FIX - CLEAN INSTALL
# Vietnamese AMR Parser - Server Setup
# Copy-paste toàn bộ script này vào server và chạy
################################################################################

echo "========================================================================"
echo "🔧 VIETNAMESE AMR PARSER - COMPLETE DEPENDENCY FIX"
echo "========================================================================"
echo ""
echo "⚠️  This will uninstall and reinstall all ML dependencies"
echo "⏱️  Estimated time: 5-10 minutes"
echo ""
read -p "Press Enter to continue or Ctrl+C to cancel..."
echo ""

# ============================================================================
# STEP 1: UNINSTALL ALL CONFLICTING PACKAGES
# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 1: Uninstalling conflicting packages"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "🗑️  Uninstalling PyTorch ecosystem..."
pip uninstall -y torch torchvision torchaudio 2>/dev/null || true

echo "🗑️  Uninstalling NumPy..."
pip uninstall -y numpy 2>/dev/null || true

echo "🗑️  Uninstalling pandas, scikit-learn..."
pip uninstall -y pandas scikit-learn 2>/dev/null || true

echo "🗑️  Uninstalling HuggingFace packages..."
pip uninstall -y transformers huggingface-hub peft accelerate 2>/dev/null || true

echo ""
echo "✅ Old packages uninstalled"
echo ""
sleep 2

# ============================================================================
# STEP 2: INSTALL NUMPY FIRST (CRITICAL ORDER)
# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 2: Installing NumPy 1.x (FIRST - critical for other packages)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "📦 Installing NumPy 1.26.4..."
pip install "numpy==1.26.4" --no-cache-dir

echo ""
echo "Verifying NumPy installation:"
python3 << 'EOF'
import numpy as np
print(f"  ✓ NumPy version: {np.__version__}")
print(f"  ✓ NumPy location: {np.__file__}")
# Test numpy.core.umath
from numpy.core import umath
print(f"  ✓ numpy.core.umath: OK")
EOF

if [ $? -ne 0 ]; then
    echo "❌ NumPy installation failed!"
    exit 1
fi

echo ""
echo "✅ NumPy installed successfully"
echo ""
sleep 2

# ============================================================================
# STEP 3: INSTALL PYTORCH WITH CORRECT CUDA VERSION
# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 3: Installing PyTorch with CUDA 11.8"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "📦 Installing PyTorch 2.3.0 + CUDA 11.8..."
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 \
    --index-url https://download.pytorch.org/whl/cu118 \
    --no-cache-dir

echo ""
echo "Verifying PyTorch installation:"
python3 << 'EOF'
import torch
import torchvision
print(f"  ✓ PyTorch: {torch.__version__}")
print(f"  ✓ Torchvision: {torchvision.__version__}")
print(f"  ✓ CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  ✓ CUDA version: {torch.version.cuda}")
    print(f"  ✓ GPU: {torch.cuda.get_device_name(0)}")
EOF

if [ $? -ne 0 ]; then
    echo "❌ PyTorch installation failed!"
    exit 1
fi

echo ""
echo "✅ PyTorch installed successfully"
echo ""
sleep 2

# ============================================================================
# STEP 4: INSTALL PANDAS AND SCIKIT-LEARN
# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 4: Installing Pandas and Scikit-learn"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "📦 Installing pandas and scikit-learn..."
pip install pandas>=2.0.0 scikit-learn>=1.3.0 --no-cache-dir

echo ""
echo "Verifying installations:"
python3 << 'EOF'
import pandas as pd
import sklearn
print(f"  ✓ Pandas: {pd.__version__}")
print(f"  ✓ Scikit-learn: {sklearn.__version__}")
EOF

if [ $? -ne 0 ]; then
    echo "❌ Pandas/Scikit-learn installation failed!"
    exit 1
fi

echo ""
echo "✅ Pandas and Scikit-learn installed successfully"
echo ""
sleep 2

# ============================================================================
# STEP 5: INSTALL HUGGINGFACE ECOSYSTEM
# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 5: Installing HuggingFace ecosystem"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "📦 Installing huggingface-hub <1.0..."
pip install "huggingface-hub>=0.24.0,<1.0" --no-cache-dir

echo "📦 Installing transformers..."
pip install transformers==4.46.3 --no-cache-dir

echo "📦 Installing accelerate..."
pip install accelerate==1.2.1 --no-cache-dir

echo "📦 Installing peft..."
pip install peft==0.13.2 --no-cache-dir

echo "📦 Installing bitsandbytes..."
pip install bitsandbytes==0.44.1 --no-cache-dir

echo ""
echo "Verifying HuggingFace installations:"
python3 << 'EOF'
import huggingface_hub
import transformers
import accelerate
import peft
import bitsandbytes
print(f"  ✓ HuggingFace Hub: {huggingface_hub.__version__}")
print(f"  ✓ Transformers: {transformers.__version__}")
print(f"  ✓ Accelerate: {accelerate.__version__}")
print(f"  ✓ PEFT: {peft.__version__}")
print(f"  ✓ BitsAndBytes: {bitsandbytes.__version__}")
EOF

if [ $? -ne 0 ]; then
    echo "❌ HuggingFace packages installation failed!"
    exit 1
fi

echo ""
echo "✅ HuggingFace ecosystem installed successfully"
echo ""
sleep 2

# ============================================================================
# STEP 6: INSTALL REMAINING DEPENDENCIES
# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 6: Installing remaining dependencies"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "📦 Installing datasets, tqdm, penman, smatch, python-dotenv..."
pip install datasets>=2.14.0 tqdm>=4.65.0 penman>=1.2.0 smatch>=1.0.4 python-dotenv>=1.0.0 --no-cache-dir

echo ""
echo "✅ Additional dependencies installed"
echo ""

# ============================================================================
# STEP 7: FINAL VERIFICATION
# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 7: Final verification - Testing all imports"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

python3 << 'EOF'
import sys

print("Testing all critical imports...\n")

try:
    # Core ML
    import numpy as np
    print(f"✓ NumPy: {np.__version__}")

    import torch
    print(f"✓ PyTorch: {torch.__version__} (CUDA {torch.version.cuda})")

    import torchvision
    print(f"✓ Torchvision: {torchvision.__version__}")

    # Data processing
    import pandas as pd
    print(f"✓ Pandas: {pd.__version__}")

    import sklearn
    print(f"✓ Scikit-learn: {sklearn.__version__}")

    # HuggingFace
    import huggingface_hub
    print(f"✓ HuggingFace Hub: {huggingface_hub.__version__}")

    import transformers
    print(f"✓ Transformers: {transformers.__version__}")

    import accelerate
    print(f"✓ Accelerate: {accelerate.__version__}")

    import peft
    print(f"✓ PEFT: {peft.__version__}")

    import bitsandbytes
    print(f"✓ BitsAndBytes: {bitsandbytes.__version__}")

    # Other
    import datasets
    print(f"✓ Datasets: {datasets.__version__}")

    import penman
    print(f"✓ Penman: {penman.__version__}")

    import smatch
    print(f"✓ SMATCH: OK")

    print("\n" + "="*70)
    print("✅ ALL IMPORTS SUCCESSFUL!")
    print("="*70)
    sys.exit(0)

except Exception as e:
    print(f"\n❌ Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Some imports failed. Please check the error above."
    exit 1
fi

# ============================================================================
# SUCCESS
# ============================================================================
echo ""
echo "========================================================================"
echo "✅ ALL DEPENDENCIES INSTALLED SUCCESSFULLY"
echo "========================================================================"
echo ""
echo "📋 Summary:"
echo "  ✓ NumPy 1.26.4 (compatible with all packages)"
echo "  ✓ PyTorch 2.3.0 + Torchvision 0.18.0 (CUDA 11.8)"
echo "  ✓ Pandas + Scikit-learn (NumPy 1.x compatible)"
echo "  ✓ HuggingFace Hub <1.0 (Transformers compatible)"
echo "  ✓ Transformers 4.46.3 + PEFT + Accelerate"
echo "  ✓ All other dependencies"
echo ""
echo "🚀 Next steps:"
echo ""
echo "  1. Test with quick training:"
echo "     python3 train_mtup.py --use-case quick_test --show-sample"
echo ""
echo "  2. Run full training (in tmux):"
echo "     tmux new -s amr-training"
echo "     python3 train_mtup.py --use-case full_training"
echo "     # Press Ctrl+B, then D to detach"
echo ""
echo "  3. Reattach to tmux session:"
echo "     tmux attach -t amr-training"
echo ""
echo "========================================================================"
