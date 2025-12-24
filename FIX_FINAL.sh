#!/bin/bash
################################################################################
# FIX CUỐI CÙNG - CLEAN INSTALL TOÀN BỘ
# Chạy trong conda environment lora_py310
# Copy-paste toàn bộ và chạy trực tiếp trên server
################################################################################

echo "========================================================================"
echo "🔧 FIX HOÀN TOÀN - CLEAN INSTALL"
echo "========================================================================"
echo ""
echo "Environment: lora_py310"
echo "Thời gian: ~10 phút"
echo ""
echo "⚠️  Script sẽ:"
echo "  1. Xóa toàn bộ PyTorch, NumPy cũ"
echo "  2. Cài đặt lại từ đầu với đúng version"
echo "  3. Fix NCCL conflict"
echo ""
read -p "Nhấn Enter để tiếp tục hoặc Ctrl+C để hủy..." dummy
echo ""

# Activate environment
echo "📦 Activating conda environment lora_py310..."
source ~/anaconda3/etc/profile.d/conda.sh
conda activate lora_py310

if [ $? -ne 0 ]; then
    echo "❌ Không thể activate environment lora_py310"
    echo "Chạy thủ công: conda activate lora_py310"
    exit 1
fi

echo "✅ Environment activated: $(which python)"
echo ""

# ============================================================================
# BƯỚC 1: XÓA SẠCH CÁC PACKAGE CŨ
# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "BƯỚC 1: Xóa toàn bộ packages cũ"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "🗑️  Xóa PyTorch + torchvision + torchaudio..."
pip uninstall -y torch torchvision torchaudio 2>/dev/null || true
conda uninstall -y pytorch torchvision torchaudio 2>/dev/null || true

echo "🗑️  Xóa NumPy..."
pip uninstall -y numpy 2>/dev/null || true

echo "🗑️  Xóa pandas, scikit-learn..."
pip uninstall -y pandas scikit-learn 2>/dev/null || true

echo "🗑️  Xóa transformers, huggingface-hub, peft..."
pip uninstall -y transformers huggingface-hub peft accelerate bitsandbytes 2>/dev/null || true

echo ""
echo "✅ Đã xóa sạch packages cũ"
echo ""
sleep 2

# ============================================================================
# BƯỚC 2: CÀI NUMPY TRƯỚC (QUAN TRỌNG)
# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "BƯỚC 2: Cài NumPy 1.26.4 (TRƯỚC TIÊN)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

pip install "numpy==1.26.4" --no-cache-dir

python << 'PYEOF'
import numpy as np
print(f"  ✓ NumPy: {np.__version__}")
from numpy.core import umath
print(f"  ✓ numpy.core.umath: OK")
PYEOF

if [ $? -ne 0 ]; then
    echo "❌ NumPy cài đặt thất bại!"
    exit 1
fi

echo ""
echo "✅ NumPy OK"
echo ""
sleep 1

# ============================================================================
# BƯỚC 3: CÀI PYTORCH + FIX NCCL
# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "BƯỚC 3: Cài PyTorch 2.3.0 CUDA 11.8 + Fix NCCL"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "📦 Cài PyTorch từ conda-forge (khuyến nghị cho stability)..."
conda install -y pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=11.8 -c pytorch -c nvidia

if [ $? -ne 0 ]; then
    echo "⚠️  Conda install thất bại, thử pip install..."
    pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 \
        --index-url https://download.pytorch.org/whl/cu118 \
        --no-cache-dir
fi

echo ""
echo "🔧 Fix NCCL conflict..."
# Reinstall NCCL if needed
conda install -y nccl -c conda-forge 2>/dev/null || true

echo ""
echo "Kiểm tra PyTorch:"
python << 'PYEOF'
import torch
print(f"  ✓ PyTorch: {torch.__version__}")
print(f"  ✓ CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  ✓ CUDA version: {torch.version.cuda}")
    print(f"  ✓ GPU count: {torch.cuda.device_count()}")
    print(f"  ✓ GPU 0: {torch.cuda.get_device_name(0)}")
PYEOF

if [ $? -ne 0 ]; then
    echo "❌ PyTorch import thất bại!"
    exit 1
fi

echo ""
echo "✅ PyTorch OK"
echo ""
sleep 1

# ============================================================================
# BƯỚC 4: CÀI PANDAS, SCIKIT-LEARN
# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "BƯỚC 4: Cài Pandas, Scikit-learn"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

pip install pandas scikit-learn --no-cache-dir

python << 'PYEOF'
import pandas as pd
import sklearn
print(f"  ✓ Pandas: {pd.__version__}")
print(f"  ✓ Scikit-learn: {sklearn.__version__}")
PYEOF

echo ""
echo "✅ Pandas, Scikit-learn OK"
echo ""
sleep 1

# ============================================================================
# BƯỚC 5: CÀI HUGGINGFACE ECOSYSTEM
# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "BƯỚC 5: Cài HuggingFace ecosystem"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

pip install "huggingface-hub>=0.24.0,<1.0" --no-cache-dir
pip install transformers==4.46.3 --no-cache-dir
pip install accelerate==1.2.1 --no-cache-dir
pip install peft==0.13.2 --no-cache-dir
pip install bitsandbytes==0.44.1 --no-cache-dir

python << 'PYEOF'
import huggingface_hub
import transformers
import accelerate
import peft
import bitsandbytes
print(f"  ✓ HF Hub: {huggingface_hub.__version__}")
print(f"  ✓ Transformers: {transformers.__version__}")
print(f"  ✓ Accelerate: {accelerate.__version__}")
print(f"  ✓ PEFT: {peft.__version__}")
print(f"  ✓ BitsAndBytes: {bitsandbytes.__version__}")
PYEOF

echo ""
echo "✅ HuggingFace ecosystem OK"
echo ""
sleep 1

# ============================================================================
# BƯỚC 6: CÀI CÁC PACKAGE KHÁC
# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "BƯỚC 6: Cài các packages khác"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

pip install datasets tqdm penman smatch python-dotenv tensorboard --no-cache-dir

echo "✅ Dependencies bổ sung OK"
echo ""

# ============================================================================
# BƯỚC 7: KIỂM TRA TOÀN BỘ
# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "BƯỚC 7: Kiểm tra toàn bộ imports"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

python << 'PYEOF'
import sys
print("Kiểm tra tất cả imports...\n")

all_ok = True

try:
    import numpy as np
    print(f"✓ NumPy: {np.__version__}")
except Exception as e:
    print(f"✗ NumPy: {e}")
    all_ok = False

try:
    import torch
    print(f"✓ PyTorch: {torch.__version__} (CUDA {torch.version.cuda})")
except Exception as e:
    print(f"✗ PyTorch: {e}")
    all_ok = False

try:
    import torchvision
    print(f"✓ Torchvision: {torchvision.__version__}")
except Exception as e:
    print(f"✗ Torchvision: {e}")
    all_ok = False

try:
    import pandas as pd
    print(f"✓ Pandas: {pd.__version__}")
except Exception as e:
    print(f"✗ Pandas: {e}")
    all_ok = False

try:
    import sklearn
    print(f"✓ Scikit-learn: {sklearn.__version__}")
except Exception as e:
    print(f"✗ Scikit-learn: {e}")
    all_ok = False

try:
    import huggingface_hub
    print(f"✓ HF Hub: {huggingface_hub.__version__}")
except Exception as e:
    print(f"✗ HF Hub: {e}")
    all_ok = False

try:
    import transformers
    print(f"✓ Transformers: {transformers.__version__}")
except Exception as e:
    print(f"✗ Transformers: {e}")
    all_ok = False

try:
    import peft
    print(f"✓ PEFT: {peft.__version__}")
except Exception as e:
    print(f"✗ PEFT: {e}")
    all_ok = False

try:
    import accelerate
    print(f"✓ Accelerate: {accelerate.__version__}")
except Exception as e:
    print(f"✗ Accelerate: {e}")
    all_ok = False

try:
    import bitsandbytes
    print(f"✓ BitsAndBytes: {bitsandbytes.__version__}")
except Exception as e:
    print(f"✗ BitsAndBytes: {e}")
    all_ok = False

print("")
if all_ok:
    print("="*70)
    print("✅ TẤT CẢ IMPORTS THÀNH CÔNG!")
    print("="*70)
    sys.exit(0)
else:
    print("="*70)
    print("❌ MỘT SỐ IMPORTS THẤT BẠI")
    print("="*70)
    sys.exit(1)
PYEOF

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Vẫn còn lỗi. Vui lòng kiểm tra output phía trên."
    exit 1
fi

# ============================================================================
# HOÀN THÀNH
# ============================================================================
echo ""
echo "========================================================================"
echo "✅ HOÀN TẤT - MÔI TRƯỜNG ĐÃ SẠCH!"
echo "========================================================================"
echo ""
echo "📋 Đã cài đặt:"
echo "  ✓ NumPy 1.26.4"
echo "  ✓ PyTorch 2.3.0 CUDA 11.8"
echo "  ✓ Torchvision 0.18.0"
echo "  ✓ Pandas + Scikit-learn"
echo "  ✓ HuggingFace Hub <1.0"
echo "  ✓ Transformers 4.46.3"
echo "  ✓ PEFT, Accelerate, BitsAndBytes"
echo "  ✓ NCCL conflict đã fix"
echo ""
echo "🚀 Bây giờ chạy training:"
echo ""
echo "  python train_mtup.py --use-case quick_test --show-sample"
echo ""
echo "🎯 Full training (trong tmux):"
echo ""
echo "  tmux new -s amr"
echo "  python train_mtup.py --use-case full_training"
echo "  # Nhấn Ctrl+B rồi D để detach"
echo ""
echo "  # Attach lại:"
echo "  tmux attach -t amr"
echo ""
echo "========================================================================"
