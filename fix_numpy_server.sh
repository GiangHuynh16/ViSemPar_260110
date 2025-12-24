#!/bin/bash
################################################################################
# Fix NumPy Compatibility on Server
# Downgrades NumPy to 1.x for compatibility with pandas/sklearn
################################################################################

echo "========================================================================"
echo "FIX NUMPY COMPATIBILITY"
echo "========================================================================"
echo ""

# Check current NumPy version
echo "📦 Current NumPy version:"
python3 -c "import numpy; print(f'   NumPy {numpy.__version__}')" 2>/dev/null || echo "   NumPy not found"
echo ""

# Downgrade NumPy
echo "🔧 Downgrading NumPy to 1.x for compatibility..."
pip install "numpy<2.0.0" --upgrade

echo ""
echo "✅ NumPy downgrade complete"
echo ""

# Verify
echo "📦 New NumPy version:"
python3 -c "import numpy; print(f'   NumPy {numpy.__version__}')"
echo ""

# Reinstall dependencies to ensure compatibility
echo "🔄 Reinstalling pandas and scikit-learn for compatibility..."
pip install --force-reinstall pandas scikit-learn

echo ""
echo "🔄 Fixing huggingface-hub version conflict..."
pip install "huggingface-hub>=0.24.0,<1.0" --force-reinstall

echo ""
echo "========================================================================"
echo "✅ NUMPY & DEPENDENCIES FIX COMPLETE"
echo "========================================================================"
echo ""
echo "Verify installations:"
python3 -c "import numpy; print(f'✓ NumPy {numpy.__version__}')"
python3 -c "import pandas; print('✓ pandas OK')"
python3 -c "import sklearn; print('✓ sklearn OK')"
python3 -c "import huggingface_hub; print(f'✓ huggingface-hub {huggingface_hub.__version__}')"

echo ""
echo "Now you can run training:"
echo "  python3 train_mtup.py --use-case quick_test"
echo ""
