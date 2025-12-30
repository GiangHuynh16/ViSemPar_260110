#!/bin/bash
# Check current Python environment and package versions

echo "=========================================="
echo "ENVIRONMENT CHECK"
echo "=========================================="
echo ""

# Check Python
echo "Python:"
which python
python --version
echo ""

# Check conda
echo "Conda environment:"
if command -v conda &> /dev/null; then
    echo "Conda: $(conda --version)"
    echo "Current env: $CONDA_DEFAULT_ENV"
    conda env list | grep "*"
else
    echo "Conda not found"
fi
echo ""

# Check virtualenv
echo "Virtual environment:"
if [ -n "$VIRTUAL_ENV" ]; then
    echo "Active venv: $VIRTUAL_ENV"
else
    echo "No venv active"
fi
echo ""

# Check key packages
echo "Key package versions:"
python -c "import torch; print(f'PyTorch: {torch.__version__}')" 2>/dev/null || echo "PyTorch: NOT INSTALLED"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')" 2>/dev/null || echo "Transformers: NOT INSTALLED"
python -c "import peft; print(f'PEFT: {peft.__version__}')" 2>/dev/null || echo "PEFT: NOT INSTALLED"
python -c "import accelerate; print(f'Accelerate: {accelerate.__version__}')" 2>/dev/null || echo "Accelerate: NOT INSTALLED"
echo ""

# Check CUDA
echo "CUDA:"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')" 2>/dev/null
echo ""

# Check GPU
echo "GPU:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "nvidia-smi not available"
echo ""

echo "=========================================="
echo "To use the SAME environment as MTUP:"
echo "  1. Check what environment MTUP used"
echo "  2. Activate that environment"
echo "  3. Run baseline training"
echo "=========================================="
