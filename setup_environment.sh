#!/bin/bash
# Environment Setup Script for Vietnamese AMR Parser
# Run on server to create conda environment and install dependencies

set -e

echo "=========================================="
echo "VIETNAMESE AMR PARSER - ENVIRONMENT SETUP"
echo "=========================================="
echo ""

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found. Please install Anaconda or Miniconda first."
    echo "   Download from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo "✓ Conda found: $(conda --version)"
echo ""

# Environment name
ENV_NAME="viamr"

# Check if environment already exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "⚠️  Environment '${ENV_NAME}' already exists!"
    read -p "Do you want to remove and recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n ${ENV_NAME} -y
    else
        echo "Cancelled. Use: conda activate ${ENV_NAME}"
        exit 0
    fi
fi

echo ""
echo "=========================================="
echo "STEP 1: Creating Conda Environment"
echo "=========================================="
echo ""

# Create conda environment with Python 3.10
conda create -n ${ENV_NAME} python=3.10 -y

echo ""
echo "✓ Environment created: ${ENV_NAME}"
echo ""

echo "=========================================="
echo "STEP 2: Installing PyTorch with CUDA"
echo "=========================================="
echo ""

# Activate environment
eval "$(conda shell.bash hook)"
conda activate ${ENV_NAME}

# Check CUDA version
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -d',' -f1)
    echo "Detected CUDA version: ${CUDA_VERSION}"
else
    echo "⚠️  nvcc not found, assuming CUDA 12.1"
    CUDA_VERSION="12.1"
fi

# Install PyTorch based on CUDA version
if [[ $CUDA_VERSION == V12* ]] || [[ $CUDA_VERSION == 12* ]]; then
    echo "Installing PyTorch for CUDA 12.1..."
    conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
elif [[ $CUDA_VERSION == V11* ]] || [[ $CUDA_VERSION == 11* ]]; then
    echo "Installing PyTorch for CUDA 11.8..."
    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
else
    echo "Installing PyTorch for CUDA 12.1 (default)..."
    conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
fi

echo ""
echo "✓ PyTorch installed"
echo ""

echo "=========================================="
echo "STEP 3: Installing Python Dependencies"
echo "=========================================="
echo ""

# Install from requirements.txt
if [ -f "requirements.txt" ]; then
    echo "Installing from requirements.txt..."
    pip install -r requirements.txt
else
    echo "⚠️  requirements.txt not found. Installing core packages..."
    pip install transformers==4.46.3 \
                accelerate==1.2.1 \
                peft==0.13.2 \
                datasets>=2.14.0 \
                penman>=1.2.0 \
                smatch>=1.0.4 \
                tqdm>=4.65.0 \
                pandas>=2.0.0 \
                scikit-learn>=1.3.0 \
                "numpy>=1.24.0,<2.0.0" \
                "huggingface-hub>=0.24.0,<1.0" \
                python-dotenv>=1.0.0 \
                tensorboard>=2.15.0
fi

echo ""
echo "✓ Dependencies installed"
echo ""

echo "=========================================="
echo "STEP 4: Verification"
echo "=========================================="
echo ""

# Verify installation
echo "Verifying installation..."
echo ""

# Check Python version
PYTHON_VERSION=$(python --version)
echo "Python: $PYTHON_VERSION"

# Check PyTorch
PYTORCH_VERSION=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "ERROR")
echo "PyTorch: $PYTORCH_VERSION"

# Check CUDA availability
CUDA_AVAILABLE=$(python -c "import torch; print('Yes' if torch.cuda.is_available() else 'No')" 2>/dev/null || echo "ERROR")
echo "CUDA available: $CUDA_AVAILABLE"

# Check Transformers
TRANSFORMERS_VERSION=$(python -c "import transformers; print(transformers.__version__)" 2>/dev/null || echo "ERROR")
echo "Transformers: $TRANSFORMERS_VERSION"

# Check PEFT
PEFT_VERSION=$(python -c "import peft; print(peft.__version__)" 2>/dev/null || echo "ERROR")
echo "PEFT: $PEFT_VERSION"

# Check GPU memory
echo ""
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Information:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
else
    echo "⚠️  nvidia-smi not found"
fi

echo ""
echo "=========================================="
echo "STEP 5: Project Setup"
echo "=========================================="
echo ""

# Create necessary directories
echo "Creating project directories..."
mkdir -p data outputs logs outputs/checkpoints results

echo "✓ Directories created:"
ls -ld data outputs logs results

echo ""
echo "=========================================="
echo "INSTALLATION COMPLETE!"
echo "=========================================="
echo ""
echo "Environment: ${ENV_NAME}"
echo ""
echo "To activate:"
echo "  conda activate ${ENV_NAME}"
echo ""
echo "To verify:"
echo "  python -c \"import torch; print(f'CUDA: {torch.cuda.is_available()}')\""
echo ""
echo "Next steps:"
echo "  1. Login to HuggingFace (optional):"
echo "     huggingface-cli login"
echo ""
echo "  2. Start training:"
echo "     tmux new -s baseline_7b"
echo "     bash START_BASELINE_7B_TRAINING.sh"
echo ""
echo "=========================================="

# Deactivate environment
conda deactivate

echo ""
echo "✓ Setup complete! Environment is ready to use."
echo ""
