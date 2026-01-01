#!/bin/bash
# Check if the gradient checkpointing fix has been applied

echo "==========================================="
echo "CHECKING IF FIX IS APPLIED"
echo "==========================================="
echo ""

cd /mnt/nghiepth/giangha/visempar/ViSemPar_new1

# Check git status
echo "Step 1: Git status..."
git status
echo ""

# Check last commit
echo "Step 2: Last commit..."
git log -1 --oneline
echo ""
echo "Expected: 'Fix PYTORCH_CUDA_ALLOC_CONF for PyTorch 2.0.1' or later"
echo ""

# Check if gradient_checkpointing_enable is AFTER get_peft_model
echo "Step 3: Checking gradient_checkpointing_enable position..."
echo ""

# Find line numbers
PEFT_LINE=$(grep -n "get_peft_model" train_baseline.py | head -1 | cut -d: -f1)
CHECKPOINT_LINE=$(grep -n "gradient_checkpointing_enable" train_baseline.py | grep -v "^#" | head -1 | cut -d: -f1)

echo "  Line $PEFT_LINE: get_peft_model (LoRA applied)"
echo "  Line $CHECKPOINT_LINE: gradient_checkpointing_enable"
echo ""

if [ "$CHECKPOINT_LINE" -gt "$PEFT_LINE" ]; then
    echo "  ✅ CORRECT: gradient_checkpointing_enable is AFTER LoRA (line $CHECKPOINT_LINE > $PEFT_LINE)"
else
    echo "  ❌ WRONG: gradient_checkpointing_enable is BEFORE LoRA (line $CHECKPOINT_LINE < $PEFT_LINE)"
    echo ""
    echo "  FIX NOT APPLIED! Need to:"
    echo "  1. git reset --hard origin/main"
    echo "  2. git pull origin main"
    echo "  3. Clear cache: find . -type d -name __pycache__ -exec rm -rf {} +"
fi
echo ""

# Show the actual code around LoRA
echo "Step 4: Code around LoRA application..."
echo ""
grep -A 15 "get_peft_model" train_baseline.py | head -20
echo ""

echo "==========================================="
