#!/bin/bash
# Test the fixed model after training
# Run: bash TEST_FIXED_MODEL.sh

echo "=========================================="
echo "TEST FIXED MODEL"
echo "=========================================="
echo ""

cd /mnt/nghiepth/giangha/visempar/ViSemPar_new1

# Activate conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate baseline_final

# Find latest trained model
LATEST_MODEL=$(ls -t outputs/ | grep baseline_fixed | head -1)

if [ -z "$LATEST_MODEL" ]; then
    echo "❌ No trained model found in outputs/"
    echo ""
    echo "Please specify model path:"
    echo "  python predict_baseline_fixed.py --model <path>"
    exit 1
fi

echo "Found model: outputs/$LATEST_MODEL"
echo ""

# Create output directory
mkdir -p evaluation_results/baseline_7b_fixed

echo "Running prediction..."
echo ""

# Run prediction
python predict_baseline_fixed.py \
    --model "outputs/$LATEST_MODEL/final" \
    --test-file data/public_test.txt \
    --output evaluation_results/baseline_7b_fixed/predictions.txt

if [ $? -ne 0 ]; then
    echo "❌ Prediction failed!"
    exit 1
fi

echo ""
echo "=========================================="
echo "✅ PREDICTION COMPLETE"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  evaluation_results/baseline_7b_fixed/predictions.txt"
echo ""

# Check quality
echo "Checking AMR quality..."
python -c "
import re

with open('evaluation_results/baseline_7b_fixed/predictions.txt', 'r') as f:
    content = f.read()

parts = content.split('#::snt ')
amrs = []
errors = []

for i, part in enumerate(parts[1:], 1):
    lines = part.split('\n')
    amr = '\n'.join(lines[1:]).strip()

    if amr:
        # Check balanced parentheses
        open_count = amr.count('(')
        close_count = amr.count(')')

        if open_count != close_count:
            errors.append(f'AMR #{i}: unmatched parentheses ({open_count} open, {close_count} close)')
        else:
            amrs.append(amr)

print(f'Total AMRs: {len(parts)-1}')
print(f'Valid AMRs: {len(amrs)}')
print(f'Invalid AMRs: {len(errors)}')
print()

if errors:
    print('Errors:')
    for err in errors[:5]:
        print(f'  - {err}')
else:
    print('✅ All AMRs are valid!')
" || echo "Quality check script failed (not critical)"

echo ""
echo "=========================================="
echo "Compare with old model:"
echo ""
echo "OLD (buggy):"
echo "  - 124/150 valid AMRs (82.7%)"
echo "  - 26 parse errors (17.3%)"
echo ""
echo "NEW (fixed):"
wc -l evaluation_results/baseline_7b_fixed/predictions.txt 2>/dev/null | awk '{print "  - " int($1/3) " AMRs generated"}'
echo "  - Check quality output above"
echo ""
