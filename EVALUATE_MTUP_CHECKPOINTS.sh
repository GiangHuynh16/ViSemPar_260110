#!/bin/bash

# EVALUATE_MTUP_CHECKPOINTS.sh
# Evaluate all MTUP Fixed checkpoints to find the best one
# Similar to baseline evaluation strategy

set -e

if [ $# -eq 0 ]; then
    echo "Usage: bash EVALUATE_MTUP_CHECKPOINTS.sh <output_directory>"
    echo ""
    echo "Example:"
    echo "  bash EVALUATE_MTUP_CHECKPOINTS.sh outputs/mtup_fixed_20260104_120000"
    exit 1
fi

OUTPUT_DIR="$1"

echo "================================================================================"
echo "MTUP FIXED - CHECKPOINT EVALUATION"
echo "================================================================================"
echo ""

# Check if output directory exists
if [ ! -d "$OUTPUT_DIR" ]; then
    echo "❌ Error: Directory not found: $OUTPUT_DIR"
    exit 1
fi

# Find all checkpoints
CHECKPOINTS=$(find "$OUTPUT_DIR" -type d -name "checkpoint-*" | sort -V)

if [ -z "$CHECKPOINTS" ]; then
    echo "❌ No checkpoints found in: $OUTPUT_DIR"
    exit 1
fi

echo "📦 Found checkpoints:"
echo "$CHECKPOINTS" | while read -r ckpt; do
    echo "  - $(basename $ckpt)"
done
echo ""

# Create evaluation directory
EVAL_DIR="evaluation_results/mtup_checkpoints"
mkdir -p "$EVAL_DIR"

echo "📁 Evaluation results will be saved to: $EVAL_DIR"
echo ""

# Validation file
VALIDATION_FILE="data/public_test.txt"  # Use public test as validation

if [ ! -f "$VALIDATION_FILE" ]; then
    echo "⚠️  Validation file not found: $VALIDATION_FILE"
    echo "Using first 20 samples from training data..."
    head -n 20 data/train_amr_1.txt > /tmp/mtup_validation.txt
    VALIDATION_FILE="/tmp/mtup_validation.txt"
fi

echo "📊 Evaluating on: $VALIDATION_FILE"
echo ""

# Evaluate each checkpoint
RESULTS_FILE="$EVAL_DIR/checkpoint_comparison.txt"
echo "Checkpoint,Valid_AMRs,Total,Validity_Percentage" > "$RESULTS_FILE"

echo "🚀 Starting evaluation..."
echo ""

echo "$CHECKPOINTS" | while read -r checkpoint; do
    ckpt_name=$(basename "$checkpoint")
    echo "================================================================================"
    echo "Evaluating: $ckpt_name"
    echo "================================================================================"

    output_file="$EVAL_DIR/${ckpt_name}.txt"

    # Run prediction
    python3 predict_mtup_fixed.py \
        --model "$checkpoint" \
        --test-file "$VALIDATION_FILE" \
        --output "$output_file" \
        2>&1 | tee "$EVAL_DIR/${ckpt_name}_log.txt"

    # Count valid AMRs
    valid_count=$(python3 -c "
import re
from collections import Counter

def validate_amr(amr_text):
    if not amr_text.strip() or '(' not in amr_text:
        return False
    if amr_text.count('(') != amr_text.count(')'):
        return False
    pattern = r'\((\w+)\s*/\s*[\w_\-]+'
    nodes = re.findall(pattern, amr_text)
    duplicates = [node for node, count in Counter(nodes).items() if count > 1]
    if duplicates:
        return False
    return True

with open('$output_file', 'r', encoding='utf-8') as f:
    amrs = f.read().split('\n\n')

valid = sum(1 for amr in amrs if validate_amr(amr))
total = len(amrs)
print(f'{valid}')
")

    total_count=$(python3 -c "
with open('$output_file', 'r', encoding='utf-8') as f:
    amrs = f.read().split('\n\n')
print(len(amrs))
")

    validity_pct=$(python3 -c "print(f'{$valid_count/$total_count*100:.1f}')")

    echo "$ckpt_name,$valid_count,$total_count,$validity_pct" >> "$RESULTS_FILE"

    echo ""
    echo "✅ $ckpt_name: $valid_count/$total_count valid ($validity_pct%)"
    echo ""
done

echo ""
echo "================================================================================"
echo "EVALUATION COMPLETE"
echo "================================================================================"
echo ""

# Display results
echo "📊 Results Summary:"
echo ""
cat "$RESULTS_FILE" | column -t -s ','

echo ""
echo "🏆 Best checkpoint:"
# Find best checkpoint (highest validity)
best_checkpoint=$(tail -n +2 "$RESULTS_FILE" | sort -t',' -k4 -rn | head -n1)
echo "$best_checkpoint" | awk -F',' '{print "  " $1 " with " $4 "% validity (" $2 "/" $3 " valid AMRs)"}'

echo ""
echo "📁 Detailed results saved to: $RESULTS_FILE"
echo ""

# Cleanup
if [ -f "/tmp/mtup_validation.txt" ]; then
    rm /tmp/mtup_validation.txt
fi
