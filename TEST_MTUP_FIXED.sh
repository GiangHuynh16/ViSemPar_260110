#!/bin/bash

# TEST_MTUP_FIXED.sh
# Quick test script for MTUP Fixed implementation
# Tests on a single example to verify the two-stage generation works

set -e

echo "================================================================================"
echo "MTUP FIXED - SINGLE EXAMPLE TEST"
echo "================================================================================"
echo ""

# Configuration
MODEL_PATH="outputs/mtup_fixed_latest/checkpoint-100"  # Will be updated after training
TEST_SENTENCE="Anh ấy đã hoàn thành công việc."

echo "📝 Test Sentence: $TEST_SENTENCE"
echo ""

# Check if model exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "⚠️  Model not found at: $MODEL_PATH"
    echo ""
    echo "Please train the model first:"
    echo "  bash TRAIN_MTUP_FIXED.sh"
    echo ""
    exit 1
fi

echo "🔧 Loading model from: $MODEL_PATH"
echo ""

# Create temporary test file
echo "$TEST_SENTENCE" > /tmp/mtup_test_single.txt

echo "🚀 Running two-stage inference..."
echo ""

# Run prediction
python3 predict_mtup_fixed.py \
    --model "$MODEL_PATH" \
    --test-file /tmp/mtup_test_single.txt \
    --verbose

echo ""
echo "================================================================================"
echo "✅ TEST COMPLETE"
echo "================================================================================"
echo ""
echo "📊 Expected Output Format:"
echo ""
echo "Step 1 (AMR without variables):"
echo "  (hoàn_thành :agent (anh) :theme (công_việc) :aspect (đã))"
echo ""
echo "Step 2 (Penman format with variables):"
echo "  (h / hoàn_thành"
echo "      :agent (a / anh)"
echo "      :theme (c / công_việc)"
echo "      :aspect (đ / đã))"
echo ""
echo "⚠️  If output doesn't match this format, check:"
echo "  1. Training completed successfully (at least 100 steps)"
echo "  2. Instruction masking was applied correctly"
echo "  3. Prompt template has Penman examples"
echo ""

# Cleanup
rm /tmp/mtup_test_single.txt
