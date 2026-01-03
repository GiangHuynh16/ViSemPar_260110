#!/bin/bash

# Script to check if max_length=512 is causing truncation issues
# Run this on the server to analyze sentence lengths

echo "==================================="
echo "Checking if max_length=512 is too small"
echo "==================================="
echo ""

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate baseline_final

# Run analysis
python3 analyze_sentence_lengths.py

echo ""
echo "==================================="
echo "Analysis complete!"
echo "==================================="
