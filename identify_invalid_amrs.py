#!/usr/bin/env python3
"""
Identify which sentence indices have invalid AMRs
This helps correlate with sentence length analysis
"""

import re
from collections import Counter

def validate_amr(amr_text):
    """Validate AMR and return error types"""
    errors = []

    # Check 1: Balanced parentheses
    open_count = amr_text.count('(')
    close_count = amr_text.count(')')
    if open_count != close_count:
        errors.append(f"Unbalanced ({open_count} open, {close_count} close)")

    # Check 2: Duplicate node variables
    pattern = r'\((\w+)\s*/\s*[\w_\-]+'
    nodes = re.findall(pattern, amr_text)
    duplicates = [node for node, count in Counter(nodes).items() if count > 1]
    if duplicates:
        errors.append(f"Duplicate nodes: {', '.join(duplicates)}")

    # Check 3: Empty AMR
    if not amr_text.strip() or '(' not in amr_text:
        errors.append("Empty or invalid AMR")

    return errors

# Read predictions
print("Reading checkpoint-1500 predictions...")
try:
    with open('evaluation_results/checkpoint_comparison/checkpoint-1500.txt', 'r', encoding='utf-8') as f:
        content = f.read()
        predictions = content.split('\n\n')

    print(f"Total predictions: {len(predictions)}")

    invalid_indices = []

    print("\n=== INVALID AMRs ===\n")

    for i, pred in enumerate(predictions, 1):
        if not pred.strip():
            print(f"Sentence #{i}: MISSING/EMPTY")
            invalid_indices.append((i, "MISSING"))
            continue

        errors = validate_amr(pred)
        if errors:
            print(f"Sentence #{i}:")
            for error in errors:
                print(f"  - {error}")
            print(f"  AMR preview: {pred[:100]}...")
            print()
            invalid_indices.append((i, errors))

    print(f"\n=== SUMMARY ===")
    print(f"Total invalid: {len(invalid_indices)}")
    print(f"Invalid indices: {[idx for idx, _ in invalid_indices]}")

    # Save indices for correlation with sentence length
    with open('invalid_amr_indices.txt', 'w') as f:
        for idx, errors in invalid_indices:
            f.write(f"{idx}\n")

    print(f"\nInvalid indices saved to: invalid_amr_indices.txt")

except FileNotFoundError:
    print("⚠️ Prediction file not found!")
    print("Expected: evaluation_results/checkpoint_comparison/checkpoint-1500.txt")
