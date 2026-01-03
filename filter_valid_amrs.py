#!/usr/bin/env python3
"""
Filter predictions to only valid AMRs for SMATCH calculation
This allows calculating SMATCH even when some AMRs are invalid
"""

import argparse
import re
from collections import Counter


def validate_amr(amr_text):
    """Validate AMR structure"""
    if not amr_text.strip() or '(' not in amr_text:
        return False

    # Check balanced parentheses
    if amr_text.count('(') != amr_text.count(')'):
        return False

    # Check for duplicate node variables
    pattern = r'\((\w+)\s*/\s*[\w_\-]+'
    nodes = re.findall(pattern, amr_text)
    duplicates = [node for node, count in Counter(nodes).items() if count > 1]
    if duplicates:
        return False

    return True


def main():
    parser = argparse.ArgumentParser(
        description='Filter valid AMRs for SMATCH calculation'
    )
    parser.add_argument(
        '--predictions',
        required=True,
        help='Path to predictions file'
    )
    parser.add_argument(
        '--ground-truth',
        required=True,
        help='Path to ground truth file'
    )
    parser.add_argument(
        '--output-pred',
        required=True,
        help='Path to output filtered predictions'
    )
    parser.add_argument(
        '--output-gold',
        required=True,
        help='Path to output filtered ground truth'
    )

    args = parser.parse_args()

    # Read predictions
    with open(args.predictions, 'r', encoding='utf-8') as f:
        predictions = f.read().split('\n\n')

    # Read ground truth
    with open(args.ground_truth, 'r', encoding='utf-8') as f:
        ground_truth = f.read().split('\n\n')

    print(f"Total predictions: {len(predictions)}")
    print(f"Total ground truth: {len(ground_truth)}")

    if len(predictions) != len(ground_truth):
        print(f"⚠️ WARNING: Mismatch in counts!")
        print(f"   Using minimum: {min(len(predictions), len(ground_truth))}")

    # Filter valid pairs
    valid_predictions = []
    valid_ground_truth = []
    invalid_indices = []

    for i, (pred, gold) in enumerate(zip(predictions, ground_truth), 1):
        if validate_amr(pred):
            valid_predictions.append(pred)
            valid_ground_truth.append(gold)
        else:
            invalid_indices.append(i)
            print(f"Skipping sentence #{i} (invalid AMR)")

    # Write filtered files
    with open(args.output_pred, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(valid_predictions))

    with open(args.output_gold, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(valid_ground_truth))

    print(f"\n=== SUMMARY ===")
    print(f"Valid AMRs: {len(valid_predictions)}")
    print(f"Invalid AMRs: {len(invalid_indices)}")
    print(f"Validity rate: {len(valid_predictions)/len(predictions)*100:.1f}%")
    print(f"\nInvalid indices: {invalid_indices}")

    print(f"\n=== OUTPUT FILES ===")
    print(f"Filtered predictions: {args.output_pred}")
    print(f"Filtered ground truth: {args.output_gold}")

    print(f"\n=== NEXT STEP ===")
    print(f"Calculate SMATCH with:")
    print(f"  python -m smatch -f \\")
    print(f"      {args.output_pred} \\")
    print(f"      {args.output_gold} \\")
    print(f"      --significant 4")


if __name__ == '__main__':
    main()
