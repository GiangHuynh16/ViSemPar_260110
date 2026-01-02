#!/usr/bin/env python3
"""
Analyze AMR prediction quality - check for duplicate nodes, parse errors, etc.
"""

import re
import sys
from collections import Counter


def check_duplicate_nodes(amr_string):
    """Check if AMR has duplicate node names"""
    # Pattern to match node declarations like "x / concept"
    pattern = r'\((\w+)\s*/\s*[\w_\-]+'

    nodes = re.findall(pattern, amr_string)

    if not nodes:
        return False, []

    # Count occurrences
    node_counts = Counter(nodes)
    duplicates = [node for node, count in node_counts.items() if count > 1]

    return len(duplicates) > 0, duplicates


def analyze_amr_file(filepath):
    """Analyze all AMRs in a file"""
    print(f"Analyzing: {filepath}")
    print("=" * 70)

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split by #::snt
    parts = content.split('#::snt ')

    if len(parts) < 2:
        print("ERROR: No AMRs found!")
        return

    amrs = []
    for i, part in enumerate(parts[1:], 1):  # Skip empty first part
        lines = part.split('\n')
        sentence = lines[0]
        amr = '\n'.join(lines[1:]).strip()

        if amr:
            amrs.append((i, sentence[:80], amr))

    print(f"\nTotal AMRs: {len(amrs)}")
    print()

    # Check each AMR
    valid_count = 0
    invalid_count = 0
    duplicate_examples = []

    for idx, sent, amr in amrs:
        has_dup, dup_nodes = check_duplicate_nodes(amr)

        if has_dup:
            invalid_count += 1
            if len(duplicate_examples) < 5:
                duplicate_examples.append((idx, sent, dup_nodes, amr[:200]))
        else:
            valid_count += 1

    # Print results
    print(f"Valid AMRs (no duplicates): {valid_count} ({valid_count/len(amrs)*100:.1f}%)")
    print(f"Invalid AMRs (has duplicates): {invalid_count} ({invalid_count/len(amrs)*100:.1f}%)")
    print()

    if duplicate_examples:
        print("Examples of AMRs with duplicate nodes:")
        print("=" * 70)
        for idx, sent, dup_nodes, amr_snippet in duplicate_examples:
            print(f"\nAMR #{idx}")
            print(f"Sentence: {sent}...")
            print(f"Duplicate nodes: {', '.join(dup_nodes)}")
            print(f"AMR snippet: {amr_snippet}...")
            print()

    return {
        'total': len(amrs),
        'valid': valid_count,
        'invalid': invalid_count,
        'valid_pct': valid_count / len(amrs) * 100 if amrs else 0
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Analyze AMR quality')
    parser.add_argument('--predictions', default='predictions_formatted.txt',
                        help='Predictions file')
    parser.add_argument('--gold', default='data/public_test_ground_truth.txt',
                        help='Ground truth file')
    args = parser.parse_args()

    print("AMR QUALITY ANALYSIS")
    print("=" * 70)
    print()

    # Analyze predictions
    print("\n" + "=" * 70)
    print("PREDICTIONS")
    print("=" * 70)
    pred_stats = analyze_amr_file(args.predictions)

    # Analyze gold
    print("\n" + "=" * 70)
    print("GROUND TRUTH")
    print("=" * 70)
    gold_stats = analyze_amr_file(args.gold)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nPredictions: {pred_stats['valid']}/{pred_stats['total']} valid ({pred_stats['valid_pct']:.1f}%)")
    print(f"Ground Truth: {gold_stats['valid']}/{gold_stats['total']} valid ({gold_stats['valid_pct']:.1f}%)")
    print()

    if pred_stats['invalid'] > 0:
        print(f"⚠️  WARNING: {pred_stats['invalid']} predictions have duplicate node names!")
        print("   These AMRs cannot be parsed by SMATCH and will be skipped.")
        print()
        print("   Possible causes:")
        print("   - Model generating invalid AMR syntax")
        print("   - Need better prompt/training to enforce unique node names")
        print("   - Consider post-processing to fix duplicate nodes")
    else:
        print("✓ All predictions are valid!")


if __name__ == "__main__":
    main()
