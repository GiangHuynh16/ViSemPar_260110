#!/usr/bin/env python3
"""
MTUP Preprocessing Script
Converts standard AMR training data into two-stage MTUP format:
  Sentence → AMR without variables → AMR with variables (Penman)

Usage:
  python3 preprocess_mtup.py \
      --input data/train_amr_1.txt \
      --output data/train_amr_mtup_preprocessed.txt
"""

import argparse
import re
from pathlib import Path
from typing import Tuple, Optional


def remove_variables_from_amr(amr_with_vars: str) -> str:
    """
    Remove variable assignments from AMR to create 'AMR without variables'

    Example:
        Input:  (h / hoàn_thành :agent (a / anh))
        Output: (hoàn_thành :agent (anh))
    """
    # Pattern: (variable / concept)  →  (concept)
    # Matches: (h / hoàn_thành)  →  (hoàn_thành)
    pattern = r'\(\s*(\w+)\s*/\s*([^\s\)]+)'
    amr_no_vars = re.sub(pattern, r'(\2', amr_with_vars)

    return amr_no_vars.strip()


def validate_amr(amr: str) -> Tuple[bool, str]:
    """
    Validate AMR structure

    Returns:
        (is_valid, error_message)
    """
    if not amr.strip():
        return False, "Empty AMR"

    if '(' not in amr:
        return False, "No parentheses found"

    # Check balanced parentheses
    if amr.count('(') != amr.count(')'):
        return False, f"Unbalanced parentheses: {amr.count('(')} open, {amr.count(')')} close"

    return True, ""


def parse_training_file(file_path: str) -> list:
    """
    Parse training file with format:
        #::snt Sentence text
        (AMR graph)

    Returns:
        List of (sentence, amr) tuples
    """
    examples = []

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Skip empty lines
        if not line:
            i += 1
            continue

        # Look for sentence marker (handle both #::snt and # ::snt)
        if line.startswith('#::snt') or line.startswith('# ::snt'):
            # Remove marker (try both formats)
            if line.startswith('#::snt'):
                sentence = line.replace('#::snt', '').strip()
            else:
                sentence = line.replace('# ::snt', '').strip()

            # Collect AMR lines (until next #::snt or empty line)
            amr_lines = []
            i += 1

            while i < len(lines):
                amr_line = lines[i].strip()

                # Stop at next sentence or empty line
                if not amr_line or amr_line.startswith('#::'):
                    break

                amr_lines.append(amr_line)
                i += 1

            amr = '\n'.join(amr_lines).strip()

            # Validate
            is_valid, error = validate_amr(amr)
            if is_valid:
                examples.append((sentence, amr))
            else:
                print(f"⚠️  Skipping invalid AMR: {error}")
                print(f"    Sentence: {sentence[:50]}...")
        else:
            i += 1

    return examples


def create_mtup_example(sentence: str, amr_with_vars: str) -> dict:
    """
    Create MTUP two-stage example

    Returns:
        {
            'sentence': str,
            'amr_no_vars': str,
            'amr_with_vars': str
        }
    """
    # Generate AMR without variables
    amr_no_vars = remove_variables_from_amr(amr_with_vars)

    return {
        'sentence': sentence,
        'amr_no_vars': amr_no_vars,
        'amr_with_vars': amr_with_vars
    }


def format_mtup_output(example: dict) -> str:
    """
    Format MTUP example for output file

    Format:
        #::snt Sentence
        #::amr-no-vars AMR without variables
        #::amr-with-vars
        (full Penman AMR)
    """
    output = f"#::snt {example['sentence']}\n"
    output += f"#::amr-no-vars {example['amr_no_vars']}\n"
    output += f"#::amr-with-vars\n"
    output += example['amr_with_vars']

    return output


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess training data for MTUP two-stage approach'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input training file (standard AMR format)'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output file (MTUP format)'
    )
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate all AMRs before processing'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("MTUP PREPROCESSING")
    print("=" * 80)
    print()

    # Check input file
    if not Path(args.input).exists():
        print(f"❌ Error: Input file not found: {args.input}")
        return

    print(f"📂 Input:  {args.input}")
    print(f"📂 Output: {args.output}")
    print()

    # Parse training data
    print("🔍 Parsing training data...")
    examples = parse_training_file(args.input)
    print(f"✅ Loaded {len(examples)} examples")
    print()

    # Convert to MTUP format
    print("🔄 Converting to MTUP format...")
    mtup_examples = []

    for i, (sentence, amr) in enumerate(examples, 1):
        if i % 100 == 0:
            print(f"  Processed {i}/{len(examples)}...")

        mtup_example = create_mtup_example(sentence, amr)
        mtup_examples.append(mtup_example)

    print(f"✅ Converted {len(mtup_examples)} examples")
    print()

    # Show sample
    print("📝 Sample MTUP example:")
    print("-" * 80)
    print(format_mtup_output(mtup_examples[0]))
    print("-" * 80)
    print()

    # Write output
    print(f"💾 Writing to {args.output}...")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for i, example in enumerate(mtup_examples):
            f.write(format_mtup_output(example))

            # Add separator between examples
            if i < len(mtup_examples) - 1:
                f.write('\n\n')

    print(f"✅ Saved {len(mtup_examples)} examples")
    print()

    # Statistics
    print("=" * 80)
    print("PREPROCESSING COMPLETE")
    print("=" * 80)
    print()
    print(f"📊 Statistics:")
    print(f"  Total examples: {len(mtup_examples)}")
    print(f"  Output file: {args.output}")
    print(f"  File size: {output_path.stat().st_size / 1024:.1f} KB")
    print()

    # Validation check
    if args.validate:
        print("🔍 Validating AMRs...")
        invalid_count = 0

        for i, example in enumerate(mtup_examples, 1):
            # Validate AMR with variables
            is_valid, error = validate_amr(example['amr_with_vars'])
            if not is_valid:
                print(f"  ⚠️  Example {i}: {error}")
                invalid_count += 1

        if invalid_count == 0:
            print(f"  ✅ All {len(mtup_examples)} AMRs are valid!")
        else:
            print(f"  ⚠️  Found {invalid_count} invalid AMRs")
        print()

    print("🎯 Next steps:")
    print()
    print("1. Train MTUP model:")
    print(f"   bash TRAIN_MTUP_FIXED.sh")
    print()
    print("2. Or test on a single example:")
    print(f"   bash TEST_MTUP_FIXED.sh")
    print()


if __name__ == '__main__':
    main()
