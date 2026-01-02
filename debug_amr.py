#!/usr/bin/env python3
"""
Debug AMR parsing issues
"""

import sys

try:
    import smatch
except ImportError:
    print("ERROR: smatch not installed!")
    sys.exit(1)


def test_parse_amr(amr_string, label):
    """Test parsing a single AMR"""
    print(f"\n{'='*70}")
    print(f"Testing {label}")
    print(f"{'='*70}")
    print(f"AMR (first 200 chars):\n{amr_string[:200]}...")
    print()

    # Try to parse
    try:
        # Convert to single line
        single_line = amr_string.replace('\n', ' ')
        print(f"Single-line (first 200 chars):\n{single_line[:200]}...")
        print()

        # Use generate_amr_lines to parse
        amr_lines = list(smatch.generate_amr_lines(single_line.split('\n')))

        if amr_lines:
            print(f"✓ Parsed successfully! Got {len(amr_lines)} AMR(s)")
            return True
        else:
            print("✗ No AMRs parsed")
            return False

    except Exception as e:
        print(f"✗ Parse error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    # Read first prediction
    print("Reading predictions_formatted.txt...")
    with open('predictions_formatted.txt', 'r', encoding='utf-8') as f:
        content = f.read()

    # Split by #::snt
    parts = content.split('#::snt ')

    if len(parts) < 2:
        print("ERROR: No AMRs found in predictions_formatted.txt")
        sys.exit(1)

    # Get first AMR (skip empty first part)
    first_amr = parts[1]

    # Extract sentence and AMR
    lines = first_amr.split('\n')
    sentence = lines[0]
    amr = '\n'.join(lines[1:]).strip()

    print(f"\nFirst sentence: {sentence[:100]}...")
    test_parse_amr(amr, "First Prediction")

    # Now test first gold AMR
    print("\n\nReading data/public_test_ground_truth.txt...")
    with open('data/public_test_ground_truth.txt', 'r', encoding='utf-8') as f:
        content = f.read()

    parts = content.split('#::snt ')
    first_amr = parts[1]
    lines = first_amr.split('\n')
    sentence = lines[0]
    amr = '\n'.join(lines[1:]).strip()

    print(f"\nFirst sentence: {sentence[:100]}...")
    test_parse_amr(amr, "First Gold AMR")


if __name__ == "__main__":
    main()
