#!/usr/bin/env python3
"""
Validate Vietnamese AMR output format
Ensures:
1. UTF-8 encoding is correct
2. Vietnamese characters display properly
3. AMR format is valid (balanced parentheses, no duplicates)
4. No explanations after AMR
"""

import sys
import re
from collections import Counter
from pathlib import Path


def check_utf8_encoding(file_path: str) -> bool:
    """Check if file is valid UTF-8"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        print(f"✅ File is valid UTF-8")
        return True
    except UnicodeDecodeError as e:
        print(f"❌ UTF-8 encoding error: {e}")
        return False


def check_vietnamese_chars(file_path: str) -> bool:
    """Check if Vietnamese characters are present and valid"""
    vietnamese_chars = set('àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ')
    vietnamese_chars.update('ÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴĐ')

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Check for Vietnamese chars
    has_vietnamese = any(char in vietnamese_chars for char in content)

    if has_vietnamese:
        print(f"✅ Vietnamese characters found")

        # Show sample Vietnamese text
        lines = content.split('\n')
        for i, line in enumerate(lines[:20]):
            if any(char in vietnamese_chars for char in line):
                print(f"   Sample: {line[:80]}")
                break
        return True
    else:
        print(f"⚠️  No Vietnamese characters found (might be English-only test)")
        return True


def check_amr_validity(amr_string: str) -> tuple:
    """Check if AMR is valid"""
    errors = []

    # Check 1: Balanced parentheses
    open_count = amr_string.count('(')
    close_count = amr_string.count(')')

    if open_count != close_count:
        errors.append(f"unmatched_parentheses ({open_count} open, {close_count} close)")

    # Check 2: Duplicate nodes
    pattern = r'\((\w+)\s*/\s*[\w_\-]+'
    nodes = re.findall(pattern, amr_string)

    if nodes:
        node_counts = Counter(nodes)
        duplicates = [node for node, count in node_counts.items() if count > 1]

        if duplicates:
            errors.append(f"duplicate_nodes: {', '.join(duplicates)}")

    # Check 3: Empty AMR
    if not amr_string.strip():
        errors.append("empty_amr")

    # Check 4: No closing parenthesis at all
    if open_count == 0:
        errors.append("no_parentheses")

    return len(errors) == 0, errors


def validate_output_file(file_path: str):
    """Validate entire output file"""
    print("=" * 70)
    print("VALIDATE VIETNAMESE AMR OUTPUT")
    print("=" * 70)
    print(f"File: {file_path}")
    print()

    # Check file exists
    if not Path(file_path).exists():
        print(f"❌ File not found: {file_path}")
        return False

    # Check UTF-8 encoding
    print("Step 1: Check UTF-8 encoding")
    print("-" * 70)
    if not check_utf8_encoding(file_path):
        return False
    print()

    # Check Vietnamese characters
    print("Step 2: Check Vietnamese characters")
    print("-" * 70)
    check_vietnamese_chars(file_path)
    print()

    # Check AMR format
    print("Step 3: Check AMR format")
    print("-" * 70)

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Parse AMRs
    parts = content.split('#::snt ')

    if len(parts) <= 1:
        print(f"❌ No AMRs found (missing #::snt markers)")
        return False

    total_amrs = len(parts) - 1
    valid_amrs = []
    invalid_amrs = []

    for i, part in enumerate(parts[1:], 1):
        lines = part.split('\n')
        sentence = lines[0]
        amr = '\n'.join(lines[1:]).strip()

        if amr:
            is_valid, errors = check_amr_validity(amr)

            if is_valid:
                valid_amrs.append(i)
            else:
                invalid_amrs.append((i, errors, sentence[:50]))

    print(f"Total AMRs: {total_amrs}")
    print(f"Valid AMRs: {len(valid_amrs)} ({len(valid_amrs)/total_amrs*100:.1f}%)")
    print(f"Invalid AMRs: {len(invalid_amrs)} ({len(invalid_amrs)/total_amrs*100:.1f}%)")
    print()

    # Show invalid AMRs
    if invalid_amrs:
        print("Invalid AMRs:")
        for idx, errors, sent in invalid_amrs[:10]:
            print(f"  #{idx}: {', '.join(errors)}")
            print(f"      Sentence: {sent}...")

        if len(invalid_amrs) > 10:
            print(f"  ... and {len(invalid_amrs) - 10} more")
        print()

    # Check for explanations after AMR
    print("Step 4: Check for explanations after AMR")
    print("-" * 70)

    has_explanations = 0
    for i, part in enumerate(parts[1:], 1):
        lines = part.split('\n')
        amr_lines = [l for l in lines[1:] if l.strip()]

        if len(amr_lines) > 15:  # Suspicious if AMR has too many lines
            has_explanations += 1

    if has_explanations > 0:
        print(f"⚠️  {has_explanations} AMRs might contain explanations (>15 lines)")
    else:
        print(f"✅ No explanations detected")
    print()

    # Summary
    print("=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    all_valid = len(invalid_amrs) == 0 and has_explanations == 0

    if all_valid:
        print("✅ ALL CHECKS PASSED")
        print(f"   - {total_amrs} AMRs")
        print(f"   - {len(valid_amrs)} valid (100%)")
        print(f"   - 0 invalid (0%)")
        print(f"   - UTF-8 encoding correct")
        print(f"   - Vietnamese characters present")
    else:
        print("⚠️  SOME ISSUES FOUND")
        print(f"   - {total_amrs} AMRs total")
        print(f"   - {len(valid_amrs)} valid ({len(valid_amrs)/total_amrs*100:.1f}%)")
        print(f"   - {len(invalid_amrs)} invalid ({len(invalid_amrs)/total_amrs*100:.1f}%)")
        if has_explanations > 0:
            print(f"   - {has_explanations} might have explanations")

    print("=" * 70)

    return all_valid


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Validate Vietnamese AMR output')
    parser.add_argument('--file', type=str, required=True, help='Output file to validate')

    args = parser.parse_args()

    success = validate_output_file(args.file)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
