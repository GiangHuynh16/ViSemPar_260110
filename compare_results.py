#!/usr/bin/env python3
"""
Compare evaluation results before and after post-processing
Identify what changed and why F1 decreased
"""

import json
from pathlib import Path

def load_results(json_path):
    """Load evaluation results"""
    with open(json_path, 'r') as f:
        return json.load(f)

def load_test_data(test_file):
    """Load test data with gold AMRs"""
    data = []
    current_sentence = None
    current_amr = []

    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            if line.startswith('# ::snt'):
                current_sentence = line.replace('# ::snt', '').strip()
            elif line and not line.startswith('#'):
                current_amr.append(line)
            elif line == '' and current_sentence:
                if current_amr:
                    data.append({
                        'sentence': current_sentence,
                        'gold_amr': '\n'.join(current_amr)
                    })
                current_sentence = None
                current_amr = []

    # Last entry
    if current_sentence and current_amr:
        data.append({
            'sentence': current_sentence,
            'gold_amr': '\n'.join(current_amr)
        })

    return data

def parse_log_for_errors(log_path):
    """Parse log file to extract which samples failed"""
    errors = {}
    current_idx = 0

    with open(log_path, 'r', encoding='utf-8') as f:
        content = f.read()

    for line in content.split('\n'):
        # Track progress
        if 'Evaluating:' in line and '%' in line:
            # Extract current count
            import re
            match = re.search(r'(\d+)/(\d+)', line)
            if match:
                current_idx = int(match.group(1))

        # Capture errors
        if any(x in line for x in ['Error:', 'Duplicate', 'Unmatched', 'Format error']):
            error_msg = line.strip()

            # Try to extract AMR snippet
            amr_snippet = ""
            if 'processing' in line:
                match = re.search(r'processing\s+(.+?)(?:\s*$)', line)
                if match:
                    amr_snippet = match.group(1)[:200]

            if current_idx not in errors:
                errors[current_idx] = []

            errors[current_idx].append({
                'message': error_msg,
                'amr_snippet': amr_snippet
            })

    return errors

def main():
    print("="*80)
    print("COMPARATIVE ANALYSIS: Before vs After Post-Processing")
    print("="*80)
    print()

    # Paths
    before_log = "outputs/evaluation_full_20251225_073829.log"
    after_log = "outputs/evaluation_full_20251226_085537.log"

    before_results = "outputs/evaluation_results_full_20251225_073829.json"
    after_results = "outputs/evaluation_results_full_20251226_085537.json"

    test_file = "data/public_test_ground_truth.txt"

    # Load results
    print("Loading results...")

    if Path(before_results).exists():
        before = load_results(before_results)
        print(f"✓ Before: {before}")
    else:
        print(f"✗ Before results not found: {before_results}")
        before = None

    if Path(after_results).exists():
        after = load_results(after_results)
        print(f"✓ After: {after}")
    else:
        print(f"✗ After results not found: {after_results}")
        after = None

    print()

    # Compare metrics
    if before and after:
        print("="*80)
        print("METRIC COMPARISON")
        print("="*80)
        print()

        metrics = ['precision', 'recall', 'f1', 'valid', 'total', 'errors']

        print(f"{'Metric':<15} {'Before':<15} {'After':<15} {'Change':<15}")
        print("-"*60)

        for metric in metrics:
            b_val = before.get(metric, 0)
            a_val = after.get(metric, 0)

            if isinstance(b_val, float):
                change = a_val - b_val
                print(f"{metric:<15} {b_val:<15.4f} {a_val:<15.4f} {change:+.4f}")
            else:
                change = a_val - b_val
                print(f"{metric:<15} {b_val:<15} {a_val:<15} {change:+d}")

        print()

    # Parse errors from logs
    print("="*80)
    print("ERROR ANALYSIS")
    print("="*80)
    print()

    print("Parsing error logs...")

    if Path(before_log).exists():
        before_errors = parse_log_for_errors(before_log)
        print(f"✓ Before: {len(before_errors)} samples with errors")
    else:
        print(f"✗ Before log not found")
        before_errors = {}

    if Path(after_log).exists():
        after_errors = parse_log_for_errors(after_log)
        print(f"✓ After: {len(after_errors)} samples with errors")
    else:
        print(f"✗ After log not found")
        after_errors = {}

    print()

    # Analyze what changed
    print("="*80)
    print("WHAT CHANGED?")
    print("="*80)
    print()

    # Samples that were errors before but fixed after
    before_error_ids = set(before_errors.keys())
    after_error_ids = set(after_errors.keys())

    fixed = before_error_ids - after_error_ids
    new_errors = after_error_ids - before_error_ids
    still_errors = before_error_ids & after_error_ids

    print(f"✅ FIXED: {len(fixed)} samples")
    print(f"   (Were errors before, now parse successfully)")

    print(f"❌ NEW ERRORS: {len(new_errors)} samples")
    print(f"   (Were OK before, now have errors)")

    print(f"🔄 STILL ERRORS: {len(still_errors)} samples")
    print(f"   (Had errors before and after)")

    print()

    # Key insight
    print("="*80)
    print("KEY INSIGHT")
    print("="*80)
    print()

    print(f"Post-processing FIXED {len(fixed)} parsing errors")
    print(f"But created {len(new_errors)} NEW errors")
    print()

    print("Net gain: {} samples now parse".format(len(fixed) - len(new_errors)))
    print()

    print("⚠️  CRITICAL FINDING:")
    print("   F1 DECREASED even though more samples parse!")
    print()
    print("   This means: Post-processing is CHANGING the AMR semantics")
    print("   - Fixed samples may parse but have WRONG content")
    print("   - SMATCH scores them lower because semantics changed")
    print()

    # Show examples
    if new_errors:
        print("="*80)
        print(f"NEW ERRORS (First 5 of {len(new_errors)})")
        print("="*80)
        print()

        for i, idx in enumerate(sorted(list(new_errors))[:5], 1):
            print(f"\n{i}. Sample #{idx}:")
            if idx in after_errors:
                for err in after_errors[idx][:2]:
                    print(f"   Error: {err['message'][:150]}")
                    if err['amr_snippet']:
                        print(f"   AMR: {err['amr_snippet'][:100]}")

    print()

    # Recommendations
    print("="*80)
    print("NEXT STEPS")
    print("="*80)
    print()

    print("1. EXAMINE NEW ERRORS:")
    print("   - Why did post-processing create these errors?")
    print("   - What part of post-processing is wrong?")
    print()

    print("2. INVESTIGATE SEMANTIC CHANGES:")
    print("   - Compare 'fixed' samples: predicted AMR vs gold AMR")
    print("   - See if post-processing changed semantics")
    print()

    print("3. TARGETED FIXES:")
    print("   - Keep: Parenthesis balancing (likely safe)")
    print("   - Review: Variable renaming (likely causing issues)")
    print("   - Skip: Aggressive text removal")
    print()

    print("4. VALIDATE WITH EXAMPLES:")
    print("   - Need to see actual AMR outputs (before/after post-proc)")
    print("   - Compare with gold AMR")
    print("   - Identify which fix is harmful")
    print()

if __name__ == "__main__":
    main()
