#!/usr/bin/env python3
"""
Inspect error cases in detail
Show sentence, predicted AMR, gold AMR, and error type
"""

import json
import sys
from pathlib import Path

def load_results(json_path):
    """Load evaluation results"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_test_data(test_file):
    """Load test sentences and gold AMRs"""
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
                # End of entry
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

def analyze_error(predicted, gold):
    """Analyze what went wrong"""
    issues = []

    # Check parentheses
    pred_open = predicted.count('(')
    pred_close = predicted.count(')')

    if pred_open != pred_close:
        issues.append(f"Unbalanced parens: {pred_open} '(' vs {pred_close} ')'")

    # Check length
    if len(predicted) < 10:
        issues.append("Too short (likely incomplete)")

    if len(predicted) > len(gold) * 2:
        issues.append("Too long (over-generation)")

    # Check for common patterns
    if ')))))' in predicted:
        issues.append("Excessive closing parens")

    if '(((((' in predicted:
        issues.append("Excessive opening parens")

    # Check for obvious malformation
    if not predicted.startswith('('):
        issues.append("Doesn't start with '('")

    if not predicted.endswith(')'):
        issues.append("Doesn't end with ')'")

    return issues

def main():
    # Paths
    results_file = "outputs/evaluation_results_full_20251225_073829.json"
    test_file = "data/public_test_ground_truth.txt"
    log_file = "outputs/evaluation_full_20251225_073829.log"

    # Check files exist
    if not Path(results_file).exists():
        print(f"Error: Results file not found: {results_file}")
        print("Run evaluation first!")
        return

    if not Path(test_file).exists():
        print(f"Error: Test file not found: {test_file}")
        return

    print("="*80)
    print("DETAILED ERROR INSPECTION")
    print("="*80)
    print()

    # Load data
    print("Loading evaluation results...")
    results = load_results(results_file)

    print("Loading test data...")
    test_data = load_test_data(test_file)

    print(f"Found {len(test_data)} test examples")
    print()

    # Parse log to find which examples failed
    print("Parsing log file...")
    with open(log_file, 'r', encoding='utf-8') as f:
        log_content = f.read()

    # Count errors
    error_count = log_content.count('Error:') + \
                  log_content.count('Duplicate node') + \
                  log_content.count('Unmatched parenthesis') + \
                  log_content.count('Format error')

    print(f"Detected ~{error_count} error messages in log")
    print()

    # Show results summary
    total = len(test_data)
    errors = error_count
    success = total - errors

    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total examples: {total}")
    print(f"Successful: {success} ({success/total*100:.1f}%)")
    print(f"Failed: {errors} ({errors/total*100:.1f}%)")
    print()

    # Try to extract predictions from results
    # The JSON should have predictions
    if 'predictions' in results:
        predictions = results['predictions']
    elif isinstance(results, list):
        predictions = results
    else:
        print("Warning: Could not find predictions in results file")
        print(f"Results keys: {results.keys()}")
        predictions = []

    if not predictions:
        print()
        print("="*80)
        print("ERROR EXAMPLES FROM LOG")
        print("="*80)
        print()

        # Extract error snippets from log
        error_lines = []
        for line in log_content.split('\n'):
            if any(x in line for x in ['Error:', 'Duplicate', 'Unmatched', 'Format error']):
                error_lines.append(line.strip())

        # Group by type
        from collections import defaultdict
        errors_by_type = defaultdict(list)

        for line in error_lines:
            if 'Unmatched parenthesis' in line:
                errors_by_type['Unmatched Parentheses'].append(line)
            elif 'Duplicate node' in line:
                errors_by_type['Duplicate Nodes'].append(line)
            elif 'Node name not found' in line:
                errors_by_type['Node Not Found'].append(line)
            else:
                errors_by_type['Other'].append(line)

        for error_type, lines in sorted(errors_by_type.items(), key=lambda x: len(x[1]), reverse=True):
            print(f"\n{error_type}: {len(lines)} errors")
            print("-"*80)

            for i, line in enumerate(lines[:5], 1):
                print(f"{i}. {line[:200]}")
                if len(line) > 200:
                    print("   ...")

            if len(lines) > 5:
                print(f"   ... and {len(lines)-5} more")

        print()

    # Show detailed examples if we have predictions
    if predictions and len(predictions) > 0:
        print()
        print("="*80)
        print("DETAILED ERROR CASES")
        print("="*80)

        # Find failed cases
        failed_indices = []

        for i, pred in enumerate(predictions[:min(len(predictions), len(test_data))]):
            # Simple heuristic: check if AMR is valid
            pred_amr = pred.get('amr', pred.get('prediction', ''))

            if not pred_amr or len(pred_amr) < 5:
                failed_indices.append(i)
                continue

            # Check parentheses
            if pred_amr.count('(') != pred_amr.count(')'):
                failed_indices.append(i)
                continue

        print(f"\nFound {len(failed_indices)} likely failed predictions")
        print()

        # Show first 10 failures
        for idx in failed_indices[:10]:
            if idx >= len(test_data):
                continue

            test_item = test_data[idx]
            pred_item = predictions[idx] if idx < len(predictions) else {}

            print(f"\n{'='*80}")
            print(f"ERROR EXAMPLE #{idx+1}")
            print('='*80)
            print()

            print(f"Sentence:")
            print(f"  {test_item['sentence']}")
            print()

            print(f"Gold AMR:")
            for line in test_item['gold_amr'].split('\n')[:5]:
                print(f"  {line}")
            print()

            pred_amr = pred_item.get('amr', pred_item.get('prediction', '(no prediction)'))
            print(f"Predicted AMR:")
            print(f"  {pred_amr[:300]}")
            if len(pred_amr) > 300:
                print("  ...")
            print()

            # Analyze
            issues = analyze_error(pred_amr, test_item['gold_amr'])
            if issues:
                print(f"Issues detected:")
                for issue in issues:
                    print(f"  - {issue}")
            print()

    print()
    print("="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    print()
    print("1. Most errors are parenthesis-related")
    print("   → Implement post-processing to balance parentheses")
    print()
    print("2. Review prompt template")
    print("   → Maybe add explicit parenthesis balancing instruction")
    print()
    print("3. Consider constrained decoding")
    print("   → Force valid AMR structure during generation")
    print()

if __name__ == "__main__":
    main()
