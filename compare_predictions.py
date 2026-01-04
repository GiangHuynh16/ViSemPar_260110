#!/usr/bin/env python3
"""
Compare MTUP predictions with ground truth to identify issues
"""

import sys
import re

def load_amrs(file_path):
    """Load AMRs from file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split by double newline
    amrs = [amr.strip() for amr in content.strip().split('\n\n') if amr.strip()]
    return amrs


def analyze_amr(amr):
    """Analyze AMR structure"""
    lines = amr.split('\n')

    # Count parentheses
    open_paren = amr.count('(')
    close_paren = amr.count(')')

    # Count variables
    vars_pattern = r'\((\w+)\s*/\s*[\w_\-]+'
    variables = re.findall(vars_pattern, amr)

    # Count relations
    rel_pattern = r':[\w_\-]+'
    relations = re.findall(rel_pattern, amr)

    return {
        'lines': len(lines),
        'open_paren': open_paren,
        'close_paren': close_paren,
        'balanced': open_paren == close_paren,
        'variables': len(variables),
        'relations': len(relations),
        'chars': len(amr),
        'variable_list': variables,
        'relation_list': relations[:5],  # First 5
    }


def main():
    if len(sys.argv) < 3:
        print("Usage: python compare_predictions.py <predictions_file> <ground_truth_file>")
        print()
        print("Example:")
        print("  python compare_predictions.py \\")
        print("    evaluation_results/mtup_predictions_FIXED.txt \\")
        print("    data/public_test_ground_truth.txt")
        sys.exit(1)

    pred_file = sys.argv[1]
    truth_file = sys.argv[2]

    print("=" * 80)
    print("COMPARING PREDICTIONS WITH GROUND TRUTH")
    print("=" * 80)
    print()

    # Load files
    print(f"Loading predictions: {pred_file}")
    try:
        predictions = load_amrs(pred_file)
        print(f"  ✅ Loaded {len(predictions)} predictions")
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return

    print()
    print(f"Loading ground truth: {truth_file}")
    try:
        ground_truth = load_amrs(truth_file)
        print(f"  ✅ Loaded {len(ground_truth)} ground truth AMRs")
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return

    print()

    # Compare counts
    if len(predictions) != len(ground_truth):
        print("⚠️  WARNING: Different number of AMRs!")
        print(f"   Predictions: {len(predictions)}")
        print(f"   Ground truth: {len(ground_truth)}")
        print()

    # Analyze first 5
    num_samples = min(5, len(predictions), len(ground_truth))

    print("=" * 80)
    print(f"DETAILED COMPARISON (First {num_samples} examples)")
    print("=" * 80)
    print()

    for i in range(num_samples):
        print(f"Example {i+1}/{num_samples}")
        print("-" * 80)
        print()

        pred = predictions[i] if i < len(predictions) else "(missing)"
        truth = ground_truth[i] if i < len(ground_truth) else "(missing)"

        print("PREDICTION:")
        print(pred)
        print()

        print("GROUND TRUTH:")
        print(truth)
        print()

        if pred != "(missing)" and truth != "(missing)":
            pred_analysis = analyze_amr(pred)
            truth_analysis = analyze_amr(truth)

            print("ANALYSIS:")
            print(f"  Prediction:   {pred_analysis['lines']} lines, "
                  f"{pred_analysis['variables']} vars, "
                  f"{pred_analysis['relations']} rels, "
                  f"{'✅' if pred_analysis['balanced'] else '❌'} balanced")

            print(f"  Ground truth: {truth_analysis['lines']} lines, "
                  f"{truth_analysis['variables']} vars, "
                  f"{truth_analysis['relations']} rels, "
                  f"{'✅' if truth_analysis['balanced'] else '❌'} balanced")
            print()

            # Identify issues
            issues = []

            if pred_analysis['lines'] < truth_analysis['lines']:
                issues.append(f"❌ TOO SHORT: {pred_analysis['lines']} vs {truth_analysis['lines']} lines")

            if pred_analysis['variables'] < truth_analysis['variables']:
                issues.append(f"❌ MISSING VARIABLES: {pred_analysis['variables']} vs {truth_analysis['variables']}")

            if pred_analysis['relations'] < truth_analysis['relations']:
                issues.append(f"❌ MISSING RELATIONS: {pred_analysis['relations']} vs {truth_analysis['relations']}")

            if not pred_analysis['balanced']:
                issues.append(f"❌ UNBALANCED PARENTHESES: {pred_analysis['open_paren']} open, {pred_analysis['close_paren']} close")

            if pred_analysis['chars'] < truth_analysis['chars'] * 0.5:
                issues.append(f"❌ TOO SHORT (chars): {pred_analysis['chars']} vs {truth_analysis['chars']}")

            if issues:
                print("ISSUES:")
                for issue in issues:
                    print(f"  {issue}")
            else:
                print("✅ Structure looks similar")

            print()

        print("=" * 80)
        print()

    # Overall statistics
    print()
    print("=" * 80)
    print("OVERALL STATISTICS")
    print("=" * 80)
    print()

    pred_stats = [analyze_amr(p) for p in predictions]
    truth_stats = [analyze_amr(t) for t in ground_truth]

    avg_pred_lines = sum(s['lines'] for s in pred_stats) / len(pred_stats) if pred_stats else 0
    avg_truth_lines = sum(s['lines'] for s in truth_stats) / len(truth_stats) if truth_stats else 0

    avg_pred_vars = sum(s['variables'] for s in pred_stats) / len(pred_stats) if pred_stats else 0
    avg_truth_vars = sum(s['variables'] for s in truth_stats) / len(truth_stats) if truth_stats else 0

    avg_pred_rels = sum(s['relations'] for s in pred_stats) / len(pred_stats) if pred_stats else 0
    avg_truth_rels = sum(s['relations'] for s in truth_stats) / len(truth_stats) if truth_stats else 0

    balanced_pred = sum(1 for s in pred_stats if s['balanced'])
    balanced_truth = sum(1 for s in truth_stats if s['balanced'])

    print(f"Predictions ({len(predictions)} total):")
    print(f"  Avg lines: {avg_pred_lines:.1f}")
    print(f"  Avg variables: {avg_pred_vars:.1f}")
    print(f"  Avg relations: {avg_pred_rels:.1f}")
    print(f"  Balanced: {balanced_pred}/{len(predictions)} ({balanced_pred/len(predictions)*100:.1f}%)")
    print()

    print(f"Ground truth ({len(ground_truth)} total):")
    print(f"  Avg lines: {avg_truth_lines:.1f}")
    print(f"  Avg variables: {avg_truth_vars:.1f}")
    print(f"  Avg relations: {avg_truth_rels:.1f}")
    print(f"  Balanced: {balanced_truth}/{len(ground_truth)} ({balanced_truth/len(ground_truth)*100:.1f}%)")
    print()

    print("COMPARISON:")
    if avg_pred_lines < avg_truth_lines * 0.7:
        print(f"  ❌ Predictions TOO SHORT ({avg_pred_lines:.1f} vs {avg_truth_lines:.1f} lines)")
    elif avg_pred_lines < avg_truth_lines * 0.9:
        print(f"  ⚠️  Predictions slightly shorter ({avg_pred_lines:.1f} vs {avg_truth_lines:.1f} lines)")
    else:
        print(f"  ✅ Line count similar ({avg_pred_lines:.1f} vs {avg_truth_lines:.1f} lines)")

    if avg_pred_vars < avg_truth_vars * 0.7:
        print(f"  ❌ Predictions MISSING VARIABLES ({avg_pred_vars:.1f} vs {avg_truth_vars:.1f})")
    else:
        print(f"  ✅ Variable count similar ({avg_pred_vars:.1f} vs {avg_truth_vars:.1f})")

    if avg_pred_rels < avg_truth_rels * 0.7:
        print(f"  ❌ Predictions MISSING RELATIONS ({avg_pred_rels:.1f} vs {avg_truth_rels:.1f})")
    else:
        print(f"  ✅ Relation count similar ({avg_pred_rels:.1f} vs {avg_truth_rels:.1f})")

    print()
    print("=" * 80)
    print("DIAGNOSIS")
    print("=" * 80)
    print()

    if avg_pred_lines < avg_truth_lines * 0.7:
        print("🔴 PRIMARY ISSUE: Model generating TOO SHORT output")
        print()
        print("Possible causes:")
        print("  1. Extraction logic stopping too early")
        print("  2. Model learned to generate short AMRs (training issue)")
        print("  3. Generation config (max_new_tokens too low)")
        print("  4. EOS token generated too early")
        print()
        print("Recommended fixes:")
        print("  1. Check extraction logic in predict_mtup_fixed.py")
        print("  2. Increase max_new_tokens in config")
        print("  3. Review training data and prompts")
        print("  4. Re-train model with corrected prompts")

    elif avg_pred_vars < avg_truth_vars * 0.7:
        print("🟡 ISSUE: Model not generating enough variables")
        print()
        print("This suggests:")
        print("  - Step 2 (variable binding) not working correctly")
        print("  - Model didn't learn Penman format properly")
        print()
        print("Recommended fixes:")
        print("  1. Check Step 2 prompt template")
        print("  2. Verify training examples have correct Penman format")
        print("  3. Re-train with fixed templates")

    else:
        print("🟢 Structure looks reasonable")
        print()
        print("If F-score is still low, the issue is likely:")
        print("  - Wrong concept selection")
        print("  - Wrong relation types")
        print("  - Wrong graph structure (edges)")
        print()
        print("This suggests more training data or better prompts needed.")

    print()


if __name__ == '__main__':
    main()
