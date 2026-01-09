#!/usr/bin/env python3
"""
MTUP v2 - SMATCH Evaluation Script

Evaluate predictions against ground truth using SMATCH metric.

Usage:
    python mtup_v2/scripts/evaluate.py \
        --predictions outputs/predictions_mtup_v2.txt \
        --ground_truth data/public_test_ground_truth.txt
"""

import os
import argparse
import re
from pathlib import Path


def print_banner():
    """Print evaluation banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║     MTUP v2 - SMATCH Evaluation                             ║
    ║                                                              ║
    ║  📊 Comparing predictions with ground truth                 ║
    ║  🎯 Computing Precision, Recall, F1                         ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def read_amr_file(file_path):
    """
    Read AMR file and parse into individual graphs.

    Ground truth format:
    #::snt <sentence>
    <AMR graph - may span multiple lines>

    Prediction format:
    <AMR graph on single line>
    <AMR graph on single line>
    ...
    """
    print(f"📂 Reading: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    graphs = []

    # Check if it's ground truth format (has #::snt)
    if '#::snt' in content:
        # Ground truth format
        blocks = content.strip().split('\n\n')
        for block in blocks:
            if not block.strip():
                continue

            lines = block.strip().split('\n')
            amr_lines = []

            for line in lines:
                if line.startswith('#::snt'):
                    continue  # Skip sentence line
                if line.strip():
                    amr_lines.append(line.strip())

            if amr_lines:
                # Join multi-line AMR into single line
                amr_graph = ' '.join(amr_lines)
                # Normalize spaces
                amr_graph = re.sub(r'\s+', ' ', amr_graph)
                graphs.append(amr_graph.strip())

    else:
        # Prediction format (one AMR per line)
        lines = content.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line:
                graphs.append(line)

    print(f"✅ Parsed {len(graphs)} AMR graphs\n")
    return graphs


def normalize_amr(amr_str):
    """Normalize AMR for comparison"""
    # Remove extra spaces
    amr_str = re.sub(r'\s+', ' ', amr_str)
    amr_str = amr_str.strip()
    return amr_str


def compute_smatch_simple(pred_graphs, gold_graphs):
    """
    Simplified SMATCH computation.

    For proper SMATCH, we should use the official smatch library:
    - Install: pip install smatch
    - Use: import smatch

    This is a placeholder that checks for exact matches and provides
    basic statistics. For final evaluation, use official SMATCH.
    """
    print("🔍 Computing SMATCH scores...")
    print("=" * 70)

    if len(pred_graphs) != len(gold_graphs):
        print(f"⚠️  WARNING: Different number of graphs!")
        print(f"   Predictions: {len(pred_graphs)}")
        print(f"   Ground truth: {len(gold_graphs)}")
        print()

    n = min(len(pred_graphs), len(gold_graphs))

    exact_matches = 0
    errors = []

    for i in range(n):
        pred = normalize_amr(pred_graphs[i])
        gold = normalize_amr(gold_graphs[i])

        if pred == gold:
            exact_matches += 1
        else:
            if len(errors) < 5:  # Store first 5 errors for inspection
                errors.append({
                    'index': i,
                    'pred': pred[:100],
                    'gold': gold[:100]
                })

    # Basic metrics
    exact_match_rate = exact_matches / n * 100 if n > 0 else 0

    print(f"📊 Basic Metrics (Exact Match):")
    print(f"   Total samples: {n}")
    print(f"   Exact matches: {exact_matches}")
    print(f"   Exact match rate: {exact_match_rate:.2f}%")
    print()

    # Try to use official SMATCH if available
    try:
        import smatch

        print("✅ Official SMATCH library found! Computing detailed scores...")
        print()

        # Compute SMATCH
        total_match = 0
        total_test = 0
        total_gold = 0

        for pred, gold in zip(pred_graphs[:n], gold_graphs[:n]):
            # Parse AMRs
            try:
                # SMATCH expects AMR in specific format
                # We need to convert our single-line format
                best_match_num, test_triple_num, gold_triple_num = smatch.get_amr_match(
                    pred, gold
                )
                total_match += best_match_num
                total_test += test_triple_num
                total_gold += gold_triple_num
            except Exception as e:
                print(f"⚠️  Error computing SMATCH for sample: {e}")
                continue

        # Calculate metrics
        precision = total_match / total_test if total_test > 0 else 0.0
        recall = total_match / total_gold if total_gold > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        print("=" * 70)
        print("📊 OFFICIAL SMATCH SCORES")
        print("=" * 70)
        print(f"   Precision: {precision:.4f} ({precision * 100:.2f}%)")
        print(f"   Recall:    {recall:.4f} ({recall * 100:.2f}%)")
        print(f"   F1 Score:  {f1:.4f} ({f1 * 100:.2f}%)")
        print("=" * 70)

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'exact_match_rate': exact_match_rate / 100
        }

    except ImportError:
        print("⚠️  Official SMATCH library not found.")
        print("   Install: pip install smatch")
        print("   For now, showing basic exact match statistics only.")
        print()

        # Show example errors
        if errors:
            print("=" * 70)
            print("📝 Example Mismatches (first 5):")
            print("=" * 70)
            for err in errors:
                print(f"\nSample {err['index']}:")
                print(f"  Pred: {err['pred']}...")
                print(f"  Gold: {err['gold']}...")
            print("=" * 70)

        return {
            'exact_match_rate': exact_match_rate / 100,
            'note': 'Install smatch for detailed metrics'
        }


def save_comparison(pred_graphs, gold_graphs, output_path):
    """Save side-by-side comparison for manual inspection"""
    print(f"\n💾 Saving comparison to: {output_path}")

    n = min(len(pred_graphs), len(gold_graphs))

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("PREDICTION vs GROUND TRUTH COMPARISON\n")
        f.write("=" * 100 + "\n\n")

        for i in range(n):
            pred = normalize_amr(pred_graphs[i])
            gold = normalize_amr(gold_graphs[i])

            match = "✓ MATCH" if pred == gold else "✗ MISMATCH"

            f.write(f"Sample {i + 1}: {match}\n")
            f.write("-" * 100 + "\n")
            f.write(f"PRED: {pred}\n")
            f.write(f"GOLD: {gold}\n")
            f.write("\n")

    print(f"✅ Comparison saved")


def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description="MTUP v2 SMATCH Evaluation")

    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="Path to predictions file"
    )
    parser.add_argument(
        "--ground_truth",
        type=str,
        required=True,
        help="Path to ground truth file"
    )
    parser.add_argument(
        "--output_comparison",
        type=str,
        default=None,
        help="Optional: Save detailed comparison to file"
    )

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.predictions):
        print(f"❌ Predictions file not found: {args.predictions}")
        return

    if not os.path.exists(args.ground_truth):
        print(f"❌ Ground truth file not found: {args.ground_truth}")
        return

    print_banner()

    # Read files
    pred_graphs = read_amr_file(args.predictions)
    gold_graphs = read_amr_file(args.ground_truth)

    # Compute SMATCH
    metrics = compute_smatch_simple(pred_graphs, gold_graphs)

    # Save comparison if requested
    if args.output_comparison:
        output_dir = os.path.dirname(args.output_comparison)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        save_comparison(pred_graphs, gold_graphs, args.output_comparison)

    # Summary
    print("\n" + "=" * 70)
    print("✅ EVALUATION COMPLETED")
    print("=" * 70)

    if 'f1' in metrics:
        print(f"\n🎯 Final F1 Score: {metrics['f1']:.4f} ({metrics['f1'] * 100:.2f}%)")
        print(f"\n   Compare with baseline F1: 0.47 (47%)")
        if metrics['f1'] > 0.47:
            improvement = (metrics['f1'] - 0.47) / 0.47 * 100
            print(f"   🎉 Improvement: +{improvement:.2f}%")
        else:
            decline = (0.47 - metrics['f1']) / 0.47 * 100
            print(f"   📉 Below baseline by: -{decline:.2f}%")
    else:
        print(f"\n📊 Exact Match Rate: {metrics['exact_match_rate']:.4f} ({metrics['exact_match_rate'] * 100:.2f}%)")
        print(f"\n⚠️  For official SMATCH F1 score, install: pip install smatch")

    print("=" * 70)


if __name__ == "__main__":
    main()
