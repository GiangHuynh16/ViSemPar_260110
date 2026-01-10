#!/usr/bin/env python3
"""
Diagnose why model training failed.

Usage:
    python mtup_v2/scripts/diagnose_model.py \
        --data_path data/train_mtup_unified.txt \
        --adapter_path outputs/mtup_260110/mtup_v2/final_adapter
"""

import argparse
import os
import re


def check_data_integrity(data_path):
    """Check if training data is correct"""
    print("=" * 70)
    print("1. CHECKING DATA INTEGRITY")
    print("=" * 70)

    with open(data_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split into samples
    blocks = re.split(r'<\|im_end\|>\n\n(?=<\|im_start\|>system)', content.strip())
    blocks = [b + '<|im_end|>' if not b.endswith('<|im_end|>') else b for b in blocks]

    print(f"Total samples: {len(blocks)}")

    # Check first sample
    first_sample = blocks[0]
    print("\nFirst sample content check:")

    checks = {
        "Has system prompt": "<|im_start|>system" in first_sample,
        "Has user input": "<|im_start|>user" in first_sample,
        "Has assistant output": "<|im_start|>assistant" in first_sample,
        "Has Task 1": "Task 1:" in first_sample,
        "Has Task 2": "Task 2:" in first_sample,
        "Has 'bi kịch' sentence": "bi kịch là ở chỗ đó" in first_sample,
    }

    for check, result in checks.items():
        status = "✅" if result else "❌"
        print(f"  {status} {check}")

    # Check if Vietnamese characters are intact
    vietnamese_chars = ['ả', 'ã', 'á', 'à', 'ạ', 'ă', 'ắ', 'ằ', 'ẳ', 'ẵ', 'ặ',
                       'â', 'ấ', 'ầ', 'ẩ', 'ẫ', 'ậ', 'đ', 'é', 'è', 'ẻ', 'ẽ', 'ẹ',
                       'ê', 'ế', 'ề', 'ể', 'ễ', 'ệ', 'í', 'ì', 'ỉ', 'ĩ', 'ị',
                       'ó', 'ò', 'ỏ', 'õ', 'ọ', 'ô', 'ố', 'ồ', 'ổ', 'ỗ', 'ộ',
                       'ơ', 'ớ', 'ờ', 'ở', 'ỡ', 'ợ', 'ú', 'ù', 'ủ', 'ũ', 'ụ',
                       'ư', 'ứ', 'ừ', 'ử', 'ữ', 'ự', 'ý', 'ỳ', 'ỷ', 'ỹ', 'ỵ']

    viet_count = sum(1 for char in vietnamese_chars if char in first_sample)
    print(f"\n  Found {viet_count} different Vietnamese characters - {'✅ OK' if viet_count > 20 else '❌ CORRUPTED'}")

    # Show first sample's assistant part
    print("\n" + "=" * 70)
    print("FIRST SAMPLE ASSISTANT OUTPUT:")
    print("=" * 70)
    if "<|im_start|>assistant" in first_sample:
        assistant_part = first_sample.split("<|im_start|>assistant\n")[-1].split("<|im_end|>")[0]
        print(assistant_part)
    print("=" * 70)

    return len(blocks) > 0 and checks["Has Task 1"] and checks["Has Task 2"]


def check_training_logs(adapter_path):
    """Check training logs if available"""
    print("\n" + "=" * 70)
    print("2. CHECKING TRAINING LOGS")
    print("=" * 70)

    trainer_state_path = os.path.join(os.path.dirname(adapter_path), "trainer_state.json")

    if os.path.exists(trainer_state_path):
        import json
        with open(trainer_state_path, 'r') as f:
            state = json.load(f)

        print(f"✅ Found training logs")
        print(f"\nTotal epochs completed: {state.get('epoch', 'unknown')}")
        print(f"Global step: {state.get('global_step', 'unknown')}")

        # Check loss progression
        if 'log_history' in state and len(state['log_history']) > 0:
            print("\nLoss progression:")
            logs = [log for log in state['log_history'] if 'loss' in log]
            if len(logs) > 0:
                first_loss = logs[0]['loss']
                last_loss = logs[-1]['loss']
                print(f"  First loss: {first_loss:.4f}")
                print(f"  Last loss: {last_loss:.4f}")
                print(f"  Reduction: {((first_loss - last_loss) / first_loss * 100):.1f}%")

                if last_loss > first_loss:
                    print("  ❌ WARNING: Loss INCREASED - model is not learning!")
                elif last_loss > first_loss * 0.7:
                    print("  ⚠️  WARNING: Loss only reduced by <30% - model may not have converged")
                else:
                    print("  ✅ Loss reduced significantly")

        return True
    else:
        print("❌ No training logs found")
        print(f"   Expected at: {trainer_state_path}")
        return False


def recommend_actions(data_ok, logs_ok):
    """Recommend next steps based on diagnosis"""
    print("\n" + "=" * 70)
    print("3. RECOMMENDATIONS")
    print("=" * 70)

    if not data_ok:
        print("❌ CRITICAL: Training data is corrupted!")
        print("\nACTION REQUIRED:")
        print("1. Verify data file encoding on server:")
        print("   file -i data/train_mtup_unified.txt")
        print("2. Re-download or regenerate training data")
        print("3. Use 'git config core.quotepath false' to prevent encoding issues")

    elif logs_ok:
        print("Data is OK but model failed to learn properly.")
        print("\nRECOMMENDED ACTIONS (in order):")
        print("\n📊 Option 1: Increase model capacity")
        print("  - Increase LoRA rank from 32 to 64")
        print("  - Train for 20-25 epochs instead of 15")
        print("\n📊 Option 2: Simplify the task")
        print("  - Train ONLY on Task 2 (remove Task 1 from training data)")
        print("  - Single-task learning might be more stable")
        print("\n📊 Option 3: Change approach")
        print("  - Try training 2 separate models:")
        print("    * Model 1: Sentence -> Task 1 (skeleton)")
        print("    * Model 2: Task 1 -> Task 2 (add variables)")
        print("  - Sequential approach might work better")
        print("\n📊 Option 4: Use larger base model")
        print("  - Try Qwen2.5-14B-Instruct (if you have enough VRAM)")

    else:
        print("⚠️  Cannot determine exact issue - no training logs available")
        print("\nRECOMMENDED: Run diagnosis again with access to training outputs")


def main(args):
    print("\n🔍 MTUP v2 MODEL DIAGNOSIS\n")

    data_ok = check_data_integrity(args.data_path)
    logs_ok = check_training_logs(args.adapter_path)

    recommend_actions(data_ok, logs_ok)

    print("\n" + "=" * 70)
    print("DIAGNOSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnose MTUP v2 model training")

    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to training data"
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        required=True,
        help="Path to trained adapter"
    )

    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        print(f"❌ Data file not found: {args.data_path}")
        exit(1)

    if not os.path.exists(args.adapter_path):
        print(f"❌ Adapter not found: {args.adapter_path}")
        exit(1)

    main(args)
