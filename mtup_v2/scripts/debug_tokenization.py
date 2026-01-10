#!/usr/bin/env python3
"""
Debug tokenization to see what the model actually learns.

Usage:
    python mtup_v2/scripts/debug_tokenization.py \
        --data_path data/train_mtup_unified.txt
"""

import argparse
from transformers import AutoTokenizer


def main(args):
    print("🔍 DEBUG TOKENIZATION\n")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct",
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Read first sample
    with open(args.data_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Get first sample
    import re
    blocks = re.split(r'<\|im_end\|>\n\n(?=<\|im_start\|>system)', content.strip())
    blocks = [b + '<|im_end|>' if not b.endswith('<|im_end|>') else b for b in blocks]

    first_sample = blocks[0].strip()

    print("=" * 70)
    print("FIRST TRAINING SAMPLE (RAW TEXT):")
    print("=" * 70)
    print(first_sample)
    print("=" * 70 + "\n")

    # Tokenize
    tokens = tokenizer(first_sample, truncation=True, max_length=2048)

    print("=" * 70)
    print("TOKENIZED:")
    print("=" * 70)
    print(f"Total tokens: {len(tokens['input_ids'])}")
    print(f"\nFirst 50 token IDs: {tokens['input_ids'][:50]}")
    print("\n" + "=" * 70)
    print("DECODED TOKENS (first 50):")
    print("=" * 70)
    for i, token_id in enumerate(tokens['input_ids'][:50]):
        decoded = tokenizer.decode([token_id])
        print(f"{i:3d}: {token_id:6d} -> '{decoded}'")

    print("\n" + "=" * 70)
    print("FULL DECODED TEXT:")
    print("=" * 70)
    decoded_full = tokenizer.decode(tokens['input_ids'])
    print(decoded_full)
    print("=" * 70 + "\n")

    # Check if decoded matches original
    if decoded_full.strip() == first_sample.strip():
        print("✅ TOKENIZATION OK - Decoded matches original")
    else:
        print("❌ TOKENIZATION ISSUE - Decoded differs from original")
        print("\nDIFFERENCES:")
        print(f"Original length: {len(first_sample)}")
        print(f"Decoded length: {len(decoded_full)}")

    # Find assistant part
    print("\n" + "=" * 70)
    print("ASSISTANT PART (what model should learn):")
    print("=" * 70)

    if "<|im_start|>assistant\n" in first_sample:
        assistant_part = first_sample.split("<|im_start|>assistant\n")[-1].split("<|im_end|>")[0]
        print(assistant_part)

        # Check for Task 1 and Task 2
        if "Task 1:" in assistant_part and "Task 2:" in assistant_part:
            print("\n✅ Contains Task 1 and Task 2")
        else:
            print("\n❌ Missing Task 1 or Task 2!")
            print(f"  Has 'Task 1:': {'Task 1:' in assistant_part}")
            print(f"  Has 'Task 2:': {'Task 2:' in assistant_part}")
    else:
        print("❌ No assistant part found!")

    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug tokenization")
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/train_mtup_unified.txt",
        help="Path to training data"
    )
    args = parser.parse_args()
    main(args)
