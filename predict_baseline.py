#!/usr/bin/env python3
"""
Predict AMR using baseline 7B model and compare with ground truth
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse
from pathlib import Path
from tqdm import tqdm
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))
from config.config import PROMPT_TEMPLATE


def load_model(checkpoint_path):
    """Load model from checkpoint"""
    print(f"Loading model from {checkpoint_path}...")

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    # Load LoRA adapter
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    model.eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

    print("✓ Model loaded successfully")
    return model, tokenizer


def read_test_file(test_file):
    """Read test sentences"""
    sentences = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                sentences.append(line)
    return sentences


def predict_amr(model, tokenizer, sentence, max_new_tokens=512):
    """Predict AMR for a sentence"""
    # Create prompt
    prompt = PROMPT_TEMPLATE.format(sentence=sentence)

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.15,
            do_sample=True,
        )

    # Decode
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract AMR (everything after "AMR:\n")
    if "AMR:" in full_output:
        amr = full_output.split("AMR:")[-1].strip()
    else:
        amr = full_output.strip()

    return amr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--test-file', type=str, default='data/public_test.txt', help='Test file')
    parser.add_argument('--output', type=str, default='public_test_result_baseline_7b.txt', help='Output file')
    parser.add_argument('--ground-truth', type=str, default='data/public_test_ground_truth.txt', help='Ground truth file')
    args = parser.parse_args()

    print("=" * 70)
    print("BASELINE 7B PREDICTION")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test file: {args.test_file}")
    print(f"Output: {args.output}")
    print()

    # Load model
    model, tokenizer = load_model(args.checkpoint)

    # Read test sentences
    print(f"Reading test file: {args.test_file}")
    sentences = read_test_file(args.test_file)
    print(f"✓ Found {len(sentences)} sentences")
    print()

    # Predict
    print("Predicting AMRs...")
    predictions = []
    for i, sentence in enumerate(tqdm(sentences, desc="Progress")):
        amr = predict_amr(model, tokenizer, sentence)
        predictions.append(amr)

        # Print first few examples
        if i < 3:
            print(f"\nExample {i+1}:")
            print(f"  Sentence: {sentence}")
            print(f"  AMR: {amr[:100]}...")

    print()

    # Save predictions
    print(f"Saving predictions to {args.output}...")
    with open(args.output, 'w', encoding='utf-8') as f:
        for amr in predictions:
            f.write(amr + '\n')
    print(f"✓ Saved {len(predictions)} predictions")
    print()

    # Compare with ground truth if available
    if Path(args.ground_truth).exists():
        print(f"Reading ground truth: {args.ground_truth}")
        with open(args.ground_truth, 'r', encoding='utf-8') as f:
            ground_truth = [line.strip() for line in f if line.strip()]

        print(f"✓ Found {len(ground_truth)} ground truth AMRs")
        print()

        # Show comparison examples
        print("=" * 70)
        print("COMPARISON EXAMPLES")
        print("=" * 70)
        for i in range(min(3, len(sentences))):
            print(f"\nExample {i+1}:")
            print(f"  Sentence: {sentences[i]}")
            print(f"  Predicted: {predictions[i][:150]}...")
            if i < len(ground_truth):
                print(f"  Ground Truth: {ground_truth[i][:150]}...")
        print()

        print("=" * 70)
        print(f"Prediction complete! Results saved to: {args.output}")
        print("=" * 70)
        print()
        print("Next steps:")
        print(f"  1. Calculate SMATCH score:")
        print(f"     python calculate_smatch.py --predictions {args.output} --gold {args.ground_truth}")
        print()
        print(f"  2. View full results:")
        print(f"     less {args.output}")
        print()
    else:
        print(f"Ground truth file not found: {args.ground_truth}")
        print(f"Predictions saved to: {args.output}")


if __name__ == "__main__":
    main()
