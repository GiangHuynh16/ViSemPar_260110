#!/usr/bin/env python3
"""
Debug post-processing by showing before/after for sample predictions
This helps identify what's going wrong
"""

import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def load_model(checkpoint_path):
    """Load model"""
    print(f"Loading model from {checkpoint_path}...")

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    return model, tokenizer

def post_process_amr(amr_string: str) -> str:
    """Same post-processing as in evaluate_mtup_model.py"""
    import re

    if not amr_string or len(amr_string) < 3:
        return "(amr-empty)"

    amr = amr_string.strip()

    # Step 1: Remove prompt leakage
    prompt_artifacts = [
        r'\d+\s*bước',
        r'Hướng dẫn:',
        r'AMR hoàn chỉnh:',
        r'### NHIỆM VỤ',
        r'## Bước \d+',
    ]
    for pattern in prompt_artifacts:
        amr = re.sub(pattern, '', amr, flags=re.IGNORECASE)

    # Step 2: Extract valid AMR structure
    if '(' not in amr:
        return "(error / no-valid-structure)"

    first_paren = amr.index('(')
    amr = amr[first_paren:]

    # Step 3: Balance parentheses
    stack = []
    balanced = []

    for char in amr:
        if char == '(':
            stack.append(char)
            balanced.append(char)
        elif char == ')':
            if stack:
                stack.pop()
                balanced.append(char)
        else:
            balanced.append(char)

    while stack:
        balanced.append(')')
        stack.pop()

    amr = ''.join(balanced)

    # Step 4: Rename duplicate variables
    seen_vars = {}

    def rename_duplicate_var(match):
        var_name = match.group(1)
        concept = match.group(2)

        if var_name in seen_vars:
            counter = seen_vars[var_name]
            seen_vars[var_name] += 1
            new_var = f"{var_name}{counter}"
            return f"({new_var} / {concept}"
        else:
            seen_vars[var_name] = 2
            return match.group(0)

    var_pattern = r'\(([a-z][a-z0-9]*)\s*/\s*([^\s\)]+)'
    amr = re.sub(var_pattern, rename_duplicate_var, amr)

    # Step 5: Basic validation
    if not amr.startswith('('):
        amr = '(' + amr
    if not amr.endswith(')'):
        amr = amr + ')'

    # Step 6: Clean whitespace
    amr = re.sub(r'\s+', ' ', amr)
    amr = re.sub(r'\(\s+', '(', amr)
    amr = re.sub(r'\s+\)', ')', amr)

    return amr.strip()

def generate_prediction(model, tokenizer, sentence: str):
    """Generate prediction and show before/after post-processing"""

    full_prompt = f"""### NHIỆM VỤ: Chuyển đổi câu tiếng Việt sang AMR (2 bước)

### Câu cần phân tích:
{sentence}

### Kết quả phân tích:

## Bước 1 - Tạo cấu trúc AMR (chưa có biến):
"""

    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=512,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.eos_token_id
        )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract AMR (before post-processing)
    if "## Bước 2" in result:
        parts = result.split("## Bước 2")[1]
        if "AMR hoàn chỉnh:" in parts:
            raw_amr = parts.split("AMR hoàn chỉnh:")[-1].strip()
        else:
            raw_amr = parts.strip()
    else:
        raw_amr = result.strip()

    if '(' in raw_amr:
        first_paren = raw_amr.index('(')
        raw_amr = raw_amr[first_paren:].strip()

    # Apply post-processing
    processed_amr = post_process_amr(raw_amr)

    return raw_amr, processed_amr

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--test-file', type=str, required=True)
    parser.add_argument('--num-samples', type=int, default=5)

    args = parser.parse_args()

    print("="*80)
    print("POST-PROCESSING DEBUG")
    print("="*80)
    print()

    # Load model
    model, tokenizer = load_model(args.checkpoint)
    print()

    # Load test data
    print(f"Loading test data from {args.test_file}...")
    sentences = []
    gold_amrs = []

    current_sentence = None
    current_amr = []

    with open(args.test_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            if line.startswith('# ::snt'):
                current_sentence = line.replace('# ::snt', '').strip()
            elif line and not line.startswith('#'):
                current_amr.append(line)
            elif line == '' and current_sentence:
                if current_amr:
                    sentences.append(current_sentence)
                    gold_amrs.append('\n'.join(current_amr))
                current_sentence = None
                current_amr = []

    if current_sentence and current_amr:
        sentences.append(current_sentence)
        gold_amrs.append('\n'.join(current_amr))

    print(f"✓ Loaded {len(sentences)} test examples")
    print()

    # Generate for first N samples
    print("="*80)
    print(f"COMPARING FIRST {args.num_samples} SAMPLES")
    print("="*80)
    print()

    for i in range(min(args.num_samples, len(sentences))):
        print(f"\n{'='*80}")
        print(f"SAMPLE #{i+1}")
        print('='*80)
        print()

        sentence = sentences[i]
        gold = gold_amrs[i]

        print(f"Sentence:")
        print(f"  {sentence}")
        print()

        print(f"Gold AMR:")
        for line in gold.split('\n')[:5]:
            print(f"  {line}")
        print()

        # Generate
        raw_amr, processed_amr = generate_prediction(model, tokenizer, sentence)

        print(f"RAW Model Output (before post-processing):")
        print(f"  {raw_amr[:300]}")
        if len(raw_amr) > 300:
            print("  ...")
        print()

        print(f"AFTER Post-Processing:")
        print(f"  {processed_amr[:300]}")
        if len(processed_amr) > 300:
            print("  ...")
        print()

        # Analyze changes
        if raw_amr != processed_amr:
            print(f"⚠️  CHANGED by post-processing:")

            # Count parentheses
            raw_open = raw_amr.count('(')
            raw_close = raw_amr.count(')')
            proc_open = processed_amr.count('(')
            proc_close = processed_amr.count(')')

            if raw_open != raw_close:
                print(f"   - Balanced parens: {raw_open}( vs {raw_close}) → {proc_open}( vs {proc_close})")

            # Check length
            if len(processed_amr) < len(raw_amr) * 0.5:
                print(f"   - ⚠️  LENGTH REDUCED SIGNIFICANTLY: {len(raw_amr)} → {len(processed_amr)}")

            if len(processed_amr) < 10:
                print(f"   - ⚠️  OUTPUT TOO SHORT (likely over-cleaned)")

            # Check for duplicate fix
            if 'n2' in processed_amr or 't2' in processed_amr or 'c2' in processed_amr:
                print(f"   - Applied variable renaming (n→n2, etc.)")

        else:
            print(f"✓ No changes (post-processing had no effect)")

        print()

    print("="*80)
    print("SUMMARY")
    print("="*80)
    print()
    print("Look for patterns:")
    print("  1. Is post-processing removing too much content?")
    print("  2. Is variable renaming breaking the AMR?")
    print("  3. Are balanced parens causing semantic changes?")
    print()

if __name__ == "__main__":
    main()
