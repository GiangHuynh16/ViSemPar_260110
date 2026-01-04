#!/usr/bin/env python3
"""
Debug MTUP generation to see what's wrong
"""

import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

sys.path.append('config')
from prompt_templates_fixed import MTUP_INFERENCE_TEMPLATE, MTUP_INFERENCE_STEP2_TEMPLATE
import config_mtup_fixed as config

def debug_generation():
    """Debug MTUP generation step by step"""

    print("=" * 80)
    print("MTUP GENERATION DEBUG")
    print("=" * 80)
    print()

    # Test sentence
    test_sentence = "Tôi nhớ lời anh chủ tịch xã."
    expected_no_vars = "(nhớ :pivot (tôi) :theme (lời :poss (chủ_tịch :mod (anh) :mod (xã))))"
    expected_with_vars = """(n / nhớ
:pivot(t / tôi)
:theme(l / lời
:poss(c / chủ_tịch
:mod(a / anh)
:mod(x / xã))))"""

    print(f"Test sentence: {test_sentence}")
    print()
    print(f"Expected Step 1 (no vars):")
    print(expected_no_vars)
    print()
    print(f"Expected Step 2 (with vars):")
    print(expected_with_vars)
    print()

    # Load model
    model_path = "outputs/mtup_fixed_20260104_082506/checkpoint-148"

    print("=" * 80)
    print("LOADING MODEL")
    print("=" * 80)
    print(f"Model: {model_path}")
    print()

    try:
        tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            config.MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

        model = PeftModel.from_pretrained(base_model, model_path)
        model.eval()

        print("✅ Model loaded!")
        print()

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print()
        print("Available checkpoints:")
        import os
        for root, dirs, files in os.walk("outputs"):
            if "checkpoint" in root:
                print(f"  {root}")
        return

    # STEP 1: Generate AMR without variables
    print("=" * 80)
    print("STEP 1: GENERATE AMR WITHOUT VARIABLES")
    print("=" * 80)
    print()

    prompt1 = MTUP_INFERENCE_TEMPLATE.format(sentence=test_sentence)

    print("Prompt:")
    print("-" * 40)
    print(prompt1)
    print("-" * 40)
    print()

    # Tokenize
    inputs = tokenizer(prompt1, return_tensors="pt").to(model.device)

    print(f"Input tokens: {inputs['input_ids'].shape[1]}")
    print()

    # Generate
    print("Generating...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=config.INFERENCE_CONFIG['max_new_tokens'],
            temperature=config.INFERENCE_CONFIG['temperature'],
            top_p=config.INFERENCE_CONFIG['top_p'],
            do_sample=config.INFERENCE_CONFIG['do_sample'],
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    response1 = tokenizer.decode(outputs[0], skip_special_tokens=False)

    print("Full response:")
    print("=" * 80)
    print(response1)
    print("=" * 80)
    print()

    # Extract AMR no vars
    if "AMR không biến:" in response1:
        amr_no_vars = response1.split("AMR không biến:")[-1].strip()
        # Extract until balanced
        lines = amr_no_vars.split('\n')
        amr_lines = []
        for line in lines:
            amr_lines.append(line)
            accumulated = '\n'.join(amr_lines)
            if accumulated.count('(') == accumulated.count(')') > 0:
                break
        amr_no_vars = '\n'.join(amr_lines).strip()
    else:
        amr_no_vars = "(error)"

    print("Extracted AMR (no vars):")
    print("-" * 40)
    print(amr_no_vars)
    print("-" * 40)
    print()

    print("Expected:")
    print("-" * 40)
    print(expected_no_vars)
    print("-" * 40)
    print()

    if amr_no_vars == expected_no_vars:
        print("✅ STEP 1 CORRECT!")
    else:
        print("❌ STEP 1 WRONG!")
        print()
        print("Differences:")
        print(f"  Length: {len(amr_no_vars)} vs {len(expected_no_vars)}")
        print(f"  Paren balance: {amr_no_vars.count('(')} open, {amr_no_vars.count(')')} close")

    print()

    # STEP 2: Add variables
    print("=" * 80)
    print("STEP 2: ADD VARIABLES (PENMAN FORMAT)")
    print("=" * 80)
    print()

    prompt2 = MTUP_INFERENCE_STEP2_TEMPLATE.format(
        sentence=test_sentence,
        amr_no_vars=amr_no_vars
    )

    print("Prompt:")
    print("-" * 40)
    print(prompt2)
    print("-" * 40)
    print()

    # Tokenize
    inputs = tokenizer(prompt2, return_tensors="pt").to(model.device)

    print(f"Input tokens: {inputs['input_ids'].shape[1]}")
    print()

    # Generate
    print("Generating...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=config.INFERENCE_CONFIG['max_new_tokens'],
            temperature=config.INFERENCE_CONFIG['temperature'],
            top_p=config.INFERENCE_CONFIG['top_p'],
            do_sample=config.INFERENCE_CONFIG['do_sample'],
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    response2 = tokenizer.decode(outputs[0], skip_special_tokens=False)

    print("Full response:")
    print("=" * 80)
    print(response2)
    print("=" * 80)
    print()

    # Extract final AMR
    if "AMR chuẩn PENMAN:" in response2:
        amr_with_vars = response2.split("AMR chuẩn PENMAN:")[-1].strip()
        # Extract until balanced
        lines = amr_with_vars.split('\n')
        amr_lines = []
        for line in lines:
            if not amr_lines and not line.strip():
                continue
            amr_lines.append(line)
            accumulated = '\n'.join(amr_lines)
            if accumulated.count('(') == accumulated.count(')') > 0:
                break
        amr_with_vars = '\n'.join(amr_lines).strip()
    else:
        amr_with_vars = "(error)"

    print("Extracted AMR (with vars):")
    print("-" * 40)
    print(amr_with_vars)
    print("-" * 40)
    print()

    print("Expected:")
    print("-" * 40)
    print(expected_with_vars)
    print("-" * 40)
    print()

    # Analysis
    print()
    print("=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    print()

    # Check structure
    actual_lines = amr_with_vars.split('\n')
    expected_lines = expected_with_vars.split('\n')

    print(f"Actual lines: {len(actual_lines)}")
    print(f"Expected lines: {len(expected_lines)}")
    print()

    print(f"Actual parentheses: {amr_with_vars.count('(')} open, {amr_with_vars.count(')')} close")
    print(f"Expected parentheses: {expected_with_vars.count('(')} open, {expected_with_vars.count(')')} close")
    print()

    # Check variables
    import re
    actual_vars = re.findall(r'\((\w+)\s*/\s*[\w_]+', amr_with_vars)
    expected_vars = re.findall(r'\((\w+)\s*/\s*[\w_]+', expected_with_vars)

    print(f"Actual variables: {actual_vars}")
    print(f"Expected variables: {expected_vars}")
    print()

    if amr_with_vars == expected_with_vars:
        print("✅ STEP 2 PERFECT!")
    elif len(actual_lines) < len(expected_lines):
        print("❌ STEP 2 TOO SHORT!")
        print(f"   Missing {len(expected_lines) - len(actual_lines)} lines")
    elif amr_with_vars.count('(') != amr_with_vars.count(')'):
        print("❌ UNBALANCED PARENTHESES!")
    else:
        print("⚠️  STEP 2 DIFFERENT but valid structure")

    print()
    print("=" * 80)
    print("DEBUG COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    debug_generation()
