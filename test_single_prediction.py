#!/usr/bin/env python3
"""
Quick test to verify MTUP model can generate predictions
Tests on a single sentence to debug issues
"""

import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Import templates
sys.path.append('config')
from prompt_templates_fixed import MTUP_INFERENCE_TEMPLATE, MTUP_INFERENCE_STEP2_TEMPLATE

def test_single_prediction():
    """Test single sentence prediction"""

    print("=" * 80)
    print("MTUP SINGLE PREDICTION TEST")
    print("=" * 80)
    print()

    # Model path
    model_path = "outputs/mtup_fixed_20260104_082506/checkpoint-148"

    print(f"Loading model from: {model_path}")
    print()

    # Load model
    try:
        base_model_name = "Qwen/Qwen2.5-7B-Instruct"

        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)

        print("Loading base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

        print("Loading LoRA adapters...")
        model = PeftModel.from_pretrained(base_model, model_path)
        model = model.merge_and_unload()
        model.eval()

        print("✅ Model loaded successfully!")
        print()

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # Test sentence
    test_sentence = "Tôi nhớ lời anh chủ tịch xã."

    print("=" * 80)
    print("STAGE 1: Generate AMR without variables")
    print("=" * 80)
    print()
    print(f"Input sentence: {test_sentence}")
    print()

    # Stage 1 prompt
    prompt1 = MTUP_INFERENCE_TEMPLATE.format(sentence=test_sentence)

    print("Prompt for Stage 1:")
    print("-" * 40)
    print(prompt1)
    print("-" * 40)
    print()

    # Generate Stage 1
    try:
        inputs = tokenizer(prompt1, return_tensors="pt").to(model.device)

        print("Generating Stage 1...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id
            )

        response1 = tokenizer.decode(outputs[0], skip_special_tokens=True)

        print("Full response from model:")
        print("-" * 40)
        print(response1)
        print("-" * 40)
        print()

        # Extract AMR without vars
        marker = "AMR không biến:"
        if marker in response1:
            amr_no_vars = response1.split(marker)[-1].strip()
            # Take until end or next marker
            if '\n\n' in amr_no_vars:
                amr_no_vars = amr_no_vars.split('\n\n')[0].strip()

            print(f"Extracted AMR (no vars): {amr_no_vars}")
        else:
            print("⚠️  Warning: Could not find marker in response")
            amr_no_vars = "(fallback)"

        print()

    except Exception as e:
        print(f"❌ Error in Stage 1: {e}")
        import traceback
        traceback.print_exc()
        return

    print("=" * 80)
    print("STAGE 2: Add variables (Penman format)")
    print("=" * 80)
    print()

    # Stage 2 prompt
    prompt2 = MTUP_INFERENCE_STEP2_TEMPLATE.format(
        sentence=test_sentence,
        amr_no_vars=amr_no_vars
    )

    print("Prompt for Stage 2:")
    print("-" * 40)
    print(prompt2)
    print("-" * 40)
    print()

    # Generate Stage 2
    try:
        inputs = tokenizer(prompt2, return_tensors="pt").to(model.device)

        print("Generating Stage 2...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id
            )

        response2 = tokenizer.decode(outputs[0], skip_special_tokens=True)

        print("Full response from model:")
        print("-" * 40)
        print(response2)
        print("-" * 40)
        print()

        # Extract final AMR
        marker = "AMR chuẩn PENMAN:"
        if marker in response2:
            amr_final = response2.split(marker)[-1].strip()

            print("Extracted final AMR:")
            print("-" * 40)
            print(amr_final)
            print("-" * 40)
        else:
            print("⚠️  Warning: Could not find marker in response")

        print()

    except Exception as e:
        print(f"❌ Error in Stage 2: {e}")
        import traceback
        traceback.print_exc()
        return

    print("=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    test_single_prediction()
