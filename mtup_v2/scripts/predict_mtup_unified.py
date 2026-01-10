#!/usr/bin/env python3
"""
MTUP v2 - Unified Prediction Script

Predict using trained MTUP unified model.
Extracts Task 2 output (Full AMR with variables) in PENMAN format.

Usage:
    python mtup_v2/scripts/predict_mtup_unified.py \
        --base_model Qwen/Qwen2.5-7B-Instruct \
        --adapter_path outputs/mtup_v2_unified/final_adapter \
        --input_file data/public_test.txt \
        --output_file outputs/predictions_mtup_v2.txt
"""

import os
import argparse
import torch
import re
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm


def print_banner():
    """Print prediction banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║     MTUP v2 - Unified Model Prediction (No Quantization)    ║
    ║                                                              ║
    ║  🎯 Extracting Task 2 (Full AMR with variables)             ║
    ║  📊 PENMAN Format Output                                    ║
    ║  ⚡ bfloat16 - Matches training configuration               ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def load_model(base_model_path, adapter_path):
    """Load base model and LoRA adapter (bfloat16, no quantization)"""
    print(f"📥 Loading base model: {base_model_path}")

    # Try to use Flash Attention 2 if available, otherwise fall back to SDPA
    try:
        import flash_attn
        attn_impl = "flash_attention_2" if torch.cuda.get_device_capability()[0] >= 8 else "sdpa"
        print(f"   Using attention implementation: {attn_impl}")
    except ImportError:
        attn_impl = "sdpa"
        print(f"   Flash Attention not installed, using SDPA (still fast!)")

    # Load model with bfloat16 (same as training)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation=attn_impl
    )

    print(f"🔗 Loading LoRA adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    print("✅ Model loaded successfully\n")
    return model, tokenizer


def create_unified_prompt(sentence):
    """
    Create unified prompt for inference.

    NOTE: We use the SAME prompt structure as training.
    """
    system_prompt = """Bạn là chuyên gia phân tích AMR (Abstract Meaning Representation) cho tiếng Việt.
Nhiệm vụ: Với mỗi câu tiếng Việt, sinh ra 2 output:

Task 1 - AMR Skeleton: Cấu trúc AMR chỉ có concept và relation, KHÔNG có biến định danh.
Task 2 - Full AMR: AMR hoàn chỉnh với biến theo chuẩn PENMAN.

Quy tắc QUAN TRỌNG cho Task 2:
1. Mỗi concept định nghĩa biến MỘT lần: (t / tôi)
2. Tái sử dụng biến (co-reference): Nếu concept xuất hiện lại, CHỈ dùng tên biến, không viết lại concept
   VD: :ARG0 (t / tôi) ... :ARG1 t  (KHÔNG phải :ARG1 (t / tôi))
3. Biến dùng chữ cái đầu: (t / tôi), (b / bác_sĩ). Nếu trùng thì thêm số: (t2 / tôi)
4. Đảm bảo số ngoặc mở ( bằng số ngoặc đóng )"""

    user_input = f"Câu: {sentence}"

    prompt = f"""<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{user_input}<|im_end|>
<|im_start|>assistant
"""

    return prompt


def extract_task2_output(generated_text):
    """
    Extract Task 2 output from model generation.

    Model should output:
    Task 1: (...)
    Task 2: (...)

    We extract only Task 2.
    """
    # Remove prompt if it's still in output
    if "<|im_start|>assistant" in generated_text:
        generated_text = generated_text.split("<|im_start|>assistant")[-1]

    # Remove end token
    generated_text = generated_text.replace("<|im_end|>", "").strip()

    # Try to extract Task 2
    task2_match = re.search(r'Task 2:\s*(.+?)(?:\n|$)', generated_text, re.DOTALL)

    if task2_match:
        task2_output = task2_match.group(1).strip()
        # Remove any trailing text after the AMR graph
        # AMR should end with ')'
        # Find the last closing paren
        last_paren = task2_output.rfind(')')
        if last_paren != -1:
            task2_output = task2_output[:last_paren + 1]
        return task2_output
    else:
        # Fallback: try to extract any AMR-like structure
        # Look for pattern starting with '(' and containing '/'
        amr_match = re.search(r'\([a-z0-9]+\s*/\s*[^\)]+\)', generated_text, re.DOTALL)
        if amr_match:
            # Try to get complete AMR from first '(' to balanced ')'
            start_idx = generated_text.find('(')
            if start_idx != -1:
                # Find balanced closing paren
                open_count = 0
                for i, char in enumerate(generated_text[start_idx:]):
                    if char == '(':
                        open_count += 1
                    elif char == ')':
                        open_count -= 1
                        if open_count == 0:
                            return generated_text[start_idx:start_idx + i + 1]

        # Last resort: return everything (will be cleaned later)
        return generated_text


def clean_output(text):
    """Clean and normalize AMR output"""
    # Remove newlines and extra spaces
    text = text.replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text)

    # Remove any remaining special tokens
    text = text.replace('<|im_end|>', '')
    text = text.replace('<|im_start|>', '')

    return text.strip()


def validate_amr(amr_str):
    """Basic validation of AMR structure"""
    if not amr_str:
        return False

    # Check if starts with '(' and ends with ')'
    if not amr_str.startswith('(') or not amr_str.endswith(')'):
        return False

    # Check bracket balance
    if amr_str.count('(') != amr_str.count(')'):
        return False

    # Check if contains variables (has '/' pattern)
    if '/' not in amr_str:
        return False

    return True


def read_input_sentences(input_file):
    """Read input sentences from file"""
    print(f"📂 Reading input: {input_file}")

    with open(input_file, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]

    # Detect format
    sentences = []
    has_prefix = any(line.startswith("SENT:") or line.startswith("Input:") for line in lines[:5])

    if has_prefix:
        for line in lines:
            if line.startswith("SENT:"):
                sentences.append(line.replace("SENT:", "").strip())
            elif line.startswith("Input:"):
                sentences.append(line.replace("Input:", "").strip())
    else:
        sentences = lines

    print(f"✅ Found {len(sentences)} input sentences\n")
    return sentences


def predict(args):
    """Main prediction function"""
    print_banner()

    # Load model
    model, tokenizer = load_model(args.base_model, args.adapter_path)

    # Read input
    sentences = read_input_sentences(args.input_file)

    if not sentences:
        print("❌ No input sentences found!")
        return

    # Create output directory
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Predict
    print(f"🚀 Generating predictions for {len(sentences)} sentences...")
    print("=" * 70 + "\n")

    predictions = []
    errors = 0

    with open(args.output_file, 'w', encoding='utf-8') as f_out:
        for idx, sentence in enumerate(tqdm(sentences, desc="Predicting")):
            try:
                # Create prompt
                prompt = create_unified_prompt(sentence)

                # Tokenize
                inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

                # Generate
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=1024,
                        pad_token_id=tokenizer.eos_token_id,
                        do_sample=False,  # Greedy for consistency
                        repetition_penalty=1.2,
                        no_repeat_ngram_size=3
                    )

                # Decode
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)

                # Extract Task 2
                task2_output = extract_task2_output(generated_text)

                # Clean
                final_output = clean_output(task2_output)

                # Validate
                if not validate_amr(final_output):
                    errors += 1
                    print(f"\n⚠️  Warning: Sample {idx} may have invalid AMR structure")
                    # Still write it, but mark
                    if not final_output:
                        final_output = "(a / amr-empty)"

                # Write to file
                f_out.write(final_output + "\n")
                f_out.flush()

                predictions.append(final_output)

            except Exception as e:
                errors += 1
                print(f"\n❌ Error at sample {idx}: {e}")
                # Write placeholder
                f_out.write("(a / amr-empty)\n")
                f_out.flush()
                predictions.append("(a / amr-empty)")

    # Summary
    print("\n" + "=" * 70)
    print("✅ PREDICTION COMPLETED")
    print("=" * 70)
    print(f"📊 Total samples: {len(sentences)}")
    print(f"✅ Successful: {len(sentences) - errors}")
    print(f"⚠️  Errors/Warnings: {errors}")
    print(f"📁 Output saved to: {args.output_file}")
    print("\nNext step: Evaluate with SMATCH")
    print(f"  python mtup_v2/scripts/evaluate.py \\")
    print(f"    --predictions {args.output_file} \\")
    print(f"    --ground_truth data/public_test_ground_truth.txt")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MTUP v2 Unified Prediction")

    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Base model name or path"
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        required=True,
        help="Path to trained LoRA adapter"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Input file with sentences to parse"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Output file for predictions"
    )

    args = parser.parse_args()

    # Validate
    if not os.path.exists(args.adapter_path):
        print(f"❌ Adapter not found: {args.adapter_path}")
        exit(1)

    if not os.path.exists(args.input_file):
        print(f"❌ Input file not found: {args.input_file}")
        exit(1)

    predict(args)
