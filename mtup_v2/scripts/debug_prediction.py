#!/usr/bin/env python3
"""
Debug prediction to see raw model output.

Usage:
    python mtup_v2/scripts/debug_prediction.py \
        --adapter_path outputs/mtup_260110/mtup_v2/final_adapter \
        --test_sentence "bi kịch là ở chỗ đó !"
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def main(args):
    print("🔍 DEBUG MODE - Checking raw model output\n")

    # Load model
    print(f"📥 Loading model...")

    try:
        import flash_attn
        attn_impl = "flash_attention_2" if torch.cuda.get_device_capability()[0] >= 8 else "sdpa"
    except ImportError:
        attn_impl = "sdpa"

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation=attn_impl
    )

    model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    print(f"✅ Model loaded\n")

    # Create prompt
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

    user_input = f"Câu: {args.test_sentence}"

    prompt = f"""<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{user_input}<|im_end|>
<|im_start|>assistant
"""

    print("=" * 70)
    print("PROMPT:")
    print("=" * 70)
    print(prompt)
    print("=" * 70 + "\n")

    # Generate
    print("🚀 Generating...\n")
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3
        )

    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)

    print("=" * 70)
    print("FULL GENERATED TEXT:")
    print("=" * 70)
    print(generated_text)
    print("=" * 70 + "\n")

    # Extract assistant part only
    if "<|im_start|>assistant" in generated_text:
        assistant_output = generated_text.split("<|im_start|>assistant")[-1]
        assistant_output = assistant_output.replace("<|im_end|>", "").strip()

        print("=" * 70)
        print("ASSISTANT OUTPUT ONLY:")
        print("=" * 70)
        print(assistant_output)
        print("=" * 70 + "\n")

        # Check for Task 1 and Task 2
        if "Task 1:" in assistant_output:
            print("✅ Found Task 1")
        else:
            print("❌ Missing Task 1")

        if "Task 2:" in assistant_output:
            print("✅ Found Task 2")
        else:
            print("❌ Missing Task 2")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug prediction output")

    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Base model name"
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        required=True,
        help="Path to adapter"
    )
    parser.add_argument(
        "--test_sentence",
        type=str,
        required=True,
        help="Test sentence"
    )

    args = parser.parse_args()
    main(args)
