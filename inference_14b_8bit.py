#!/usr/bin/env python3
"""
8-bit Quantized Inference for 14B Model
Uses bitsandbytes for int8 quantization to reduce memory by ~50%
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from pathlib import Path
import json
from tqdm import tqdm

def load_sentences(file_path):
    """Load sentences from file"""
    sentences = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                sentences.append(line)
    return sentences

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--max-samples', type=int, default=None)

    args = parser.parse_args()

    print("="*80)
    print("8-BIT QUANTIZED INFERENCE - 14B MODEL")
    print("="*80)
    print()

    # Configure 8-bit quantization
    print("Configuring 8-bit quantization...")
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
    )

    print(f"Loading model from {args.model}...")
    print("This will use ~50% less memory than FP16!")
    print()

    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            quantization_config=quantization_config,
            device_map="auto",
            low_cpu_mem_usage=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(args.model)

        print("✅ Model loaded in 8-bit!")

        # Check memory usage
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"GPU Memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
            print()

    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print()
        print("Make sure bitsandbytes is installed:")
        print("  pip install bitsandbytes")
        return

    # Load sentences
    print(f"Loading sentences from {args.input}...")
    sentences = load_sentences(args.input)

    if args.max_samples:
        sentences = sentences[:args.max_samples]

    print(f"Total sentences: {len(sentences)}")
    print()

    # Generate predictions
    print("Generating predictions...")
    results = []

    for sentence in tqdm(sentences, desc="Processing"):
        prompt = f"""### NHIỆM VỤ: Chuyển đổi câu tiếng Việt sang AMR (2 bước)

### Câu cần phân tích:
{sentence}

### Kết quả phân tích:

## Bước 1 - Tạo cấu trúc AMR (chưa có biến):
"""

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)

        try:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

            result = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract AMR
            if "AMR hoàn chỉnh:" in result:
                amr = result.split("AMR hoàn chỉnh:")[-1].strip()
            elif "## Bước 2" in result:
                amr = result.split("## Bước 2")[1].strip()
            else:
                amr = result.strip()

            if '(' in amr:
                amr = amr[amr.index('('):].strip()

            results.append({'sentence': sentence, 'amr': amr})

        except Exception as e:
            print(f"\n❌ Error on sentence: {sentence[:50]}... - {e}")
            results.append({'sentence': sentence, 'amr': "(error / failed)"})

        finally:
            del inputs, outputs
            torch.cuda.empty_cache()

    # Save results
    print()
    print(f"Saving results to {args.output}...")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("✅ Done!")
    print(f"Generated {len(results)} predictions")
    print()
    print("="*80)

if __name__ == "__main__":
    main()
