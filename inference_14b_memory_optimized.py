#!/usr/bin/env python3
"""
Memory-Optimized Inference for 14B Model
Handles OOM by using aggressive memory optimization
"""

import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import json
from tqdm import tqdm

# Memory optimization settings
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def clear_gpu_memory():
    """Aggressively clear GPU memory"""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

def load_sentences(file_path):
    """Load sentences from file"""
    sentences = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                sentences.append(line)
    return sentences

def generate_amr_batch(model, tokenizer, sentences, max_length=512):
    """
    Generate AMR with memory optimization
    Process one sentence at a time to avoid OOM
    """
    results = []

    for sentence in tqdm(sentences, desc="Generating"):
        # Build prompt (Vietnamese format matching training)
        prompt = f"""### NHIỆM VỤ: Chuyển đổi câu tiếng Việt sang AMR (2 bước)

### Câu cần phân tích:
{sentence}

### Kết quả phân tích:

## Bước 1 - Tạo cấu trúc AMR (chưa có biến):
"""

        # Tokenize with attention to memory
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length
        ).to(model.device)

        try:
            # Generate with memory-efficient settings
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,  # Limit new tokens to save memory
                    do_sample=False,     # Greedy decoding (no sampling overhead)
                    num_beams=1,         # No beam search
                    pad_token_id=tokenizer.eos_token_id,
                    use_cache=True,      # Use KV cache
                )

            # Decode
            result = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract AMR
            if "AMR hoàn chỉnh:" in result:
                amr = result.split("AMR hoàn chỉnh:")[-1].strip()
            elif "## Bước 2" in result:
                amr = result.split("## Bước 2")[1].strip()
            else:
                amr = result.strip()

            # Clean up - only keep AMR structure
            if '(' in amr:
                first_paren = amr.index('(')
                amr = amr[first_paren:].strip()

            results.append({
                'sentence': sentence,
                'amr': amr
            })

        except torch.cuda.OutOfMemoryError:
            print(f"\n⚠️  OOM on sentence: {sentence[:50]}...")
            print("Clearing cache and retrying...")

            # Clear memory
            del inputs, outputs
            clear_gpu_memory()

            # Retry with shorter max length
            try:
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=256  # Much shorter
                ).to(model.device)

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=128,  # Very limited
                        do_sample=False,
                        num_beams=1,
                        pad_token_id=tokenizer.eos_token_id,
                    )

                result = tokenizer.decode(outputs[0], skip_special_tokens=True)
                amr = "(error / retry-failed)"  # Placeholder

                results.append({
                    'sentence': sentence,
                    'amr': amr
                })

            except:
                print(f"❌ Failed even with reduced settings")
                results.append({
                    'sentence': sentence,
                    'amr': "(error / oom)"
                })

        finally:
            # Clean up after each sentence
            del inputs
            if 'outputs' in locals():
                del outputs
            clear_gpu_memory()

    return results

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--input', type=str, required=True,
                        help='Input file with sentences')
    parser.add_argument('--output', type=str, required=True,
                        help='Output JSON file')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Limit number of samples')

    args = parser.parse_args()

    print("="*80)
    print("MEMORY-OPTIMIZED INFERENCE - 14B MODEL")
    print("="*80)
    print()

    # Clear GPU before starting
    print("Clearing GPU memory...")
    clear_gpu_memory()

    # Load model with memory optimizations
    print(f"Loading model from {args.model}...")
    print("Using aggressive memory optimization...")

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",           # Auto device placement
        torch_dtype=torch.float16,   # FP16 to save memory
        low_cpu_mem_usage=True,      # Reduce CPU RAM usage
        offload_folder="offload",    # Offload to disk if needed
        offload_state_dict=True,     # Offload state dict
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    print("✅ Model loaded!")

    # Check GPU memory
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")

    print()

    # Load sentences
    print(f"Loading sentences from {args.input}...")
    sentences = load_sentences(args.input)

    if args.max_samples:
        sentences = sentences[:args.max_samples]

    print(f"Total sentences: {len(sentences)}")
    print()

    # Generate
    print("Generating predictions...")
    print("Note: Processing one at a time to avoid OOM")
    print()

    results = generate_amr_batch(model, tokenizer, sentences)

    # Save results
    print()
    print(f"Saving results to {args.output}...")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("✅ Done!")
    print()
    print(f"Generated {len(results)} predictions")
    print(f"Saved to: {args.output}")
    print()
    print("="*80)

if __name__ == "__main__":
    main()
