#!/usr/bin/env python3
"""
MTUP v2 - Multi-Task Unified Prompting Training (No Quantization)

Uses bfloat16 instead of 4-bit quantization - requires more VRAM but no bitsandbytes

Usage:
    python mtup_v2/scripts/train_mtup_no_quant.py \
        --data_path data/train_mtup_unified.txt \
        --model_name Qwen/Qwen2.5-7B-Instruct \
        --output_dir outputs/mtup_v2 \
        --epochs 5
"""

import os
import argparse
import torch
import gc
import re
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model


def print_banner():
    """Print training banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║     MTUP v2 - Multi-Task Unified Prompting Training         ║
    ║                                                              ║
    ║  🎯 Single Model, Unified Prompt, Two Tasks                 ║
    ║  📊 Task 1: AMR Skeleton (no variables)                     ║
    ║  📊 Task 2: Full AMR (with variables)                       ║
    ║                                                              ║
    ║  ⚡ bfloat16 (No Quantization) - Requires >30GB VRAM        ║
    ║                                                              ║
    ║              Vietnamese AMR Parsing - VLSP 2025             ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)
    print(f"⚡ Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print("=" * 70)


def load_and_validate_dataset(file_path):
    """
    Load unified MTUP dataset from file.

    Expected format per sample (separated by double newlines):
    <|im_start|>system
    ...
    <|im_end|>
    <|im_start|>user
    Câu: {sentence}
    <|im_end|>
    <|im_start|>assistant
    Task 1: {no_var_amr}
    Task 2: {with_var_amr}
    <|im_end|>
    """
    print(f"📂 Reading file: {file_path}")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Cannot find: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split by pattern: <|im_end|>\n\n<|im_start|>
    # This splits samples but keeps the structure intact
    import re

    # Split on double newline between samples (after <|im_end|> before next <|im_start|>system)
    blocks = re.split(r'<\|im_end\|>\n\n(?=<\|im_start\|>system)', content.strip())

    # Add back the closing tag to all but the last block
    blocks = [b + '<|im_end|>' if not b.endswith('<|im_end|>') else b for b in blocks]

    # Filter empty blocks
    blocks = [b.strip() for b in blocks if b.strip()]

    valid_data = []
    errors = 0

    print("🔍 Validating data format...")

    for idx, block in enumerate(blocks):
        # Basic validation: check if it contains required markers
        if '<|im_start|>system' not in block:
            errors += 1
            if errors <= 3:  # Show first 3 errors
                print(f"  ⚠️  Sample {idx}: Missing <|im_start|>system")
            continue

        if '<|im_start|>user' not in block:
            errors += 1
            if errors <= 3:
                print(f"  ⚠️  Sample {idx}: Missing <|im_start|>user")
            continue

        if '<|im_start|>assistant' not in block:
            errors += 1
            if errors <= 3:
                print(f"  ⚠️  Sample {idx}: Missing <|im_start|>assistant")
            continue

        # Check for Task 1 and Task 2
        if 'Task 1:' not in block or 'Task 2:' not in block:
            errors += 1
            if errors <= 3:
                print(f"  ⚠️  Sample {idx}: Missing Task 1 or Task 2")
            continue

        # Validate bracket balance in assistant output
        try:
            assistant_part = block.split('<|im_start|>assistant')[-1].split('<|im_end|>')[0]

            # Extract Task 1 and Task 2
            task1_match = re.search(r'Task 1:\s*(.+?)(?=Task 2:|$)', assistant_part, re.DOTALL)
            task2_match = re.search(r'Task 2:\s*(.+?)$', assistant_part, re.DOTALL)

            if not task1_match or not task2_match:
                errors += 1
                if errors <= 3:
                    print(f"  ⚠️  Sample {idx}: Cannot extract Task outputs")
                continue

            task1_amr = task1_match.group(1).strip()
            task2_amr = task2_match.group(1).strip()

            # Check bracket balance
            if task1_amr.count('(') != task1_amr.count(')'):
                errors += 1
                if errors <= 3:
                    print(f"  ⚠️  Sample {idx}: Task 1 unbalanced brackets")
                continue

            if task2_amr.count('(') != task2_amr.count(')'):
                errors += 1
                if errors <= 3:
                    print(f"  ⚠️  Sample {idx}: Task 2 unbalanced brackets")
                continue

            valid_data.append(block)

        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"  ⚠️  Sample {idx}: Exception - {e}")
            continue

    print(f"✅ Dataset: {len(blocks)} raw → {len(valid_data)} valid samples")

    if errors > 0:
        print(f"⚠️  Skipped {errors} invalid samples")

    if not valid_data:
        print("\n❌ NO VALID SAMPLES FOUND!")
        print("\nDEBUG: First block content:")
        if blocks:
            print(blocks[0][:500])
        raise ValueError("❌ DATASET IS EMPTY OR FORMAT WRONG!")

    return Dataset.from_dict({"text": valid_data})


def tokenize_with_masking(batch, tokenizer):
    """
    Tokenize and mask prompts so loss is only computed on assistant outputs.
    """
    tokenized_inputs = tokenizer(
        batch['text'],
        truncation=True,
        max_length=2048,
        padding=False
    )

    input_ids_list = tokenized_inputs["input_ids"]
    attention_mask_list = tokenized_inputs["attention_mask"]
    labels_list = []

    for input_ids, text in zip(input_ids_list, batch['text']):
        # Find where assistant response starts
        split_text = text.split("<|im_start|>assistant\n")

        if len(split_text) < 2:
            # Invalid format, mask everything
            labels_list.append([-100] * len(input_ids))
            continue

        # Tokenize just the prompt part
        prompt_part = split_text[0] + "<|im_start|>assistant\n"
        prompt_ids = tokenizer(
            prompt_part,
            truncation=True,
            max_length=2048,
            add_special_tokens=False
        )["input_ids"]

        prompt_len = len(prompt_ids)

        # Create labels: start with copy of input_ids
        labels = list(input_ids)

        # Mask prompt part (set to -100)
        for i in range(min(prompt_len, len(labels))):
            labels[i] = -100

        labels_list.append(labels)

    return {
        "input_ids": input_ids_list,
        "attention_mask": attention_mask_list,
        "labels": labels_list
    }


def train(args):
    """Main training function"""
    print_banner()

    print(f"\n🚀 STARTING MTUP v2 UNIFIED TRAINING")
    print("=" * 70)

    torch.cuda.empty_cache()
    gc.collect()

    # 1. Load Dataset
    raw_dataset = load_and_validate_dataset(args.data_path)
    print(f"✅ Loaded {len(raw_dataset)} training samples\n")

    # 2. Model Configuration (bfloat16 - No Quantization)
    print("🔧 Configuring model with bfloat16...")

    # 3. Load Tokenizer
    print(f"📥 Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 4. Load Model (No Quantization)
    print(f"📥 Loading model: {args.model_name}")

    # Try to use Flash Attention 2 if available, otherwise fall back to SDPA
    try:
        import flash_attn
        attn_impl = "flash_attention_2" if torch.cuda.get_device_capability()[0] >= 8 else "sdpa"
        print(f"   Using attention implementation: {attn_impl}")
    except ImportError:
        attn_impl = "sdpa"
        print(f"   Flash Attention not installed, using SDPA (still fast!)")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation=attn_impl
    )

    # 5. Enable gradient checkpointing BEFORE applying LoRA
    print("🔧 Enabling gradient checkpointing...")
    model.gradient_checkpointing_enable()

    # 6. LoRA Configuration
    print("🔧 Configuring LoRA adapters...")
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    model = get_peft_model(model, peft_config)

    # Make sure LoRA parameters require grad
    for name, param in model.named_parameters():
        if 'lora' in name.lower():
            param.requires_grad = True

    print("\n📊 Trainable Parameters:")
    model.print_trainable_parameters()
    print()

    # 7. Tokenization & Masking
    print("🔄 Tokenizing and masking inputs...")
    tokenized_dataset = raw_dataset.map(
        lambda batch: tokenize_with_masking(batch, tokenizer),
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing"
    )

    # 8. Training Arguments
    print("⚙️  Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=16,
        learning_rate=1e-4,
        weight_decay=0.01,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        optim="adamw_torch",  # Use standard PyTorch AdamW (no bitsandbytes needed)
        report_to="none",
        warmup_ratio=0.03,
        group_by_length=True,
        gradient_checkpointing=False,  # Already enabled on model directly
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    # 9. Data Collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        pad_to_multiple_of=8,
        return_tensors="pt",
        padding=True
    )

    # 10. Trainer
    print("🏗️  Creating Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator
    )

    # 10. Train!
    print("\n" + "=" * 70)
    print("🔥 TRAINING STARTED")
    print("=" * 70 + "\n")

    trainer.train()

    # 11. Save
    print("\n💾 Saving final model...")
    final_adapter_path = os.path.join(args.output_dir, "final_adapter")
    trainer.model.save_pretrained(final_adapter_path)
    tokenizer.save_pretrained(final_adapter_path)

    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(f"📁 Model saved to: {final_adapter_path}")
    print("\nNext steps:")
    print("  1. Run prediction: python mtup_v2/scripts/predict_mtup_unified.py")
    print("  2. Evaluate results: python mtup_v2/scripts/evaluate.py")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MTUP v2 Unified Training (No Quantization)")

    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to train_mtup_unified.txt"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Base model name or path"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for checkpoints and final model"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs"
    )

    args = parser.parse_args()

    # Validate paths
    if not os.path.exists(args.data_path):
        print(f"❌ Error: Data file not found: {args.data_path}")
        exit(1)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Start training
    train(args)
