#!/usr/bin/env python3
# 14:14 250110 - fixed từ file no_quant
# Learning rate: 5e-5 (giảm từ 1e-4)
# LoRA rank: 32 (giảm từ 64)
# LoRA dropout: 0.1 (tăng từ 0.05)
# Batch size: 1 (giảm từ 2)
# Gradient accumulation: 32 (tăng từ 16) 
# Warmup: 0.1 (tăng từ 0.03)
# Gradient clipping: 0.3 
# Epochs: 15 (default) 
"""
MTUP v2 - Optimized Training Script

Better hyperparameters for small dataset (1840 samples).

Usage:
    python mtup_v2/scripts/train_mtup_optimized.py \
        --data_path data/train_mtup_unified.txt \
        --model_name Qwen/Qwen2.5-7B-Instruct \
        --output_dir outputs/mtup_v2_optimized \
        --epochs 15
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Import from original script
from scripts.train_mtup_no_quant import (
    load_and_validate_dataset,
    tokenize_with_masking,
    print_banner
)

import argparse
import torch
import gc
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model


def train_optimized(args):
    """Optimized training function"""
    print_banner()

    print(f"\n🚀 MTUP v2 OPTIMIZED TRAINING")
    print("=" * 70)
    print("🎯 Optimizations:")
    print("  • Lower learning rate (5e-5 instead of 1e-4)")
    print("  • Smaller LoRA rank (32 instead of 64)")
    print("  • More epochs (15 default)")
    print("  • Smaller batch size for stability")
    print("=" * 70 + "\n")

    torch.cuda.empty_cache()
    gc.collect()

    # 1. Load Dataset
    raw_dataset = load_and_validate_dataset(args.data_path)
    print(f"✅ Loaded {len(raw_dataset)} training samples\n")

    # 2. Load Tokenizer
    print(f"📥 Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 3. Load Model
    print(f"📥 Loading model: {args.model_name}")

    try:
        import flash_attn
        attn_impl = "flash_attention_2" if torch.cuda.get_device_capability()[0] >= 8 else "sdpa"
        print(f"   Using attention: {attn_impl}")
    except ImportError:
        attn_impl = "sdpa"
        print(f"   Using SDPA attention")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation=attn_impl
    )

    # 4. Enable gradient checkpointing
    print("🔧 Enabling gradient checkpointing...")
    model.gradient_checkpointing_enable()

    # 5. LoRA Configuration (OPTIMIZED)
    print("🔧 Configuring LoRA adapters (rank=32, alpha=16)...")
    peft_config = LoraConfig(
        lora_alpha=16,          # Keep alpha same
        lora_dropout=0.1,       # Increased dropout for regularization
        r=32,                   # REDUCED from 64 to 32 for stability
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    model = get_peft_model(model, peft_config)

    # Ensure gradients
    for name, param in model.named_parameters():
        if 'lora' in name.lower():
            param.requires_grad = True

    print("\n📊 Trainable Parameters:")
    model.print_trainable_parameters()
    print()

    # 6. Tokenization
    print("🔄 Tokenizing and masking inputs...")
    tokenized_dataset = raw_dataset.map(
        lambda batch: tokenize_with_masking(batch, tokenizer),
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing"
    )

    # 7. Training Arguments (OPTIMIZED)
    print("⚙️  Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,      # REDUCED from 2 to 1
        gradient_accumulation_steps=32,      # INCREASED from 16 to 32
        learning_rate=5e-5,                  # REDUCED from 1e-4 to 5e-5
        weight_decay=0.01,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=3,                  # Keep more checkpoints
        optim="adamw_torch",
        report_to="none",
        warmup_ratio=0.1,                    # INCREASED warmup
        group_by_length=True,
        gradient_checkpointing=False,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=0.3,                   # ADD gradient clipping
    )

    # 8. Data Collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        pad_to_multiple_of=8,
        return_tensors="pt",
        padding=True
    )

    # 9. Trainer
    print("🏗️  Creating Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator
    )

    # 10. Train
    print("\n" + "=" * 70)
    print("🔥 TRAINING STARTED (OPTIMIZED)")
    print("=" * 70 + "\n")

    trainer.train()

    # 11. Save
    print("\n💾 Saving final model...")
    final_adapter_path = os.path.join(args.output_dir, "final_adapter")
    trainer.model.save_pretrained(final_adapter_path)
    tokenizer.save_pretrained(final_adapter_path)

    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETED")
    print("=" * 70)
    print(f"📁 Model saved to: {final_adapter_path}")
    print("\nNext: Test with debug script:")
    print(f"  python mtup_v2/scripts/debug_prediction.py \\")
    print(f"    --adapter_path {final_adapter_path} \\")
    print(f'    --test_sentence "bi kịch là ở chỗ đó !"')
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MTUP v2 Optimized Training")

    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=15)

    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        print(f"❌ Error: Data file not found: {args.data_path}")
        exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    train_optimized(args)
