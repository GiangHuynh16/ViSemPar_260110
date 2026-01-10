#!/usr/bin/env python3
"""
MTUP v2 - Higher Capacity Training Script

Tăng capacity của model để học tốt hơn:
- LoRA rank: 64 (tăng từ 32)
- LoRA alpha: 32 (tăng từ 16)
- Learning rate: 3e-5 (giảm để stable hơn)
- Epochs: 20 (tăng từ 15)

Usage:
    python mtup_v2/scripts/train_mtup_higher_capacity.py \
        --data_path data/train_mtup_unified.txt \
        --model_name Qwen/Qwen2.5-7B-Instruct \
        --output_dir outputs/mtup_v2_rank64 \
        --epochs 20
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


def train_higher_capacity(args):
    """Higher capacity training function"""
    print_banner()

    print(f"\n🚀 MTUP v2 HIGHER CAPACITY TRAINING")
    print("=" * 70)
    print("🎯 Improvements:")
    print("  • HIGHER LoRA rank (64 instead of 32)")
    print("  • HIGHER LoRA alpha (32 instead of 16)")
    print("  • LOWER learning rate (3e-5 instead of 5e-5)")
    print("  • MORE epochs (20 instead of 15)")
    print("  • More model capacity to learn complex patterns")
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

    # 5. LoRA Configuration (HIGHER CAPACITY)
    print("🔧 Configuring LoRA adapters (rank=64, alpha=32)...")
    peft_config = LoraConfig(
        lora_alpha=32,          # INCREASED from 16 to 32
        lora_dropout=0.1,
        r=64,                   # INCREASED from 32 to 64
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

    # 7. Training Arguments (HIGHER CAPACITY)
    print("⚙️  Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=32,
        learning_rate=3e-5,                  # LOWER from 5e-5 to 3e-5 for stability
        weight_decay=0.01,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=3,
        optim="adamw_torch",
        report_to="none",
        warmup_ratio=0.1,
        group_by_length=True,
        gradient_checkpointing=False,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=0.3,
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
    print("🔥 TRAINING STARTED (HIGHER CAPACITY)")
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
    parser = argparse.ArgumentParser(description="MTUP v2 Higher Capacity Training")

    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=20)

    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        print(f"❌ Error: Data file not found: {args.data_path}")
        exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    train_higher_capacity(args)
