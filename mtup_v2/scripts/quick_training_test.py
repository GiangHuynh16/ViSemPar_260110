#!/usr/bin/env python3
"""
Quick Training Test - Test với 10 samples, 1 epoch (~5-10 phút)
Verify output format và SMATCH score calculation trước khi train đầy đủ
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model

def create_small_test_dataset(data_path, num_samples=10):
    """Load only first N samples for quick test"""
    print(f"📦 Loading {num_samples} samples for quick test...")

    with open(data_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split by conversation blocks
    conversations = []
    current = []

    for line in content.split('\n'):
        if line.startswith('<|im_start|>system') and current:
            conversations.append('\n'.join(current))
            current = [line]
        else:
            current.append(line)

    if current:
        conversations.append('\n'.join(current))

    # Take first N samples
    small_dataset = conversations[:num_samples]

    dataset = Dataset.from_dict({"text": small_dataset})
    print(f"✅ Created test dataset with {len(dataset)} samples\n")
    return dataset

def tokenize_with_masking(batch, tokenizer):
    """Tokenize with input masking (same as original)"""
    texts = batch["text"]

    encodings = tokenizer(
        texts,
        truncation=True,
        max_length=2048,
        padding=False,
        return_tensors=None
    )

    input_ids_list = []
    labels_list = []

    for input_ids in encodings["input_ids"]:
        labels = input_ids.copy()

        # Mask everything before last assistant response
        assistant_token = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)

        last_assistant_idx = -1
        for i in range(len(input_ids) - len(assistant_token)):
            if input_ids[i:i+len(assistant_token)] == assistant_token:
                last_assistant_idx = i + len(assistant_token)

        if last_assistant_idx > 0:
            labels[:last_assistant_idx] = [-100] * last_assistant_idx

        input_ids_list.append(input_ids)
        labels_list.append(labels)

    return {
        "input_ids": input_ids_list,
        "labels": labels_list,
        "attention_mask": encodings["attention_mask"]
    }

def quick_test_training(args):
    """Quick training test with minimal setup"""

    print("=" * 70)
    print("⚡ QUICK TRAINING TEST (10 samples, 1 epoch)")
    print("=" * 70)
    print("🎯 Purpose: Verify everything works before full training")
    print("⏱️  Expected time: 5-10 minutes")
    print("=" * 70 + "\n")

    # 1. Small dataset
    raw_dataset = create_small_test_dataset(args.data_path, num_samples=10)

    # 2. Tokenizer
    print(f"📥 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 3. Model (smaller LoRA for quick test)
    print(f"📥 Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa"
    )

    # Small LoRA config for testing
    print("🔧 Configuring LoRA (rank=8 for quick test)...")
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=8,  # Small rank for quick test
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 4. Tokenization
    print("\n🔄 Tokenizing...")
    tokenized_dataset = raw_dataset.map(
        lambda batch: tokenize_with_masking(batch, tokenizer),
        batched=True,
        remove_columns=["text"]
    )

    # 5. Training args (1 epoch only)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=1,  # Only 1 epoch!
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,  # Smaller for quick test
        learning_rate=3e-5,
        bf16=True,
        logging_steps=1,  # Log every step
        save_strategy="no",  # Don't save during test
        optim="adamw_torch",
        report_to="none",
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        pad_to_multiple_of=8,
        return_tensors="pt",
        padding=True
    )

    # 6. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator
    )

    # 7. Train
    print("\n" + "=" * 70)
    print("🔥 QUICK TRAINING STARTED")
    print("=" * 70 + "\n")

    trainer.train()

    # 8. Test inference
    print("\n" + "=" * 70)
    print("🧪 TESTING INFERENCE")
    print("=" * 70 + "\n")

    test_input = """<|im_start|>system
Bạn là chuyên gia phân tích AMR (Abstract Meaning Representation) cho tiếng Việt.
Nhiệm vụ: Với mỗi câu tiếng Việt, sinh ra 2 output:

Task 1 - AMR Skeleton: Cấu trúc AMR chỉ có concept và relation, KHÔNG có biến định danh.
Task 2 - Full AMR: AMR hoàn chỉnh với biến theo chuẩn PENMAN.

Quy tắc QUAN TRỌNG cho Task 2:
1. Mỗi concept định nghĩa biến MỘT lần: (t / tôi)
2. Tái sử dụng biến (co-reference): Nếu concept xuất hiện lại, CHỈ dùng tên biến, không viết lại concept
   VD: :ARG0 (t / tôi) ... :ARG1 t  (KHÔNG phải :ARG1 (t / tôi))
3. Biến dùng chữ cái đầu: (t / tôi), (b / bác_sĩ). Nếu trùng thì thêm số: (t2 / tôi)
4. Đảm bảo số ngoặc mở ( bằng số ngoặc đóng )<|im_end|>
<|im_start|>user
Câu: bi kịch là ở chỗ đó !<|im_end|>
<|im_start|>assistant
"""

    inputs = tokenizer(test_input, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)

    # Extract only the assistant response
    if "<|im_start|>assistant" in generated_text:
        response = generated_text.split("<|im_start|>assistant")[-1]
        response = response.replace("<|im_end|>", "").strip()

        print("📝 Generated Output:")
        print("-" * 70)
        print(response)
        print("-" * 70)

        # Check format
        print("\n✅ Format Checks:")
        has_task1 = "Task 1:" in response
        has_task2 = "Task 2:" in response
        has_parentheses = "(" in response and ")" in response

        print(f"  • Has 'Task 1:': {has_task1}")
        print(f"  • Has 'Task 2:': {has_task2}")
        print(f"  • Has AMR parentheses: {has_parentheses}")

        if has_task1 and has_task2 and has_parentheses:
            print("\n✅ OUTPUT FORMAT LOOKS GOOD!")
            print("✅ Ready for full training")
        else:
            print("\n⚠️  OUTPUT FORMAT MAY NEED ADJUSTMENT")
            print("⚠️  Check training data format")

    # Save test model
    print(f"\n💾 Saving test model to: {args.output_dir}/test_adapter")
    trainer.model.save_pretrained(f"{args.output_dir}/test_adapter")
    tokenizer.save_pretrained(f"{args.output_dir}/test_adapter")

    print("\n" + "=" * 70)
    print("✅ QUICK TEST COMPLETED")
    print("=" * 70)
    print("\nNext steps:")
    print("1. If output looks good → Run full training")
    print("2. If output has issues → Fix data/format first")
    print(f"\nFull training command:")
    print(f"  python mtup_v2/scripts/train_mtup_higher_capacity.py \\")
    print(f"    --data_path {args.data_path} \\")
    print(f"    --model_name {args.model_name} \\")
    print(f"    --output_dir outputs/mtup_v2_rank64 \\")
    print(f"    --epochs 20")
    print("=" * 70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quick Training Test")

    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--output_dir", type=str, default="outputs/quick_test")

    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        print(f"❌ Error: Data file not found: {args.data_path}")
        exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    quick_test_training(args)
