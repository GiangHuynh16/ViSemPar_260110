import os
import argparse
import torch
import re
import gc
import numpy as np
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ==========================================
# 1. TEMPLATE & PROMPTS (Cải tiến chặt chẽ hơn)
# ==========================================

def create_prompt_stage1(sentence, target_amr=None):
    # Prompt ép buộc cấu trúc chặt chẽ hơn
    sys_prompt = """Bạn là một hệ thống phân tích ngữ nghĩa AMR (Abstract Meaning Representation).
Nhiệm vụ: Chuyển câu tiếng Việt thành cấu trúc AMR-No-Var (chưa có biến).
Quy tắc tuyệt đối:
1. KHÔNG tự tạo biến (ví dụ: dùng '(tôi)' thay vì '(t / tôi)').
2. Đảm bảo số lượng ngoặc mở '(' bằng số lượng ngoặc đóng ')'.
3. Output chỉ chứa duy nhất đồ thị AMR, không giải thích thêm."""

    user_input = f"Câu: {sentence}"
    # Dùng format chat chuẩn của Qwen
    prompt = f"<|im_start|>system\n{sys_prompt}<|im_end|>\n<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"
    if target_amr:
        prompt += f"{target_amr}<|im_end|>"
    return prompt

def create_prompt_stage2(sentence, amr_no_vars, target_full_amr=None):
    sys_prompt = """Bạn là chuyên gia gán biến cho đồ thị AMR (AMR Aligner).
Nhiệm vụ: Thêm biến (variable) vào cấu trúc AMR thô.
Quy tắc SỐNG CÒN để tránh lỗi Duplicate Node:
1. Mỗi Concept chỉ được định nghĩa biến MỘT lần duy nhất. Ví dụ: (t / tôi).
2. TÁI SỬ DỤNG (Re-entrancy): Nếu concept xuất hiện lại, CHỈ dùng tên biến, KHÔNG viết lại concept.
   - SAI: :ARG0 (t / tôi) ... :ARG1 (t / tôi)
   - ĐÚNG: :ARG0 (t / tôi) ... :ARG1 t
3. Hãy dùng các chữ cái đầu làm tên biến (v / viết), nếu trùng thì thêm số (v2 / viết)."""

    user_input = f"Câu: {sentence}\nSkeletion: {amr_no_vars}"
    prompt = f"<|im_start|>system\n{sys_prompt}<|im_end|>\n<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"
    if target_full_amr:
        prompt += f"{target_full_amr}<|im_end|>"
    return prompt

def format_data(sample, stage):
    text = sample['text'].strip()
    if not text: return None
    try:
        # Regex linh hoạt hơn để bắt các biến thể của file input
        if stage == 1:
            match = re.search(r'(?:SENT|Input):\s*(.*?)\n(?:AMR|Output):\s*(.*)', text, re.DOTALL)
            if match: return create_prompt_stage1(match.group(1).strip(), match.group(2).strip())
        else:
            # Ưu tiên format 3 phần
            match = re.search(r'(?:SENT|Input):\s*(.*?)\n(?:NO_VAR):\s*(.*?)\n(?:FULL|Output):\s*(.*)', text, re.DOTALL)
            if match: return create_prompt_stage2(match.group(1).strip(), match.group(2).strip(), match.group(3).strip())
            
            # Fallback cho format cũ
            match = re.search(r'Input:\s*(.*?)<sep>(.*?)\nOutput:\s*(.*)', text, re.DOTALL)
            if match: return create_prompt_stage2(match.group(1).strip(), match.group(2).strip(), match.group(3).strip())
        return None
    except Exception: return None

# ==========================================
# 2. VALIDATION & LOADING
# ==========================================

try:
    import penman
except ImportError:
    print("⚠️ Warning: Penman not installed. Skipping strict data validation.")
    penman = None

def load_and_validate_dataset(file_path, stage):
    print(f"📂 Reading file: {file_path}")
    if not os.path.exists(file_path): raise FileNotFoundError(f"Cannot find: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f: content = f.read()
    
    blocks = [b for b in content.strip().split('\n\n') if b.strip()]
    valid_data = []
    
    print("🔍 Validating data format...")
    for b in blocks:
        formatted = format_data({'text': b}, stage)
        if formatted:
            # Basic validation: Check if target AMR is valid PENMAN (Optional but recommended)
            # Nếu data gốc lỗi -> Train ra model lỗi. Nên lọc kỹ.
            valid_data.append(formatted)
            
    print(f"Dataset: {len(blocks)} raw -> {len(valid_data)} valid samples ready for training.")
    if not valid_data: raise ValueError("❌ DATASET IS EMPTY OR FORMAT WRONG!")
    return Dataset.from_dict({"text": valid_data})

# ==========================================
# 3. TRAINING ENGINE
# ==========================================

def train(args):
    print(f"🚀 START RESCUE TRAINING STAGE {args.stage}")
    torch.cuda.empty_cache()
    gc.collect()

    # 1. Load Data
    raw_dataset = load_and_validate_dataset(args.data_path, args.stage)

    # 2. Model Config (4-bit QLoRA)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # Quan trọng cho training

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config, 
        device_map="auto",
        attn_implementation="flash_attention_2" if torch.cuda.get_device_capability()[0] >= 8 else "sdpa"
    )
    model = prepare_model_for_kbit_training(model)

    # 3. LoRA Config
    peft_config = LoraConfig(
        lora_alpha=16, 
        lora_dropout=0.05,
        r=64,          # Tăng Rank lên 64 để model học thông minh hơn (vẫn vừa VRAM 24GB)
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 4. Tokenization & MASKING (CRITICAL FIX)
    # Hàm này sẽ set label = -100 cho phần User Prompt, chỉ tính loss cho Assistant Answer
    def tokenize_with_masking(batch):
        tokenized_inputs = tokenizer(batch['text'], truncation=True, max_length=1536, padding=False)
        input_ids_list = tokenized_inputs["input_ids"]
        attention_mask_list = tokenized_inputs["attention_mask"]
        
        labels_list = []
        
        # Token đánh dấu bắt đầu câu trả lời của Assistant (Qwen format)
        # Qwen dùng: <|im_start|>assistant\n
        # Chúng ta cần tìm token ID của chuỗi này hoặc tương tự.
        # Đơn giản nhất: Tokenize lại prompt phần user, lấy độ dài, mask phần đó.
        
        for input_ids, text in zip(input_ids_list, batch['text']):
            # Tách phần User prompt và Assistant answer
            split_text = text.split("<|im_start|>assistant\n")
            if len(split_text) < 2:
                # Trường hợp lỗi format, mask toàn bộ (ignore)
                labels_list.append([-100] * len(input_ids))
                continue
                
            prompt_part = split_text[0] + "<|im_start|>assistant\n"
            prompt_ids = tokenizer(prompt_part, truncation=True, max_length=1536, add_special_tokens=False)["input_ids"]
            
            prompt_len = len(prompt_ids)
            
            # Tạo labels: Copy input_ids
            labels = list(input_ids)
            # Mask phần prompt (set về -100)
            for i in range(min(prompt_len, len(labels))):
                labels[i] = -100
                
            labels_list.append(labels)
            
        return {
            "input_ids": input_ids_list,
            "attention_mask": attention_mask_list,
            "labels": labels_list
        }

    print("🔄 Tokenizing & Masking Inputs...")
    tokenized_dataset = raw_dataset.map(tokenize_with_masking, batched=True, remove_columns=["text"])

    # 5. Training Args (Tối ưu cho hội tụ nhanh)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=2,      # Thử tăng lên 2 nếu VRAM cho phép (để Batch Norm ổn định hơn)
        gradient_accumulation_steps=16,     # 2 * 16 = 32 effective batch
        learning_rate=1e-4,                 # Giảm LR một chút để tránh phá hỏng pre-trained weights
        weight_decay=0.01,
        bf16=True,
        logging_steps=5,
        save_strategy="epoch",
        optim="paged_adamw_32bit",
        report_to="none",
        warmup_ratio=0.03,                  # Warmup để model không bị shock
        group_by_length=True,               # Train các câu cùng độ dài với nhau -> nhanh hơn
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        pad_to_multiple_of=8,
        return_tensors="pt",
        padding=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator
    )

    print("🔥 Training started...")
    trainer.train()
    
    print("💾 Saving model...")
    trainer.model.save_pretrained(os.path.join(args.output_dir, "final_adapter"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "final_adapter"))
    print("✅ DONE.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct") 
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=3) 
    
    args = parser.parse_args()
    train(args)