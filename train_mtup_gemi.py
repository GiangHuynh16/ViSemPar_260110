import os
import argparse
import torch
import re
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq # <--- Dùng Collator này để fix lỗi padding
)
from peft import LoraConfig, get_peft_model

# ==========================================
# 1. TEMPLATE & PROMPTS
# ==========================================

def create_prompt_stage1(sentence, target_amr=None):
    sys_prompt = """Bạn là một chuyên gia ngôn ngữ học về cấu trúc AMR (Abstract Meaning Representation).
Nhiệm vụ: Chuyển đổi câu tiếng Việt đầu vào sang định dạng đồ thị AMR chuẩn PENMAN.
Yêu cầu đặc biệt:
1. KHÔNG sử dụng biến (variables) định danh (ví dụ: không dùng 't / tôi', chỉ dùng '(tôi)').
2. Đảm bảo cấu trúc ngoặc đơn () cân bằng chính xác.
3. Chỉ giữ lại các Concept và Relation (ví dụ: :ARG0, :ARG1, :mod).

Ví dụ mẫu:
Input: Cậu bé đang đọc sách.
Output: (đọc :ARG0 (cậu_bé) :ARG1 (sách))"""

    user_input = f"Input: {sentence}"
    prompt = f"<|im_start|>system\n{sys_prompt}<|im_end|>\n<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"
    if target_amr:
        prompt += f"{target_amr}<|im_end|>"
    return prompt

def create_prompt_stage2(sentence, amr_no_vars, target_full_amr=None):
    sys_prompt = """Bạn là một chuyên gia gán nhãn dữ liệu AMR (Abstract Meaning Representation).
Nhiệm vụ: Hoàn thiện đồ thị AMR từ cấu trúc thô (chưa có biến) và câu gốc.

Yêu cầu QUAN TRỌNG:
1. Gán biến (variables) định danh cho mỗi concept (vd: '(tôi)' -> '(t / tôi)').
2. TÁI SỬ DỤNG BIẾN (Re-entrancy): Nếu một đối tượng xuất hiện nhiều lần hoặc được thay thế bằng đại từ (anh ấy, nó, cậu ta...), hãy dùng lại biến đã khai báo trước đó thay vì tạo biến mới.
3. Đảm bảo đúng định dạng PENMAN.

Ví dụ mẫu (Complex Re-entrancy):
Input: Nam cố gắng học bài vì cậu ấy muốn đỗ. <sep> (cố_gắng :ARG0 (Nam) :ARG1 (học :ARG1 (bài)) :cause (muốn :ARG0 (cậu_ấy) :ARG1 (đỗ)))
Output: (c / cố_gắng
    :ARG0 (n / Nam)
    :ARG1 (h / học
        :ARG0 n
        :ARG1 (b / bài))
    :cause (m / muốn
        :ARG0 n
        :ARG1 (đ / đỗ
            :ARG0 n)))"""

    user_input = f"Input: {sentence} <sep> {amr_no_vars}"
    prompt = f"<|im_start|>system\n{sys_prompt}<|im_end|>\n<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"
    if target_full_amr:
        prompt += f"{target_full_amr}<|im_end|>"
    return prompt

def format_data(sample, stage):
    text = sample['text'].strip()
    if not text: return None
    try:
        if stage == 1:
            match = re.search(r'SENT:\s*(.*?)\nAMR:\s*(.*)', text, re.DOTALL)
            if match: return create_prompt_stage1(match.group(1).strip(), match.group(2).strip())
            match = re.search(r'Input:\s*(.*?)\nOutput:\s*(.*)', text, re.DOTALL)
            if match: return create_prompt_stage1(match.group(1).strip(), match.group(2).strip())
        else:
            match = re.search(r'SENT:\s*(.*?)\nNO_VAR:\s*(.*?)\nFULL:\s*(.*)', text, re.DOTALL)
            if match: return create_prompt_stage2(match.group(1).strip(), match.group(2).strip(), match.group(3).strip())
            match = re.search(r'Input:\s*(.*?)<sep>(.*?)\nOutput:\s*(.*)', text, re.DOTALL)
            if match: return create_prompt_stage2(match.group(1).strip(), match.group(2).strip(), match.group(3).strip())
        return None
    except Exception: return None

# ==========================================
# 2. TRAINING SETUP (Fix Padding Error)
# ==========================================

def load_and_filter_dataset(file_path, stage):
    print(f"📂 Reading file: {file_path}")
    if not os.path.exists(file_path): raise FileNotFoundError(f"Cannot find: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f: content = f.read()
    
    blocks = [b for b in content.strip().split('\n\n') if b.strip()]
    valid_data = [b for b in blocks if format_data({'text': b}, stage)]
            
    print(f"Dataset: {len(blocks)} raw -> {len(valid_data)} valid samples.")
    if not valid_data: raise ValueError("❌ DATASET IS EMPTY!")
    return Dataset.from_dict({"text": valid_data})

def train(args):
    print(f"🚀 START TRAINING STAGE {args.stage} | GPU 48GB Optimization | Seq2Seq Collator")
    
    raw_dataset = load_and_filter_dataset(args.data_path, args.stage)
    
    print("✨ Loading Model & Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token 
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,       
        device_map="auto",
        attn_implementation="sdpa" 
    )
    
    print("🛠️  Applying LoRA Config...")
    peft_config = LoraConfig(
        lora_alpha=64, lora_dropout=0.05, r=128, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    print("🔄 Tokenizing dataset...")
    def tokenize_function(batch):
        prompts = [format_data({'text': t}, args.stage) for t in batch['text']]
        prompts = [p + tokenizer.eos_token for p in prompts if p]
        
        # Tokenize nhưng KHÔNG padding tại đây (để padding=False)
        # Để DataCollator làm việc đó tối ưu hơn
        outputs = tokenizer(prompts, truncation=True, max_length=2048, padding=False)
        
        outputs["labels"] = outputs["input_ids"].copy()
        return outputs

    tokenized_dataset = raw_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=8,      
        gradient_accumulation_steps=4,      
        learning_rate=2e-4,
        weight_decay=0.01,
        bf16=True,       
        fp16=False,
        logging_steps=10,
        save_strategy="epoch",
        optim="adamw_torch", 
        report_to="none",
        remove_unused_columns=False,
    )

    # --- SỬA CHÍNH: Dùng DataCollatorForSeq2Seq ---
    # Collator này rất giỏi việc padding input_ids và labels cùng lúc
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        pad_to_multiple_of=8, # Tối ưu cho Tensor Cores
        return_tensors="pt",
        padding=True 
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator # Thay thế Collator cũ
    )

    trainer.train()
    
    final_path = os.path.join(args.output_dir, "final_adapter")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"✅ Training Done. Saved to {final_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct") 
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=5) 
    
    args = parser.parse_args()
    train(args)