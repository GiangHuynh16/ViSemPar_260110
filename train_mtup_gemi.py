import os
import argparse
import torch
import inspect
import re
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer

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
    # Hàm này nhận vào 1 dictionary sample và trả về STRING hoặc None
    text = sample['text'].strip()
    if not text: return None
    
    try:
        # STAGE 1
        if stage == 1:
            match = re.search(r'SENT:\s*(.*?)\nAMR:\s*(.*)', text, re.DOTALL)
            if match:
                return create_prompt_stage1(match.group(1).strip(), match.group(2).strip())
            
            match = re.search(r'Input:\s*(.*?)\nOutput:\s*(.*)', text, re.DOTALL)
            if match:
                return create_prompt_stage1(match.group(1).strip(), match.group(2).strip())

        # STAGE 2
        else:
            match = re.search(r'SENT:\s*(.*?)\nNO_VAR:\s*(.*?)\nFULL:\s*(.*)', text, re.DOTALL)
            if match:
                return create_prompt_stage2(match.group(1).strip(), match.group(2).strip(), match.group(3).strip())
            
            match = re.search(r'Input:\s*(.*?)<sep>(.*?)\nOutput:\s*(.*)', text, re.DOTALL)
            if match:
                return create_prompt_stage2(match.group(1).strip(), match.group(2).strip(), match.group(3).strip())

        return None
    except Exception:
        return None

# ==========================================
# 2. TRAINING SETUP
# ==========================================

def load_and_filter_dataset(file_path, stage):
    print(f"📂 Reading file: {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Cannot find file: {file_path}")
        
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    blocks = content.strip().split('\n\n')
    blocks = [b for b in blocks if b.strip()]
    
    # Chỉ giữ lại các block raw text hợp lệ trước
    valid_data = []
    for b in blocks:
        # Test format thử, nếu ok thì giữ
        if format_data({'text': b}, stage): 
            valid_data.append(b)
            
    print(f"Dataset: {len(blocks)} raw -> {len(valid_data)} valid samples.")
    if len(valid_data) == 0:
        raise ValueError("❌ DATASET IS EMPTY!")
        
    return Dataset.from_dict({"text": valid_data})

def train(args):
    print(f"🚀 START TRAINING STAGE {args.stage} | GPU 48GB Optimization")
    
    # 1. Load Data
    raw_dataset = load_and_filter_dataset(args.data_path, args.stage)
    
    # 2. APPLY FORMATTING MANUALLY (FIX LỖI HERE)
    # Thay vì để Trainer format, ta format luôn tại đây
    print("🛠️  Pre-formatting dataset...")
    
    def apply_format_map(batch):
        # Batch['text'] là list các raw block
        formatted_prompts = []
        for raw_text in batch['text']:
            prompt = format_data({'text': raw_text}, args.stage)
            formatted_prompts.append(prompt)
        return {"text": formatted_prompts} # Ghi đè cột text bằng prompt chuẩn
    
    dataset = raw_dataset.map(apply_format_map, batched=True)
    
    print("✨ GPU 48GB Detected: Loading model in BFloat16")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,       
        device_map="auto",
        attn_implementation="sdpa" 
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    peft_config = LoraConfig(
        lora_alpha=64,
        lora_dropout=0.05,
        r=128,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

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

    trainer_kwargs = {
        "model": model,
        "train_dataset": dataset, # Dataset đã format sẵn
        "peft_config": peft_config,
        "processing_class": tokenizer,
        "args": training_args,
        "dataset_text_field": "text", # Chỉ định cột text đã format
        # KHÔNG TRUYỀN formatting_func NỮA ĐỂ TRÁNH LỖI
    }
    
    sig = inspect.signature(SFTTrainer.__init__)
    if 'max_seq_length' in sig.parameters:
        trainer_kwargs['max_seq_length'] = 2048
        trainer_kwargs['packing'] = False

    trainer = SFTTrainer(**trainer_kwargs)

    trainer.train()
    
    final_path = os.path.join(args.output_dir, "final_adapter")
    trainer.model.save_pretrained(final_path)
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