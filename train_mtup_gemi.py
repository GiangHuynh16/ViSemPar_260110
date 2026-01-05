import os
import argparse
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    # BitsAndBytesConfig, # Không dùng nữa vì chạy native BFloat16
)
from peft import LoraConfig
# SỬA 1: Import SFTConfig thay vì TrainingArguments (hoặc import cả 2 nhưng dùng SFTConfig là chính)
from trl import SFTTrainer, SFTConfig 

# ... (GIỮ NGUYÊN CÁC HÀM create_prompt_stage1, create_prompt_stage2, format_data, load_dataset_from_text) ...
# Để ngắn gọn tôi không paste lại đoạn create_prompt và format_data ở trên, 
# bạn giữ nguyên logic hàm đó nhé.

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
Nhiệm vụ: Hoàn thiện đồ thị AMR chuẩn PENMAN từ cấu trúc thô (chưa có biến) và câu gốc.

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
    text = sample['text']
    try:
        if stage == 1:
            sent = text.split("SENT: ")[1].split("\nAMR: ")[0].strip()
            amr = text.split("\nAMR: ")[1].strip()
            return create_prompt_stage1(sent, amr)
        else:
            sent = text.split("SENT: ")[1].split("\nNO_VAR: ")[0].strip()
            no_var = text.split("\nNO_VAR: ")[1].split("\nFULL: ")[0].strip()
            full = text.split("\nFULL: ")[1].strip()
            return create_prompt_stage2(sent, no_var, full)
    except:
        return ""

def load_dataset_from_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    blocks = content.strip().split('\n\n')
    return Dataset.from_dict({"text": blocks})

# --- HÀM TRAIN ĐÃ SỬA ---
def train(args):
    print(f"🚀 START TRAINING STAGE {args.stage} | GPU 48GB Optimization")
    
    dataset = load_dataset_from_text(args.data_path)
    
    print("✨ GPU 48GB Detected: Loading model in BFloat16")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,       
        device_map="auto",
        attn_implementation="sdpa" # Dùng sdpa cho lành
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

    # SỬA 2: Dùng SFTConfig thay cho TrainingArguments
    # SFTConfig chứa cả tham số training thường lẫn tham số của SFTTrainer (như max_seq_length)
    training_args = SFTConfig(
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
        optim="paged_adamw_32bit",
        report_to="none",
        
        # QUAN TRỌNG: max_seq_length bây giờ nằm ở đây
        max_seq_length=2048, 
        dataset_text_field="text", # Cần khai báo trường text dù dùng formatting_func để tránh warning
        packing=False # Tắt packing để tránh lỗi với formatting_func
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        formatting_func=lambda x: [format_data(item, args.stage) for item in x],
        # SỬA 3: Đổi 'tokenizer' thành 'processing_class' để fix warning
        processing_class=tokenizer, 
        args=training_args,
        # SỬA 4: Đã xóa max_seq_length ở đây vì nó đã nằm trong args (SFTConfig)
    )

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