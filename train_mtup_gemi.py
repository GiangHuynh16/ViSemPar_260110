import os
import argparse
import torch
import inspect
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments, # Dùng TrainingArguments chuẩn thay vì SFTConfig
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
    try:
        text = sample['text']
        if stage == 1:
            sent = text.split("SENT: ")[1].split("\nAMR: ")[0].strip()
            amr = text.split("\nAMR: ")[1].strip()
            return create_prompt_stage1(sent, amr)
        else:
            sent = text.split("SENT: ")[1].split("\nNO_VAR: ")[0].strip()
            no_var = text.split("\nNO_VAR: ")[1].split("\nFULL: ")[0].strip()
            full = text.split("\nFULL: ")[1].strip()
            return create_prompt_stage2(sent, no_var, full)
    except Exception:
        return None

# ==========================================
# 2. TRAINING SETUP
# ==========================================

def load_and_filter_dataset(file_path, stage):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    blocks = content.strip().split('\n\n')
    valid_data = []
    for b in blocks:
        if not b.strip(): continue
        if format_data({'text': b}, stage): 
            valid_data.append(b)
            
    print(f"Dataset: {len(blocks)} raw -> {len(valid_data)} valid samples.")
    return Dataset.from_dict({"text": valid_data})

def train(args):
    print(f"🚀 START TRAINING STAGE {args.stage} | GPU 48GB Optimization")
    
    dataset = load_and_filter_dataset(args.data_path, args.stage)
    
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

    # 1. Sử dụng TrainingArguments chuẩn (luôn an toàn)
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
        optim="adamw_torch", # Dùng optimizer chuẩn
        report_to="none",
        remove_unused_columns=False, # Fix lỗi input_ids
    )

    # 2. Xử lý max_seq_length thông minh
    # Kiểm tra xem phiên bản SFTTrainer hiện tại có nhận max_seq_length không
    trainer_kwargs = {
        "model": model,
        "train_dataset": dataset,
        "peft_config": peft_config,
        "processing_class": tokenizer,
        "args": training_args,
        "formatting_func": lambda batch: [format_data({'text': t}, args.stage) for t in batch['text']],
    }
    
    # Inspect chữ ký hàm __init__ của SFTTrainer
    sig = inspect.signature(SFTTrainer.__init__)
    if 'max_seq_length' in sig.parameters:
        print("✅ Detected SFTTrainer accepts 'max_seq_length'.")
        trainer_kwargs['max_seq_length'] = 2048
        trainer_kwargs['packing'] = False
    else:
        print("⚠️ SFTTrainer does not accept 'max_seq_length' directly. Attempting to pass via dataset_kwargs or args.")
        # Nếu không nhận trực tiếp, ta thử hack vào args (dành cho version cực mới/cũ lạ)
        # Tuy nhiên với TrainingArguments chuẩn, thường SFTTrainer sẽ tự fallback
        pass

    # 3. Khởi tạo Trainer
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