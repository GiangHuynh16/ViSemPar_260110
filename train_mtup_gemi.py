import os
import argparse
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer

# --- TEMPLATE TIẾNG VIỆT ---
def format_prompt_vietnamese(sample, stage):
    text = sample['text']
    try:
        input_part = text.split("Input:")[1].split("Output:")[0].strip()
        output_part = text.split("Output:")[1].strip()
    except IndexError:
        return "" 

    if stage == 1:
        # Template Stage 1: Tạo cấu trúc
        instruction = "Hãy chuyển đổi câu tiếng Việt sau sang định dạng đồ thị ngữ nghĩa trừu tượng (AMR) nhưng loại bỏ các biến (variable), chỉ giữ lại concept và quan hệ."
        inp = input_part
    else:
        # Template Stage 2: Thêm biến & Alignment
        instruction = "Hãy hoàn thiện đồ thị AMR sau bằng cách thêm các biến (variable) chính xác để định danh và căn chỉnh (align) phù hợp với câu gốc."
        # Input ở đây đã có dạng: "Câu gốc <sep> AMR_no_vars" từ bước prepare
        inp = input_part 

    # Template Qwen Chat (ChatML format) hoặc Alpaca style translated
    # Ở đây dùng format rõ ràng kiểu Instruction-Input-Response Tiếng Việt
    prompt = f"""### Hướng dẫn:
{instruction}

### Đầu vào:
{inp}

### Phản hồi:
{output_part}
"""
    return prompt

def load_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    blocks = content.strip().split('\n\n')
    return Dataset.from_dict({"text": blocks})

def train(args):
    print(f"--- START TRAINING STAGE {args.stage} (Model: {args.model_name}) ---")
    
    dataset = load_dataset(args.data_path)
    
    # Config QLoRA cho 7B model
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation="flash_attention_2" if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else "eager"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    peft_config = LoraConfig(
        lora_alpha=32, # Tăng nhẹ cho 7B
        lora_dropout=0.05,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum, # Quan trọng để fit VRAM
        learning_rate=2e-4,
        weight_decay=0.01,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        optim="paged_adamw_32bit",
        report_to="none" # Tắt wandb nếu không cần
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        formatting_func=lambda x: [format_prompt_vietnamese(item, args.stage) for item in x],
        tokenizer=tokenizer,
        args=training_args,
        max_seq_length=1536, # Tăng lên xíu cho an toàn với 7B
    )

    trainer.train()
    
    # Save adapter
    final_path = os.path.join(args.output_dir, "final_adapter")
    trainer.model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"Training Done. Saved to {final_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    # Default đổi thành Qwen 2.5 7B
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct") 
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2) # Giảm batch vì model to
    parser.add_argument("--grad_accum", type=int, default=8) # Tăng accum để bù batch size
    
    args = parser.parse_args()
    train(args)