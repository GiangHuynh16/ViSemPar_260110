import os
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm

def load_model(base_model_path, adapter_path):
    print(f"Loading base model: {base_model_path}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation="sdpa"
    )
    
    print(f"Loading adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    return model, tokenizer

def create_prompt(text, stage):
    # Dùng đúng Template lúc train
    if stage == 1:
        sys_prompt = "Bạn là một chuyên gia ngôn ngữ học về cấu trúc AMR (Abstract Meaning Representation).\nNhiệm vụ: Chuyển đổi câu tiếng Việt đầu vào sang định dạng đồ thị AMR chuẩn PENMAN.\nYêu cầu đặc biệt:\n1. KHÔNG sử dụng biến (variables) định danh (ví dụ: không dùng 't / tôi', chỉ dùng '(tôi)').\n2. Đảm bảo cấu trúc ngoặc đơn () cân bằng chính xác.\n3. Chỉ giữ lại các Concept và Relation (ví dụ: :ARG0, :ARG1, :mod)."
        user_input = f"Input: {text}"
    else:
        sys_prompt = "Bạn là một chuyên gia gán nhãn dữ liệu AMR (Abstract Meaning Representation).\nNhiệm vụ: Hoàn thiện đồ thị AMR từ cấu trúc thô (chưa có biến) và câu gốc.\n\nYêu cầu QUAN TRỌNG:\n1. Gán biến (variables) định danh cho mỗi concept (vd: '(tôi)' -> '(t / tôi)').\n2. TÁI SỬ DỤNG BIẾN (Re-entrancy): Nếu một đối tượng xuất hiện nhiều lần hoặc được thay thế bằng đại từ (anh ấy, nó, cậu ta...), hãy dùng lại biến đã khai báo trước đó thay vì tạo biến mới.\n3. Đảm bảo đúng định dạng PENMAN."
        user_input = f"Input: {text}" # Text ở đây đã bao gồm <sep> skeleton nếu là stage 2
        
    prompt = f"<|im_start|>system\n{sys_prompt}<|im_end|>\n<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"
    return prompt

def predict(args):
    model, tokenizer = load_model(args.base_model, args.adapter_path)
    
    with open(args.input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    # Lọc lấy input từ file test (Giả sử file test có format "SENT: ...")
    inputs = []
    for line in lines:
        if line.startswith("SENT:") or line.startswith("Input:"):
            clean_line = line.replace("SENT:", "").replace("Input:", "").strip()
            inputs.append(clean_line)
    
    # Nếu file chỉ chứa raw text mỗi dòng 1 câu thì dùng: inputs = [l.strip() for l in lines]
            
    print(f"Generating for {len(inputs)} samples...")
    
    results = []
    for text in tqdm(inputs):
        prompt = create_prompt(text, args.stage)
        inputs_ids = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs_ids, 
                max_new_tokens=1024,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False, # Greedy decoding để kết quả ổn định
                repetition_penalty=1.1
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Cắt bỏ phần prompt, chỉ lấy phần output của assistant
        response = generated_text.split("assistant\n")[-1].strip()
        results.append(response)

    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    with open(args.output_file, 'w', encoding='utf-8') as f:
        for res in results:
            f.write(res + "\n")
            
    print(f"Done! Results saved to {args.output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--adapter_path", type=str, required=True)
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--stage", type=int, default=1)
    args = parser.parse_args()
    predict(args)