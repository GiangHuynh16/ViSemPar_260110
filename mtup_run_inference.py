import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm
import gc
import re
import os

# ================= CONFIG =================
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
STAGE1_ADAPTER = "checkpoints/mtup/stage1_v2/final_adapter"
STAGE2_ADAPTER = "checkpoints/mtup/stage2_v2/final_adapter"

INPUT_FILE = "data/public_test" # File chứa câu input
TEMP_STAGE1_OUT = "evaluation_results/mtup_v2/pred_stage1_skeleton.txt"
FINAL_RAW_OUT = "evaluation_results/mtup_v2/pred_final_raw.txt"

# Tạo thư mục output nếu chưa có
os.makedirs("evaluation_results/mtup_v2", exist_ok=True)

# ================= PROMPTS (PHẢI GIỐNG HỆT LÚC TRAIN) =================
def create_prompt_stage1(sentence):
    sys_prompt = """Bạn là một hệ thống phân tích ngữ nghĩa AMR (Abstract Meaning Representation).
Nhiệm vụ: Chuyển câu tiếng Việt thành cấu trúc AMR-No-Var (chưa có biến).
Quy tắc tuyệt đối:
1. KHÔNG tự tạo biến (ví dụ: dùng '(tôi)' thay vì '(t / tôi)').
2. Đảm bảo số lượng ngoặc mở '(' bằng số lượng ngoặc đóng ')'.
3. Output chỉ chứa duy nhất đồ thị AMR, không giải thích thêm."""
    
    # Format Qwen Chat
    return f"<|im_start|>system\n{sys_prompt}<|im_end|>\n<|im_start|>user\nCâu: {sentence}<|im_end|>\n<|im_start|>assistant\n"

def create_prompt_stage2(sentence, skeleton):
    sys_prompt = """Bạn là chuyên gia gán biến cho đồ thị AMR (AMR Aligner).
Nhiệm vụ: Thêm biến (variable) vào cấu trúc AMR thô.
Quy tắc SỐNG CÒN để tránh lỗi Duplicate Node:
1. Mỗi Concept chỉ được định nghĩa biến MỘT lần duy nhất. Ví dụ: (t / tôi).
2. TÁI SỬ DỤNG (Re-entrancy): Nếu concept xuất hiện lại, CHỈ dùng tên biến, KHÔNG viết lại concept.
   - SAI: :ARG0 (t / tôi) ... :ARG1 (t / tôi)
   - ĐÚNG: :ARG0 (t / tôi) ... :ARG1 t
3. Hãy dùng các chữ cái đầu làm tên biến (v / viết), nếu trùng thì thêm số (v2 / viết)."""
    
    return f"<|im_start|>system\n{sys_prompt}<|im_end|>\n<|im_start|>user\nCâu: {sentence}\nSkeletion: {skeleton}<|im_end|>\n<|im_start|>assistant\n"

# ================= HELPER FUNCTIONS =================
def load_model(adapter_path):
    print(f"🔄 Loading Adapter: {adapter_path}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb_config, device_map="auto", attn_implementation="sdpa"
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer

def clean_memory(model, tokenizer):
    print("🧹 Cleaning VRAM...")
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

def infer_batch(model, tokenizer, prompts, batch_size=8):
    results = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Inferencing"):
        batch_prompts = prompts[i:i+batch_size]
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=512, 
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False, # Greedy decoding để kết quả ổn định nhất
                num_beams=1
            )
        
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        # Cắt bỏ phần prompt, chỉ lấy phần output mới sinh ra
        for j, text in enumerate(decoded):
            # Qwen output thường dính cả prompt, cần split
            # Tìm keyword "assistant" cuối cùng
            parts = text.split("assistant\n")
            if len(parts) > 1:
                result = parts[-1].strip()
            else:
                result = text.strip()
            results.append(result)
    return results

# ================= MAIN PIPELINE =================
def main():
    # 1. Đọc dữ liệu Input
    print(f"📖 Reading input: {INPUT_FILE}")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        # Giả sử file public_test chỉ chứa các câu (mỗi câu 1 dòng)
        # Nếu file có dạng SENT: ..., code sẽ strip sạch
        lines = [line.strip().replace("SENT:", "").strip() for line in f.readlines() if line.strip()]

    # ---------------- STAGE 1 ----------------
    print("\n🚀 STARTING STAGE 1 (Structure Generation)...")
    model, tokenizer = load_model(STAGE1_ADAPTER)
    tokenizer.padding_side = "left" # Quan trọng cho batch generation
    
    prompts_s1 = [create_prompt_stage1(sent) for sent in lines]
    skeletons = infer_batch(model, tokenizer, prompts_s1, batch_size=4) # Giảm batch nếu OOM
    
    # Lưu tạm
    with open(TEMP_STAGE1_OUT, 'w', encoding='utf-8') as f:
        for s in skeletons: f.write(s + "\n")
    
    clean_memory(model, tokenizer) # Giải phóng RAM

    # ---------------- STAGE 2 ----------------
    print("\n🚀 STARTING STAGE 2 (Variable Alignment)...")
    model, tokenizer = load_model(STAGE2_ADAPTER)
    tokenizer.padding_side = "left"

    prompts_s2 = [create_prompt_stage2(sent, skel) for sent, skel in zip(lines, skeletons)]
    final_amrs = infer_batch(model, tokenizer, prompts_s2, batch_size=4)
    
    clean_memory(model, tokenizer)

    # Lưu kết quả thô
    with open(FINAL_RAW_OUT, 'w', encoding='utf-8') as f:
        for amr in final_amrs:
            f.write(amr + "\n")
            
    print(f"✅ Inference Complete! Raw output saved to: {FINAL_RAW_OUT}")

if __name__ == "__main__":
    main()