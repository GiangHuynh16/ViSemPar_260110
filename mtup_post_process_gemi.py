import sys
import re

def linearize_amr(amr_string):
    """Xóa xuống dòng thừa, đưa về 1 dòng duy nhất"""
    return re.sub(r'\s+', ' ', amr_string).strip()

def clean_and_align(pred_file, gold_file, output_file):
    print("🧹 Cleaning and Aligning outputs...")
    
    # 1. Đọc file Gold để biết số lượng câu chuẩn
    with open(gold_file, 'r', encoding='utf-8') as f:
        gold_content = f.read().strip()
        # Tách các graph trong gold (thường cách nhau bằng dòng trống hoặc bắt đầu bằng dấu ngoặc)
        # Cách đơn giản nhất: đếm số lượng câu bắt đầu bằng #::snt nếu có, hoặc đếm block
        gold_blocks = gold_content.split('\n\n') 
        gold_blocks = [b for b in gold_blocks if b.strip()]
        num_samples = len(gold_blocks)
    
    print(f"   -> Gold file has {num_samples} samples.")

    # 2. Đọc file Predicted
    with open(pred_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Logic xử lý: File pred hiện tại có thể có nhiều dòng cho 1 câu.
    # Nhưng vì chúng ta chạy predict theo batch/loop, nếu code predict viết 'w' (ghi đè) mỗi lần loop thì sai, 
    # nhưng code tôi đưa là append vào list rồi write 1 lần.
    # -> Khả năng cao là trong 1 prompt, model sinh ra output có chứa ký tự xuống dòng "\n".
    
    # Chúng ta sẽ đọc file pred, nếu thấy dòng bắt đầu bằng "(" thì coi là 1 graph.
    # Tuy nhiên, để khớp 1-1, ta cần biết code predict đã ghi ra bao nhiêu dòng.
    # Nếu code predict ghi đúng số dòng bằng số input, thì chỉ cần linearize.
    
    cleaned_amrs = []
    for line in lines:
        line = line.strip()
        if not line: continue
        # Nếu dòng chứa nhiều graph con (vd: "(a / b) (c / d)"), format lại
        # Smatch yêu cầu 1 root duy nhất. Nếu model in ra 2 root cạnh nhau, ta phải bao nó lại bằng (m / multi-sentence)
        
        # Đếm số lượng mở ngoặc ở cấp cao nhất
        open_count = 0
        roots = []
        current_root = ""
        for char in line:
            current_root += char
            if char == '(': open_count += 1
            if char == ')': open_count -= 1
            if open_count == 0 and current_root.strip():
                roots.append(current_root.strip())
                current_root = ""
        
        if len(roots) > 1:
            # Gộp nhiều mảnh thành 1 multi-sentence
            merged = "(m / multi-sentence"
            for i, r in enumerate(roots):
                merged += f" :snt{i+1} {r}"
            merged += ")"
            cleaned_amrs.append(merged)
        else:
            cleaned_amrs.append(line)

    # Cắt hoặc thêm cho đủ số lượng (Padding/Truncating)
    if len(cleaned_amrs) > num_samples:
        print(f"⚠️ Warning: Prediction has {len(cleaned_amrs)} lines, Gold has {num_samples}. Truncating...")
        cleaned_amrs = cleaned_amrs[:num_samples]
    elif len(cleaned_amrs) < num_samples:
        print(f"⚠️ Warning: Prediction has {len(cleaned_amrs)} lines, Gold has {num_samples}. Padding with empty graphs...")
        while len(cleaned_amrs) < num_samples:
            cleaned_amrs.append("(a / amr-empty)")

    # 3. Lưu file sạch
    with open(output_file, 'w', encoding='utf-8') as f:
        for amr in cleaned_amrs:
            f.write(linearize_amr(amr) + "\n")
            
    print(f"✅ Saved cleaned predictions to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python post_process_amr.py <pred_file> <gold_file> <output_file>")
    else:
        clean_and_align(sys.argv[1], sys.argv[2], sys.argv[3])