import sys
import re

try:
    import penman
except ImportError:
    print("❌ Lỗi: Chưa cài thư viện penman.")
    print("👉 Hãy chạy: pip install penman")
    sys.exit(1)

# === CẤU HÌNH ===
# File đầu vào (đang bị lỗi)
INPUT_FILE = "evaluation_results/mtup/final_amr_ready_for_smatch.txt"
# File đầu ra (sạch hoàn toàn)
OUTPUT_FILE = "evaluation_results/mtup/final_amr_submittable.txt"

def ensure_unique_variables(amr_string):
    """
    Hàm này quét chuỗi AMR và đổi tên các biến bị định nghĩa lại.
    Ví dụ: (a / person :ARG0 (a / person)) -> (a / person :ARG0 (a_2 / person))
    """
    if not amr_string or "(" not in amr_string:
        return amr_string

    # Regex tìm các đoạn định nghĩa biến: (tên_biến /
    # Group 1: tên biến
    pattern = re.compile(r'\(\s*([a-z0-9][a-z0-9-]*)\s*/', re.IGNORECASE)
    
    seen_vars = {} # Đếm số lần xuất hiện của biến định nghĩa
    
    # Xây dựng lại chuỗi từ đầu để đảm bảo thay thế đúng vị trí
    new_string = ""
    last_end = 0
    
    for match in pattern.finditer(amr_string):
        start, end = match.span()
        var_name = match.group(1)
        
        # Thêm phần text trước match vào chuỗi mới
        new_string += amr_string[last_end:start]
        
        # Xử lý tên biến
        if var_name in seen_vars:
            # Nếu biến đã được định nghĩa trước đó -> Đổi tên (a -> a_2)
            seen_vars[var_name] += 1
            new_var_name = f"{var_name}_{seen_vars[var_name]}"
        else:
            # Lần đầu gặp -> Giữ nguyên
            seen_vars[var_name] = 1
            new_var_name = var_name
            
        # Thêm phần định nghĩa mới vào: "(var /"
        new_string += f"({new_var_name} /"
        
        last_end = end
        
    # Thêm phần đuôi còn lại của chuỗi
    new_string += amr_string[last_end:]
    return new_string

def aggressive_syntax_fix(text):
    """Sửa các lỗi cú pháp cơ bản (ngoặc, dấu hai chấm, khoảng trắng)"""
    if not text: return ""
    
    # 1. Fix lỗi khoảng trắng sau dấu : (vd: ": arg0" -> ":arg0")
    text = re.sub(r':\s+([a-zA-Z0-9-]+)', r':\1', text)
    
    # 2. Xóa các node rác kiểu :wiki( hoặc :op1( treo lơ lửng ở cuối dòng
    text = re.sub(r':[a-z0-9-]+\s*[({]?\s*$', '', text)

    # 3. Cân bằng ngoặc đơn
    open_c = text.count('(')
    close_c = text.count(')')
    if open_c > close_c:
        text += ')' * (open_c - close_c)
    elif close_c > open_c:
        # Cắt bớt ngoặc đóng thừa ở cuối
        diff = close_c - open_c
        if text.endswith(')' * diff):
            text = text[:-diff]
            
    return text

def validate_and_repair(line, line_num):
    """
    Quy trình sửa lỗi 3 bước:
    1. Parse thử.
    2. Nếu lỗi -> Fix syntax -> Fix trùng biến -> Parse thử lại.
    3. Nếu vẫn lỗi -> Trả về Graph rỗng (để cứu smatch khỏi crash).
    """
    line = line.strip()
    if not line or line.startswith("#"):
        return "(e / amr-empty)"

    # --- Bước 1: Thử parse nguyên bản ---
    try:
        g = penman.decode(line)
        return penman.encode(g, indent=None) # Encode lại để chuẩn hóa
    except Exception:
        pass # Lỗi thì đi tiếp

    # --- Bước 2: Fix cú pháp + Fix trùng biến (QUAN TRỌNG) ---
    fixed_line = aggressive_syntax_fix(line)
    fixed_line = ensure_unique_variables(fixed_line)
    
    # --- Bước 3: Thử parse lại lần nữa ---
    try:
        g = penman.decode(fixed_line)
        return penman.encode(g, indent=None)
    except Exception as e:
        # --- Bước 4: Vẫn lỗi -> Bỏ cuộc, trả về rỗng ---
        print(f"⚠️ Dòng {line_num} lỗi quá nặng (Duplicate/Structure). Thay thế bằng graph rỗng.")
        return "(e / amr-empty)"

def main():
    print(f"🚀 Bắt đầu sửa lỗi AMR Robust...")
    print(f"📂 Input: {INPUT_FILE}")
    
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"❌ Không tìm thấy file {INPUT_FILE}")
        return
        
    clean_lines = []
    
    for i, line in enumerate(lines):
        # Validate từng dòng một
        clean_line = validate_and_repair(line, i+1)
        clean_lines.append(clean_line)
        
    # Ghi file kết quả
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        for line in clean_lines:
            f_out.write(line + "\n")
            
    print(f"✅ Đã xử lý xong {len(lines)} dòng.")
    print(f"💾 File sạch 100% để chạy smatch: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()