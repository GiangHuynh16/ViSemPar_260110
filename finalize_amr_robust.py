import sys
import re
import argparse

try:
    import penman
    from penman.models.amr import model as amr_model
except ImportError:
    print("❌ Lỗi: Chưa cài thư viện penman.")
    print("👉 Hãy chạy: pip install penman")
    sys.exit(1)

# === CẤU HÌNH ===
# File đang bị lỗi của bạn
INPUT_FILE = "evaluation_results/mtup/final_amr_ready_for_smatch.txt"
# File kết quả sạch sẽ 100%
OUTPUT_FILE = "evaluation_results/mtup/final_amr_submittable.txt"

def ensure_unique_variables(amr_string):
    """
    Hàm này can thiệp trực tiếp vào chuỗi AMR để đổi tên các biến bị định nghĩa trùng.
    Ví dụ: (a / boy :ARG0 (a / girl)) -> (a / boy :ARG0 (a_2 / girl))
    """
    if not amr_string or "(" not in amr_string:
        return amr_string

    # Regex tìm các đoạn định nghĩa biến: (tên_biến /
    # Group 1: tên biến
    pattern = re.compile(r'\(\s*([a-z0-9][a-z0-9-]*)\s*/', re.IGNORECASE)
    
    seen_vars = {} # Đếm số lần xuất hiện của biến định nghĩa
    
    # Chúng ta sẽ build lại chuỗi từ đầu
    new_string = ""
    last_end = 0
    
    for match in pattern.finditer(amr_string):
        start, end = match.span()
        var_name = match.group(1)
        
        # Thêm phần text trước match vào chuỗi mới
        new_string += amr_string[last_end:start]
        
        # Xử lý tên biến
        if var_name in seen_vars:
            # Nếu biến đã được định nghĩa trước đó -> Đổi tên
            seen_vars[var_name] += 1
            new_var_name = f"{var_name}_{seen_vars[var_name]}"
        else:
            # Lần đầu gặp -> Giữ nguyên
            seen_vars[var_name] = 1
            new_var_name = var_name
            
        # Thêm phần định nghĩa mới vào: "(var /"
        new_string += f"({new_var_name} /"
        
        last_end = end
        
    # Thêm phần đuôi còn lại
    new_string += amr_string[last_end:]
    return new_string

def aggressive_syntax_fix(text):
    """Sửa các lỗi cú pháp cơ bản (ngoặc, dấu hai chấm)"""
    if not text: return ""
    
    # 1. Fix lỗi khoảng trắng sau dấu : (vd: ": arg0" -> ":arg0")
    text = re.sub(r':\s+([a-zA-Z0-9-]+)', r':\1', text)
    
    # 2. Xóa các node rác kiểu :wiki( hoặc :op1( ở cuối dòng
    text = re.sub(r':[a-z0-9-]+\s*[({]?\s*$', '', text)

    # 3. Cân bằng ngoặc
    open_c = text.count('(')
    close_c = text.count(')')
    if open_c > close_c:
        text += ')' * (open_c - close_c)
    elif close_c > open_c:
        # Cắt bớt ngoặc đóng thừa
        diff = close_c - open_c
        # Chỉ cắt nếu nó nằm ở cuối
        if text.endswith(')' * diff):
            text = text[:-diff]
            
    return text

def validate_and_repair(line, line_num):
    """
    Cố gắng parse bằng Penman. 
    Nếu lỗi -> Fix Syntax -> Fix Duplicate -> Parse lại.
    Nếu vẫn lỗi -> Trả về Empty Graph.
    """
    line = line.strip()
    if not line or line.startswith("#"):
        return "(e / amr-empty)"

    # Bước 1: Thử parse nguyên bản
    try:
        g = penman.decode(line)
        # Nếu parse được, encode lại để chuẩn hóa format
        return penman.encode(g, indent=None)
    except Exception:
        pass # Lỗi thì đi tiếp xuống dưới

    # Bước 2: Fix cú pháp + Fix trùng biến
    fixed_line = aggressive_syntax_fix(line)
    fixed_line = ensure_unique_variables(fixed_line)
    
    # Bước 3: Thử parse lại
    try:
        g = penman.decode(fixed_line)
        return penman.encode(g, indent=None)
    except Exception as e:
        # Bước 4: Vẫn lỗi -> Bỏ cuộc, trả về rỗng để cứu chương trình
        print(f"⚠️ Dòng {line_num}: Không thể parse AMR (Lỗi: {str(e)[:50]}...). Thay thế bằng graph rỗng.")
        return "(e / amr-empty)"

def main():
    print(f"🚀 Bắt đầu quá trình sửa lỗi AMR toàn diện...")
    print(f"📂 Input: {INPUT_FILE}")
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    clean_lines = []
    
    for i, line in enumerate(lines):
        clean_line = validate_and_repair(line, i+1)
        clean_lines.append(clean_line)
        
    # Ghi file
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        for line in clean_lines:
            f_out.write(line + "\n")
            
    print(f"✅ Đã xử lý xong {len(lines)} dòng.")
    print(f"💾 Kết quả lưu tại: {OUTPUT_FILE}")
    print("👉 Bây giờ bạn có thể chạy smatch mà không lo bị crash nữa!")

if __name__ == "__main__":
    main()