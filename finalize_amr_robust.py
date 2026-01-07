import sys
import re

try:
    import penman
except ImportError:
    print("❌ Lỗi: Chưa cài thư viện penman.")
    sys.exit(1)

# === INPUT / OUTPUT ===
INPUT_FILE = "evaluation_results/mtup/final_amr_ready_for_smatch.txt" # File đang lỗi
OUTPUT_FILE = "evaluation_results/mtup/final_amr_nuclear_clean.txt"   # File sạch 100%

def force_resolve_duplicates(amr_string):
    """
    Tìm mọi pattern (biến / concept) và đổi tên nếu biến đã xuất hiện.
    Sử dụng Regex linh hoạt hơn để bắt mọi loại tên biến.
    """
    if not amr_string or "(" not in amr_string:
        return amr_string

    # Regex bắt pattern: ( tên_biến /
    # [^\s/()]+ nghĩa là chuỗi ký tự không chứa khoảng trắng, /, (, )
    pattern = re.compile(r'\(\s*([^\s/()]+)\s*/')
    
    seen_vars = {} 
    new_string = ""
    last_end = 0
    
    for match in pattern.finditer(amr_string):
        start, end = match.span()
        var_name = match.group(1)
        
        new_string += amr_string[last_end:start]
        
        # Logic đổi tên
        if var_name in seen_vars:
            seen_vars[var_name] += 1
            # Thêm suffix số đếm để đảm bảo unique
            new_var_name = f"{var_name}_dup{seen_vars[var_name]}"
        else:
            seen_vars[var_name] = 1
            new_var_name = var_name
            
        new_string += f"({new_var_name} /"
        last_end = end
        
    new_string += amr_string[last_end:]
    return new_string

def sanitize_line(line, line_num):
    """
    Quy trình lọc cực đoan:
    1. Clean text cơ bản.
    2. Force đổi tên biến trùng.
    3. PENMAN VALIDATION (Quan trọng nhất).
    """
    line = line.strip()
    if not line or line.startswith("#"):
        return "(e / amr-empty)"

    # 1. Sửa lỗi syntax cơ bản
    line = re.sub(r':\s+([a-zA-Z0-9-]+)', r':\1', line) # : arg -> :arg
    line = re.sub(r':[a-z0-9-]+\s*[({]?\s*$', '', line) # Xóa node rác cuối dòng
    
    # Cân bằng ngoặc
    open_c = line.count('(')
    close_c = line.count(')')
    if open_c > close_c: line += ')' * (open_c - close_c)
    elif close_c > open_c: line = line[:-(close_c - open_c)]

    # 2. Xử lý trùng biến (Nguyên nhân crash chính)
    line = force_resolve_duplicates(line)

    # 3. KIỂM TRA BẰNG PENMAN
    try:
        # Nếu dòng này parse được -> OK -> Return
        g = penman.decode(line)
        # Encode lại để chuẩn hóa format (xóa khoảng trắng thừa)
        return penman.encode(g, indent=None)
    except Exception as e:
        # Nếu vẫn lỗi -> BỎ LUÔN -> Trả về rỗng
        # Đây là bước chặn crash
        print(f"⚠️ Dòng {line_num} bị lỗi AM (Duplicate/Syntax) không thể sửa. Thay thế bằng graph rỗng.")
        return "(e / amr-empty)"

def main():
    print(f"☢️  NUCLEAR FIX MODE ACTIVATED")
    print(f"📂 Reading: {INPUT_FILE}")
    
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print("❌ Không tìm thấy file input.")
        return

    clean_lines = []
    replaced_count = 0
    
    for i, line in enumerate(lines):
        original_line = line
        processed_line = sanitize_line(line, i+1)
        
        if processed_line == "(e / amr-empty)" and "(e / amr-empty)" not in original_line:
            replaced_count += 1
            
        clean_lines.append(processed_line)

    print(f"📊 Tổng số dòng: {len(lines)}")
    print(f"🔥 Số dòng bị thay thế bằng Empty (do lỗi nặng): {replaced_count}")
    print(f"✅ Đã ghi file sạch tuyệt đối vào: {OUTPUT_FILE}")
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        for line in clean_lines:
            f_out.write(line + "\n")

if __name__ == "__main__":
    main()