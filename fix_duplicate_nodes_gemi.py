import re
import sys

# === CẤU HÌNH ===
# File input là file bạn vừa clean ở bước trước (hoặc file gốc bị lỗi)
INPUT_FILE = "evaluation_results/mtup/final_amr_clean.txt" 
# Output ra file mới hoàn toàn sạch
OUTPUT_FILE = "evaluation_results/mtup/final_amr_ready_for_smatch.txt"

def rename_duplicates(amr_line):
    """
    Tìm các biến được định nghĩa (var / concept)
    Nếu var xuất hiện lần 2 ở vị trí định nghĩa, đổi tên thành var_2, var_3...
    """
    if not amr_line or amr_line.strip().startswith("#"):
        return amr_line

    # Pattern tìm định nghĩa biến: dấu mở ngoặc (, khoảng trắng, tên biến, khoảng trắng, dấu /
    # Group 1: tên biến
    pattern = re.compile(r'\(\s*([a-z0-9][a-z0-9-]*)\s*/')
    
    seen_vars = {} # Lưu các biến đã thấy trong dòng này: {name: count}
    
    # Chúng ta không thể thay thế trực tiếp bằng string.replace vì sẽ hỏng các tham chiếu.
    # Chúng ta sẽ duyệt và xây dựng lại chuỗi.
    
    new_line = ""
    last_idx = 0
    
    # Duyệt qua tất cả các vị trí định nghĩa biến
    for match in pattern.finditer(amr_line):
        start, end = match.span()
        var_name = match.group(1)
        
        # Copy phần text từ lần match trước đến match này
        new_line += amr_line[last_idx:start]
        
        # Xử lý biến
        if var_name in seen_vars:
            # Nếu đã gặp biến này rồi -> Đây là duplicate -> Cần đổi tên
            seen_vars[var_name] += 1
            new_var_name = f"{var_name}_{seen_vars[var_name]}" # vd: a -> a_2
        else:
            # Lần đầu gặp -> Giữ nguyên
            seen_vars[var_name] = 1
            new_var_name = var_name
            
        # Thêm phần định nghĩa đã (hoặc không) đổi tên vào chuỗi mới
        # Cấu trúc gốc: (var /
        # Cấu trúc mới: (new_var /
        new_line += f"({new_var_name} /"
        
        last_idx = end
        
    # Thêm phần còn lại của chuỗi
    new_line += amr_line[last_idx:]
    
    return new_line

def main():
    print(f"🔧 Đang quét và sửa lỗi Duplicate Nodes trong: {INPUT_FILE}")
    
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"❌ Không tìm thấy file {INPUT_FILE}. Hãy kiểm tra lại tên file.")
        return

    fixed_lines = []
    count_fixed = 0
    
    for line in lines:
        line = line.strip()
        if not line: continue
        
        # Sửa lỗi duplicate
        fixed_line = rename_duplicates(line)
        
        if fixed_line != line:
            count_fixed += 1
            
        # Kiểm tra lần cuối: Nếu dòng quá ngắn hoặc lỗi, thay bằng rỗng để tránh crash
        if len(fixed_line) < 5 or not fixed_line.startswith("("):
            fixed_line = "(a / amr-empty)"
            
        fixed_lines.append(fixed_line)

    print(f"📊 Đã xử lý {len(lines)} dòng.")
    print(f"🛠️ Đã sửa tên biến trùng lặp cho {count_fixed} dòng.")
    print(f"✅ File kết quả: {OUTPUT_FILE}")

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        for line in fixed_lines:
            f_out.write(line + "\n")

if __name__ == "__main__":
    main()