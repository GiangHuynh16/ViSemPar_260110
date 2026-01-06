import re
import sys

# === CẤU HÌNH ===
INPUT_PRED_FILE = "evaluation_results/mtup/final_amr_submission.txt"
OUTPUT_FIXED_FILE = "evaluation_results/mtup/final_amr_submission_fixed.txt"

def fix_amr_string(amr_string):
    """Hàm sửa các lỗi cú pháp phổ biến do LLM sinh ra"""
    if not amr_string or amr_string.strip() == "":
        return "(a / amr-empty)"

    # 1. Sửa lỗi khoảng trắng sau dấu hai chấm (VD: ": arg1" -> ":arg1")
    # Regex tìm dấu : theo sau là khoảng trắng và chữ cái
    amr_string = re.sub(r':\s+([a-zA-Z0-9-]+)', r':\1', amr_string)

    # 2. Cân bằng dấu ngoặc đơn (QUAN TRỌNG NHẤT)
    open_count = amr_string.count('(')
    close_count = amr_string.count(')')
    
    if open_count > close_count:
        # Thiếu ngoặc đóng -> Thêm vào cuối
        amr_string += ')' * (open_count - close_count)
    elif close_count > open_count:
        # Thừa ngoặc đóng -> Cắt bớt từ cuối (nguy hiểm hơn, nhưng cần thiết)
        # Cách an toàn: Giữ nguyên, hy vọng parser bỏ qua, hoặc xóa bớt
        # Ở đây ta chọn cách xóa bớt các ký tự ) ở cuối chuỗi
        diff = close_count - open_count
        amr_string = amr_string.rstrip()
        if amr_string.endswith(')' * diff):
             amr_string = amr_string[:-diff]
    
    # 3. Sửa lỗi biến bị trùng hoặc sai format (cơ bản)
    # VD: (t / tôi) -> model đôi khi sinh (t/tôi) dính liền
    amr_string = amr_string.replace("/", " / ")
    # Xóa khoảng trắng thừa do bước trên tạo ra
    amr_string = re.sub(r'\s+', ' ', amr_string).strip()
    
    # 4. Kiểm tra xem có bắt đầu bằng ( không, nếu không thì wrap lại
    if not amr_string.startswith("("):
        # Cố gắng tìm điểm bắt đầu
        start = amr_string.find("(")
        if start != -1:
            amr_string = amr_string[start:]
        else:
            return "(a / amr-empty)" # Không cứu được

    return amr_string

def main():
    print(f"🔧 Đang sửa lỗi file: {INPUT_PRED_FILE}")
    
    with open(INPUT_PRED_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    fixed_lines = []
    error_count = 0
    
    for i, line in enumerate(lines):
        line = line.strip()
        # Bỏ qua các dòng metadata nếu có lỡ lọt vào
        if line.startswith("#"):
            continue
            
        fixed_amr = fix_amr_string(line)
        
        # Kiểm tra sơ bộ
        if fixed_amr == "(a / amr-empty)" and line != "(a / amr-empty)":
            error_count += 1
            print(f"⚠️ Dòng {i+1} không thể sửa, thay thế bằng graph rỗng.")
            
        fixed_lines.append(fixed_amr)
        
    print(f"📊 Đã xử lý {len(lines)} dòng.")
    print(f"🛠️ Đã sửa lỗi và lưu vào: {OUTPUT_FIXED_FILE}")

    # Ghi file
    with open(OUTPUT_FIXED_FILE, 'w', encoding='utf-8') as f_out:
        for line in fixed_lines:
            f_out.write(line + "\n")

if __name__ == "__main__":
    main()