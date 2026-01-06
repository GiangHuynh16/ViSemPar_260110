import sys
import re
try:
    import penman
except ImportError:
    print("❌ Lỗi: Chưa cài penman. Hãy chạy: pip install penman")
    sys.exit(1)

# === INPUT / OUTPUT ===
INPUT_FILE = "evaluation_results/mtup/final_amr_submission.txt" # File gốc từ model
OUTPUT_FILE = "evaluation_results/mtup/final_amr_clean.txt"     # File sạch 100%

def aggressive_fix(text):
    """Cố gắng sửa chuỗi text nát bươm thành AMR hợp lệ"""
    if not text or len(text.strip()) < 3: return "(a / amr-empty)"
    
    # 1. Xóa các key bị treo lơ lửng ở cuối (ví dụ: :wiki( hoặc :arg1)
    # Tìm các pattern :key chưa có value ở cuối câu
    text = re.sub(r':\w+\s*[({]?$', '', text)
    
    # 2. Cân bằng ngoặc
    open_c = text.count('(')
    close_c = text.count(')')
    if open_c > close_c:
        text += ')' * (open_c - close_c)
    elif close_c > open_c:
        # Cắt bớt ngoặc thừa
        text = text[:-(close_c - open_c)]
        
    return text

def main():
    print(f"🔧 Đang clean file bằng Penman Validator...")
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    clean_lines = []
    error_count = 0
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line: continue
        
        # Bỏ qua dòng comment #
        if line.startswith("#"): continue

        try:
            # Thử parse chuẩn ngay lập tức
            g = penman.decode(line)
            # Re-encode để chuẩn hóa format (xóa khoảng trắng thừa)
            clean_line = penman.encode(g, indent=None)
            clean_lines.append(clean_line)
            
        except Exception as e:
            # Nếu lỗi, thử fix aggressive
            fixed_text = aggressive_fix(line)
            try:
                g = penman.decode(fixed_text)
                clean_line = penman.encode(g, indent=None)
                clean_lines.append(clean_line)
                # print(f"⚠️ Dòng {i+1}: Đã sửa lỗi syntax.")
            except Exception:
                # Vẫn lỗi -> Bỏ cuộc, điền graph rỗng
                error_count += 1
                print(f"❌ Dòng {i+1}: Không thể cứu chữa -> Thay bằng (a / amr-empty)")
                # In ra lỗi để debug nếu cần
                # print(f"   Content: {line}")
                clean_lines.append("(a / amr-empty)")

    print(f"📊 Tổng: {len(lines)} dòng. Lỗi không cứu được: {error_count}")
    print(f"✅ Đã lưu file sạch vào: {OUTPUT_FILE}")
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for l in clean_lines:
            f.write(l + "\n")

if __name__ == "__main__":
    main()