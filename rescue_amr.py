import sys
import re

class GraphDeduplicator:
    """Class giúp đổi tên các biến bị trùng lặp trong cùng 1 graph"""
    def __init__(self):
        self.seen_vars = {} # Lưu số lần xuất hiện: {'c': 1, 'n': 2...}

    def replace_duplicate(self, match):
        full_str = match.group(0) # VD: "(c /"
        var_name = match.group(1) # VD: "c"
        
        # Nếu chưa gặp biến này trong câu này -> giữ nguyên, đánh dấu đã gặp
        if var_name not in self.seen_vars:
            self.seen_vars[var_name] = 1
            return full_str
        
        # Nếu đã gặp -> Tăng số đếm và đổi tên biến mới
        self.seen_vars[var_name] += 1
        count = self.seen_vars[var_name]
        new_var = f"{var_name}_{count}" # VD: c -> c_2
        
        # Thay thế tên biến cũ bằng tên biến mới (chỉ thay 1 lần ở vị trí này)
        # full_str là "(c /" -> thay thành "(c_2 /"
        return full_str.replace(var_name, new_var, 1)

def clean_and_fix_amr(amr_str):
    """Sửa lỗi cú pháp AMR: trùng biến và ngoặc"""
    
    # 1. Khởi tạo bộ deduplicator cho dòng này
    deduplicator = GraphDeduplicator()
    
    # 2. Tìm tất cả các định nghĩa biến: (biến / concept
    # Regex tìm: Dấu mở ngoặc ( -> khoảng trắng -> Tên biến -> khoảng trắng -> Dấu /
    pattern = r'\(\s*([a-zA-Z0-9\-_]+)\s*/'
    
    # Thay thế bằng hàm callback để đổi tên nếu trùng
    cleaned_str = re.sub(pattern, deduplicator.replace_duplicate, amr_str)
    
    return cleaned_str

def rescue_pipeline(pred_file, gold_file, output_file):
    print("🚑 Rescue Mission V2 (Syntax Repair) Started...")
    
    # --- GIAI ĐOẠN 1: Tách dòng bị dính ---
    with open(pred_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Xóa xuống dòng thừa, đưa về 1 dòng dài
    content = re.sub(r'\s+', ' ', content)
    
    # Tách dòng dựa trên cấu trúc kết thúc ')' và bắt đầu '(' mới
    # Thêm \n vào giữa
    content = re.sub(r'\)\s*\((\s*[a-z0-9]+\s*/)', r')\n(\1', content)
    
    raw_lines = content.split('\n')
    valid_graphs = []
    
    print(f"   -> Raw split found {len(raw_lines)} potential lines.")

    # --- GIAI ĐOẠN 2: Sửa lỗi từng dòng ---
    for line in raw_lines:
        line = line.strip()
        if not line: continue
        if not line.startswith('('): continue

        # 1. Cân bằng ngoặc (Bracket Balancing)
        opens = line.count('(')
        closes = line.count(')')
        if opens > closes:
            line += ')' * (opens - closes)
        elif closes > opens:
            # Cắt bớt ngoặc đóng thừa ở cuối
            while closes > opens and line.endswith(')'):
                line = line[:-1]
                closes -= 1
        
        # 2. Sửa lỗi trùng tên biến (Duplicate Node Fix)
        try:
            line = clean_and_fix_amr(line)
        except Exception as e:
            print(f"   ⚠️ Error fixing line: {line[:30]}... -> Using empty graph.")
            line = "(a / amr-empty)"
            
        valid_graphs.append(line)
        
    print(f"   -> Successfully repaired {len(valid_graphs)} graphs.")

    # --- GIAI ĐOẠN 3: Align với Gold Standard ---
    with open(gold_file, 'r', encoding='utf-8') as f:
        # Tách gold theo dòng trống (paragraph split)
        gold_blocks = [b for b in re.split(r'\n\s*\n', f.read()) if b.strip()]
        num_gold = len(gold_blocks)
        
    print(f"   -> Gold Standard has {num_gold} samples.")
    
    # Padding (lấp đầy) hoặc Cắt bớt (truncate)
    final_graphs = valid_graphs
    
    # Nếu thừa (do code tách dòng quá nhạy), cắt bớt cho bằng gold
    if len(final_graphs) > num_gold:
        final_graphs = final_graphs[:num_gold]
        
    # Nếu thiếu, điền rỗng
    while len(final_graphs) < num_gold:
        final_graphs.append("(a / amr-empty)")
            
    # Lưu file
    with open(output_file, 'w', encoding='utf-8') as f:
        for g in final_graphs:
            f.write(g + "\n")
            
    print(f"✅ Saved CLEAN & VALID file to: {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python rescue_amr_v2.py <pred_raw> <gold_file> <output_final>")
    else:
        rescue_pipeline(sys.argv[1], sys.argv[2], sys.argv[3])