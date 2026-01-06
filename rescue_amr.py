import sys
import re

class NodeRenamer:
    """Class hỗ trợ đổi tên biến trùng lặp"""
    def __init__(self):
        self.seen_vars = {} # Lưu số lần xuất hiện của mỗi biến

    def replace_func(self, match):
        full_str = match.group(0) # VD: "(c /"
        var_name = match.group(1) # VD: "c"
        
        # Nếu chưa gặp biến này bao giờ -> giữ nguyên
        if var_name not in self.seen_vars:
            self.seen_vars[var_name] = 1
            return full_str
        
        # Nếu đã gặp -> tăng số đếm và đổi tên
        self.seen_vars[var_name] += 1
        count = self.seen_vars[var_name]
        new_var = f"{var_name}_{count}" # VD: c -> c_2
        
        # Thay thế tên biến trong chuỗi match (chỉ thay thế lần xuất hiện đầu tiên để giữ dấu ngoặc và gạch chéo)
        return full_str.replace(var_name, new_var, 1)

def fix_amr_syntax(amr_str):
    """Sửa các lỗi cú pháp AMR phổ biến"""
    
    # 1. Sửa lỗi Duplicate Node (Trùng tên biến)
    # Tìm tất cả các pattern dạng: ( biến /
    # Regex: \(\s*([a-z0-9\-_]+)\s*/
    renamer = NodeRenamer()
    # Dùng re.sub với callback function để xử lý từng match
    amr_str = re.sub(r'\(\s*([a-z0-9\-_]+)\s*/', renamer.replace_func, amr_str)
    
    return amr_str

def rescue_pipeline(pred_file, gold_file, output_file):
    print("🚑 Rescue Mission V2 (Deduplication) Started...")
    
    # 1. Đọc file Pred
    with open(pred_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Xóa xuống dòng thừa, đưa về 1 dòng dài rồi xử lý lại
    content = re.sub(r'\s+', ' ', content)
    
    # 2. Tách dòng dựa trên cấu trúc ) (
    content = re.sub(r'\)\s*\((\s*[a-z0-9]+\s*/)', r')\n(\1', content)
    
    raw_lines = content.split('\n')
    valid_graphs = []
    
    # 3. Xử lý từng dòng
    for line in raw_lines:
        line = line.strip()
        if not line: continue
        if not line.startswith('('): continue

        # Cân bằng ngoặc
        opens = line.count('(')
        closes = line.count(')')
        if opens > closes:
            line += ')' * (opens - closes)
        elif closes > opens:
            while closes > opens and line.endswith(')'):
                line = line[:-1]
                closes -= 1
        
        # --- FIX QUAN TRỌNG: Sửa trùng biến ---
        try:
            line = fix_amr_syntax(line)
        except Exception as e:
            print(f"   ⚠️ Warning: Could not fix syntax for line starting with {line[:20]}... Replacing with empty.")
            line = "(a / amr-empty)"
            
        valid_graphs.append(line)
        
    print(f"   -> Extracted and fixed {len(valid_graphs)} graphs.")

    # 4. Đọc Gold để align
    with open(gold_file, 'r', encoding='utf-8') as f:
        # Tách gold theo 1 hoặc nhiều dòng trống
        gold_blocks = [b for b in re.split(r'\n\s*\n', f.read()) if b.strip()]
        num_gold = len(gold_blocks)
        
    print(f"   -> Gold expects {num_gold} samples.")
    
    # 5. Padding
    final_graphs = valid_graphs
    if len(final_graphs) > num_gold:
        final_graphs = final_graphs[:num_gold]
    while len(final_graphs) < num_gold:
        final_graphs.append("(a / amr-empty)")
            
    # 6. Lưu file
    with open(output_file, 'w', encoding='utf-8') as f:
        for g in final_graphs:
            f.write(g + "\n")
            
    print(f"✅ Saved clean file to: {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python rescue_amr.py <pred_raw> <gold_file> <output_final>")
    else:
        rescue_pipeline(sys.argv[1], sys.argv[2], sys.argv[3])