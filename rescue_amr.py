import sys
import re

def fix_duplicate_variables(amr_str):
    """Sửa lỗi trùng biến trong cùng 1 graph"""
    matches = list(re.finditer(r'\(\s*([a-z0-9]+)\s*/', amr_str))
    seen_vars = set()
    mapping = {}
    
    for m in matches:
        var = m.group(1)
        if var in seen_vars:
            suffix = 2
            new_var = f"{var}{suffix}"
            while new_var in seen_vars:
                suffix += 1
                new_var = f"{var}{suffix}"
            seen_vars.add(new_var)
            mapping[m.start(1)] = (var, new_var)
        else:
            seen_vars.add(var)
            
    if not mapping: return amr_str
        
    result = ""
    last_idx = 0
    for match in matches:
        start_idx = match.start(1)
        end_idx = match.end(1)
        result += amr_str[last_idx:start_idx]
        if start_idx in mapping:
            old_var, new_var = mapping[start_idx]
            result += new_var
        else:
            result += match.group(1)
        last_idx = end_idx
    result += amr_str[last_idx:]
    return result

def rescue_pipeline(pred_file, gold_file, output_file):
    print("🚑 Rescue Mission Started...")
    
    # 1. Đọc toàn bộ file Pred thành 1 cục văn bản
    with open(pred_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Xóa sạch xuống dòng cũ để xử lý từ đầu
    content = re.sub(r'\s+', ' ', content)
    
    # 2. MAGIC SPLIT: Tìm điểm giao nhau giữa ')' và '(' bắt đầu một biến mới
    # Regex này tìm: Dấu đóng ngoặc ) -> Khoảng trắng -> Dấu mở ngoặc ( -> Tên biến -> Dấu /
    # Ví dụ: ...kết thúc). (c / có... -> Sẽ bị tách ở giữa.
    # Logic này an toàn hơn đếm ngoặc vì nó dựa vào cấu trúc biến.
    
    # Thay thế bằng: ) \n (
    content = re.sub(r'\)\s*\((\s*[a-z0-9]+\s*/)', r')\n(\1', content)
    
    # Tách thành list các dòng
    raw_lines = content.split('\n')
    valid_graphs = []
    
    # 3. Cân bằng ngoặc cho từng dòng (Force Close)
    for line in raw_lines:
        line = line.strip()
        if not line: continue
        
        # Chỉ lấy dòng bắt đầu bằng (
        if not line.startswith('('): continue

        # Cân bằng ngoặc
        opens = line.count('(')
        closes = line.count(')')
        
        if opens > closes:
            # Thiếu ngoặc đóng -> Thêm vào
            line += ')' * (opens - closes)
        elif closes > opens:
            # Thừa ngoặc đóng (do cắt sai) -> Xóa bớt ở đuôi
            while closes > opens and line.endswith(')'):
                line = line[:-1]
                closes -= 1
        
        # Sửa lỗi trùng biến luôn
        line = fix_duplicate_variables(line)
        valid_graphs.append(line)
        
    print(f"   -> Extracted {len(valid_graphs)} graphs.")

    # 4. Đọc Gold để align
    with open(gold_file, 'r', encoding='utf-8') as f:
        gold_blocks = [b for b in f.read().split('\n\n') if b.strip()]
        num_gold = len(gold_blocks)
        
    print(f"   -> Gold expects {num_gold} samples.")
    
    # 5. Padding hoặc Truncate
    final_graphs = []
    if len(valid_graphs) >= num_gold:
        final_graphs = valid_graphs[:num_gold]
    else:
        print(f"   ⚠️ Still missing {num_gold - len(valid_graphs)}. Padding...")
        final_graphs = valid_graphs
        while len(final_graphs) < num_gold:
            final_graphs.append("(a / amr-empty)")
            
    # 6. Lưu file
    with open(output_file, 'w', encoding='utf-8') as f:
        for g in final_graphs:
            f.write(g + "\n")
            
    print(f"✅ Saved rescued file to: {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python rescue_amr.py <pred_raw> <gold_file> <output_final>")
    else:
        rescue_pipeline(sys.argv[1], sys.argv[2], sys.argv[3])