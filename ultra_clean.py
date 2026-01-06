import sys
import re

def extract_graphs_by_brackets(text):
    """
    Tách các graph AMR dựa vào việc đếm ngoặc, bất chấp dòng.
    Xử lý trường hợp: (a / b)(c / d) dính liền nhau.
    """
    graphs = []
    current_graph = []
    balance = 0
    in_graph = False
    
    # Duyệt từng ký tự trong toàn bộ file
    for char in text:
        if char == '(':
            if balance == 0:
                in_graph = True # Bắt đầu 1 graph mới
            balance += 1
        
        if in_graph:
            current_graph.append(char)
            
        if char == ')':
            balance -= 1
            if balance == 0 and in_graph:
                # Kết thúc 1 graph
                graph_str = "".join(current_graph)
                # Dọn dẹp khoảng trắng thừa, đưa về 1 dòng
                clean_str = re.sub(r'\s+', ' ', graph_str).strip()
                if clean_str:
                    graphs.append(clean_str)
                current_graph = []
                in_graph = False
                
    return graphs

def fix_duplicate_variables(amr_str):
    """Sửa lỗi trùng biến (ví dụ: c lặp lại)"""
    # Logic: Tìm tất cả (var / concept...
    matches = list(re.finditer(r'\(\s*([a-z0-9]+)\s*/', amr_str))
    seen_vars = set()
    mapping = {} 
    
    # Quét để tìm biến trùng
    for m in matches:
        var = m.group(1)
        if var in seen_vars:
            # Tạo tên mới
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
        
    # Rebuild string
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

def main(pred_file, gold_file, output_file):
    print("🧹 Ultra Cleaning started...")
    
    # 1. Đọc toàn bộ file Pred thành 1 chuỗi khổng lồ
    with open(pred_file, 'r', encoding='utf-8') as f:
        pred_content = f.read()
    
    # 2. Tách graph bằng thuật toán đếm ngoặc
    pred_graphs = extract_graphs_by_brackets(pred_content)
    print(f"   -> Found {len(pred_graphs)} graphs in prediction.")

    # 3. Đọc Gold để lấy số lượng chuẩn (đếm block cách nhau bởi dòng trống)
    with open(gold_file, 'r', encoding='utf-8') as f:
        gold_content = f.read().strip()
        # Tách dựa trên dòng trống hoặc #::snt
        gold_blocks = [b for b in gold_content.split('\n\n') if b.strip()]
        num_gold = len(gold_blocks)
    print(f"   -> Gold standard has {num_gold} samples.")

    # 4. Align (Cắt hoặc Bù)
    final_graphs = []
    if len(pred_graphs) >= num_gold:
        final_graphs = pred_graphs[:num_gold]
    else:
        print(f"   ⚠️ Warning: Missing {num_gold - len(pred_graphs)} graphs. Padding with empty AMR.")
        final_graphs = pred_graphs
        while len(final_graphs) < num_gold:
            final_graphs.append("(a / amr-empty)")

    # 5. Fix lỗi trùng biến và lưu
    with open(output_file, 'w', encoding='utf-8') as f:
        for g in final_graphs:
            # Fix duplicate vars
            fixed_g = fix_duplicate_variables(g)
            f.write(fixed_g + "\n")
            
    print(f"✅ Saved clean file to: {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python ultra_clean.py <pred_file> <gold_file> <output_file>")
    else:
        main(sys.argv[1], sys.argv[2], sys.argv[3])