import sys
import re

def fix_duplicate_variables(amr_str):
    """
    Sửa lỗi trùng biến (Duplicate node name).
    Ví dụ: (c / cat ... (c / car)) -> (c / cat ... (c_2 / car))
    """
    # Tìm tất cả định nghĩa biến: (x / concept
    matches = list(re.finditer(r'\(\s*([a-z0-9]+)\s*/', amr_str))
    
    seen_vars = {}
    new_amr = list(amr_str)
    
    # Duyệt ngược để không ảnh hưởng index khi thay thế string (mặc dù ở đây ta thay thế cùng độ dài hoặc dài hơn chút)
    # Cách đơn giản hơn: Thay thế tuyến tính và rebuild string
    
    mapping = {} # mapping vị trí cũ -> tên mới
    used_vars = set()
    
    # Quét lần 1 để xem biến nào bị trùng
    for m in matches:
        var = m.group(1)
        if var in used_vars:
            # Đây là biến trùng! Cần rename
            # Tạo tên mới: c -> c2, c3...
            suffix = 2
            new_var = f"{var}{suffix}"
            while new_var in used_vars:
                suffix += 1
                new_var = f"{var}{suffix}"
            
            used_vars.add(new_var)
            mapping[m.start(1)] = (var, new_var) # Lưu vị trí và tên mới
        else:
            used_vars.add(var)
            
    # Nếu không có gì trùng thì trả về luôn
    if not mapping:
        return amr_str
        
    # Rebuild string với tên biến mới
    # Lưu ý: Việc replace này chỉ đổi chỗ định nghĩa (x / ...). 
    # Còn chỗ tham chiếu :arg (x) thì script này chưa xử lý sâu (vì cần parse tree).
    # Tuy nhiên, để vượt qua lỗi Smatch crash thì chỉ cần sửa chỗ định nghĩa là được.
    
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

def smart_read_amr(file_path):
    """Đọc file và gom dòng dựa trên cân bằng ngoặc"""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    graphs = []
    current_graph = []
    balance = 0 # +1 cho '(', -1 cho ')'
    started = False
    
    for line in lines:
        line = line.strip()
        if not line: continue
        
        # Bỏ qua dòng comment
        if line.startswith("#"): continue
        
        current_graph.append(line)
        balance += line.count('(')
        balance -= line.count(')')
        
        if "(" in line: started = True
        
        # Nếu đã bắt đầu graph và ngoặc đã đóng hết -> Kết thúc 1 graph
        if started and balance == 0:
            full_graph_str = " ".join(current_graph)
            # Xử lý: Xóa khoảng trắng thừa
            full_graph_str = re.sub(r'\s+', ' ', full_graph_str)
            graphs.append(full_graph_str)
            
            # Reset
            current_graph = []
            started = False
            balance = 0
            
    return graphs

def process_pipeline(pred_file, gold_file, output_file):
    print("🛠️  Running Smart Fix...")
    
    # 1. Đọc Gold để lấy số lượng chuẩn
    gold_graphs = smart_read_amr(gold_file)
    num_gold = len(gold_graphs)
    print(f"   -> Gold lines: {num_gold}")
    
    # 2. Đọc Pred với logic cân bằng ngoặc
    pred_graphs = smart_read_amr(pred_file)
    print(f"   -> Pred lines (Merged): {len(pred_graphs)}")
    
    # 3. Align (Cảnh báo nếu vẫn lệch)
    final_graphs = []
    if len(pred_graphs) == num_gold:
        print("   ✅ Perfect alignment detected!")
        final_graphs = pred_graphs
    elif len(pred_graphs) > num_gold:
        print(f"   ⚠️ Still finding {len(pred_graphs)} graphs. Truncating last {len(pred_graphs)-num_gold}...")
        final_graphs = pred_graphs[:num_gold]
    else:
        print(f"   ⚠️ Missing graphs ({len(pred_graphs)}/{num_gold}). Padding with empty AMR...")
        final_graphs = pred_graphs
        while len(final_graphs) < num_gold:
            final_graphs.append("(a / amr-empty)")
            
    # 4. Fix Duplicate Variables và Lưu
    print("   🔧 Fixing duplicate variables...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for g in final_graphs:
            fixed_g = fix_duplicate_variables(g)
            f.write(fixed_g + "\n")
            
    print(f"✅ Saved fixed AMR to: {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python smart_fix_amr.py <pred_raw> <gold_file> <output_clean>")
    else:
        process_pipeline(sys.argv[1], sys.argv[2], sys.argv[3])