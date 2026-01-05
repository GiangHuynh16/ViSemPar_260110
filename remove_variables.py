import re
import os

def remove_variables_robust_vietnamese(amr_string):
    """
    Loại bỏ biến (kể cả biến tiếng Việt như 'đ') và expand tree.
    VD: (n / nhớ :ARG0 (t / tôi) :ARG1 t) 
    -> (nhớ :ARG0 (tôi) :ARG1 tôi)
    """
    # 1. Tìm tất cả các cặp (var / concept)
    # Regex mới: [^\s/()]+ bắt tất cả ký tự không phải khoảng trắng, /, hay ngoặc
    # Điều này giúp bắt được biến là 'đ', 't1', 'x_y'...
    var_pattern = re.compile(r'\((?P<var>[^\s/()]+)\s*/\s*(?P<concept>[^:\)\s]+)')
    
    var_to_concept = {}
    for match in var_pattern.finditer(amr_string):
        var_to_concept[match.group('var')] = match.group('concept')

    # 2. Thay thế các biến tham chiếu (re-entrancies) bằng concept tương ứng
    # VD: thay thế ' t ' bằng ' tôi '
    processed_amr = amr_string
    
    # Sắp xếp biến theo độ dài giảm dần để tránh thay thế nhầm (vd: thay 't1' trước 't')
    sorted_vars = sorted(var_to_concept.keys(), key=len, reverse=True)
    
    for var in sorted_vars:
        concept = var_to_concept[var]
        # Regex này tìm biến 'var' đứng độc lập (bị bao quanh bởi khoảng trắng hoặc ngoặc)
        # Và KHÔNG được đứng trước dấu / (để tránh thay thế phần định nghĩa biến)
        # (?<!/) check phía trước ko phải dấu /
        # (?!\s*/) check phía sau ko phải dấu /
        pattern = re.compile(r'(?<!/)([\s\(\)])' + re.escape(var) + r'([\s\(\)])(?!\s*/)')
        
        # Thay thế bằng concept
        # Group 1 và 2 là các dấu ngoặc/khoảng trắng giữ nguyên
        processed_amr = pattern.sub(r'\1' + concept + r'\2', processed_amr)

    # 3. Xóa phần định nghĩa biến: (var / concept -> (concept
    # Thay thế "(var / " bằng "("
    processed_amr = re.sub(r'\([^\s/()]+\s*/\s*', '(', processed_amr)
    
    # Cleanup khoảng trắng thừa
    processed_amr = re.sub(r'\s+', ' ', processed_amr)
    return processed_amr

def combine_and_process(file1_path, file2_path, output_merged_path, stage1_out, stage2_out):
    print(">>> Đang gộp dữ liệu...")
    
    # Đọc và gộp file
    content_merged = ""
    
    if os.path.exists(file1_path):
        with open(file1_path, 'r', encoding='utf-8') as f:
            c1 = f.read().strip()
            content_merged += c1
            
    if os.path.exists(file2_path):
        with open(file2_path, 'r', encoding='utf-8') as f:
            c2 = f.read().strip()
            # Thêm xuống dòng trước khi nối file 2
            if content_merged:
                content_merged += "\n\n"
            content_merged += c2
    
    # Lưu file gộp để backup
    with open(output_merged_path, 'w', encoding='utf-8') as f:
        f.write(content_merged)
    print(f"Đã lưu file gộp tại: {output_merged_path}")

    # Xử lý tạo Stage 1 & 2
    print(">>> Đang xử lý biến và tạo dataset...")
    blocks = content_merged.split('\n\n')
    
    s1_data = []
    s2_data = []
    
    count = 0
    for block in blocks:
        lines = block.strip().split('\n')
        sentence = ""
        amr_lines = []
        
        for line in lines:
            if line.startswith('#::snt'):
                sentence = line.replace('#::snt', '').strip()
            elif not line.startswith('#') and line.strip():
                amr_lines.append(line.strip())
        
        if not sentence or not amr_lines:
            continue
            
        full_amr = ' '.join(amr_lines)
        
        # Xử lý xóa biến
        try:
            amr_no_vars = remove_variables_robust_vietnamese(full_amr)
        except Exception as e:
            print(f"Lỗi khi xử lý câu: {sentence}")
            continue

        # Format Text cho file training
        # Stage 1: Input -> AMR No Vars
        s1_data.append(f"Input: {sentence}\nOutput: {amr_no_vars}")
        
        # Stage 2: Input + AMR No Vars -> Full AMR
        s2_data.append(f"Input: {sentence} <sep> {amr_no_vars}\nOutput: {full_amr}")
        count += 1

    with open(stage1_out, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(s1_data))
        
    with open(stage2_out, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(s2_data))
        
    print(f"Hoàn tất! Đã xử lý {count} mẫu.")
    print(f"- Stage 1: {stage1_out}")
    print(f"- Stage 2: {stage2_out}")

if __name__ == "__main__":
    # Đường dẫn file của bạn
    base_dir = "data"
    f1 = os.path.join(base_dir, "train_amr_1.txt")
    f2 = os.path.join(base_dir, "train_amr_2.txt")
    
    # Output file
    f_merged = os.path.join(base_dir, "train_amr_12.txt")
    s1_out = os.path.join(base_dir, "train_stage1.txt")
    s2_out = os.path.join(base_dir, "train_stage2.txt")
    
    # Tạo folder data nếu chưa có
    os.makedirs(base_dir, exist_ok=True)
    
    combine_and_process(f1, f2, f_merged, s1_out, s2_out)