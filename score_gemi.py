import sys
import re
import os
import subprocess

try:
    import penman
except ImportError:
    print("❌ Chưa cài penman -> pip install penman")
    sys.exit(1)

# ================= CONFIG =================
RAW_FILE = "evaluation_results/mtup_v2/pred_final_raw.txt"
CLEAN_FILE = "evaluation_results/mtup_v2/final_submission.txt"
GOLD_FILE = "data/public_test_ground_truth.txt"

# ================= CLEANING LOGIC (NUCLEAR) =================
def nuclear_fix(line):
    line = line.strip()
    if not line or line.startswith("#"): return "(e / amr-empty)"

    # 1. Fix Syntax cơ bản
    line = re.sub(r':\s+([a-zA-Z0-9-]+)', r':\1', line)
    line = re.sub(r':[a-z0-9-]+\s*[({]?\s*$', '', line)
    
    # Cân bằng ngoặc
    open_c = line.count('(')
    close_c = line.count(')')
    if open_c > close_c: line += ')' * (open_c - close_c)
    elif close_c > open_c: line = line[:-(close_c - open_c)]

    # 2. Xử lý Duplicate Node (Thêm số đếm vào biến trùng)
    # Regex: Tìm (biến /
    pattern = re.compile(r'\(\s*([^\s/()]+)\s*/')
    seen_vars = {} 
    new_string = ""
    last_end = 0
    
    try:
        for match in pattern.finditer(line):
            start, end = match.span()
            var_name = match.group(1)
            new_string += line[last_end:start]
            
            if var_name in seen_vars:
                seen_vars[var_name] += 1
                new_var = f"{var_name}_{seen_vars[var_name]}"
            else:
                seen_vars[var_name] = 1
                new_var = var_name
                
            new_string += f"({new_var} /"
            last_end = end
        new_string += line[last_end:]
        
        # 3. PENMAN VALIDATION
        g = penman.decode(new_string)
        return penman.encode(g, indent=None) # Format chuẩn 1 dòng
        
    except Exception:
        # Nếu vẫn lỗi -> Trả về rỗng để cứu Smatch
        return "(e / amr-empty)"

def main():
    print("🧹 Cleaning & Fixing AMR Output...")
    with open(RAW_FILE, 'r') as f:
        lines = f.readlines()
        
    clean_lines = []
    error_count = 0
    for line in lines:
        fixed = nuclear_fix(line)
        if fixed == "(e / amr-empty)": error_count += 1
        clean_lines.append(fixed)
        
    with open(CLEAN_FILE, 'w') as f:
        for line in clean_lines:
            f.write(line + "\n")
            
    print(f"📊 Fixed {len(lines)} lines. Total broken/empty lines: {error_count}")
    print(f"💾 Saved clean file to: {CLEAN_FILE}")
    
    # ================= SMATCH CALCULATION =================
    print("\n🏆 CALCULATING SMATCH SCORE...")
    cmd = [
        sys.executable, "-m", "smatch", "--pr",
        "-f", CLEAN_FILE, GOLD_FILE,
        "--significant", "3"
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Smatch failed: {e}")

if __name__ == "__main__":
    main()