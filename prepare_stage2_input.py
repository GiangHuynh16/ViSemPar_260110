import os

# === CẤU HÌNH ĐƯỜNG DẪN ===
# 1. File chứa câu tiếng Việt gốc
INPUT_TEXT_FILE = "data/public_test.txt"

# 2. File kết quả vừa chạy ra từ Stage 1 (file chứa các graph (nghe...), (có...))
INPUT_STAGE1_FILE = "evaluation_results/mtup/stage1_output_concepts_v2.txt" 
# (Hãy đảm bảo bạn trỏ đúng file chứa cái đống (nghe...), (có...) mà bạn vừa paste)

# 3. File đầu ra cho Stage 2
OUTPUT_FILE = "evaluation_results/mtup/stage2_input_final.txt"

def main():
    print("🚀 Bắt đầu ghép dữ liệu cho Stage 2...")

    # --- BƯỚC 1: ĐỌC VÀ LỌC FILE TEXT ---
    print(f"1️⃣ Đọc file Text: {INPUT_TEXT_FILE}")
    clean_texts = []
    with open(INPUT_TEXT_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            # Chỉ lấy dòng có chữ, bỏ qua dòng trống hoàn toàn
            if line:
                clean_texts.append(line)
    
    print(f"   => Tìm thấy {len(clean_texts)} câu văn bản.")

    # --- BƯỚC 2: ĐỌC FILE STAGE 1 ---
    print(f"2️⃣ Đọc file Stage 1: {INPUT_STAGE1_FILE}")
    clean_graphs = []
    with open(INPUT_STAGE1_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line:
                clean_graphs.append(line)

    print(f"   => Tìm thấy {len(clean_graphs)} đồ thị concept.")

    # --- BƯỚC 3: KIỂM TRA KHỚP DÒNG ---
    if len(clean_texts) != len(clean_graphs):
        print(f"❌ LỖI LỆCH DÒNG: Text ({len(clean_texts)}) != Graph ({len(clean_graphs)})")
        # Cơ chế tự cắt nếu lệch ít (để code không crash)
        min_len = min(len(clean_texts), len(clean_graphs))
        print(f"⚠️ Đang tự động cắt cả 2 file về {min_len} dòng để tiếp tục...")
        clean_texts = clean_texts[:min_len]
        clean_graphs = clean_graphs[:min_len]
    else:
        print("✅ Số lượng dòng đã khớp hoàn hảo!")

    # --- BƯỚC 4: GHÉP VÀ GHI FILE ---
    print(f"3️⃣ Ghi file merged: {OUTPUT_FILE}")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        for text, graph in zip(clean_texts, clean_graphs):
            # Format ghép: Câu gốc <sep> Graph sơ khai
            # Đây là input để Stage 2 nhìn vào và điền biến
            combined_line = f"{text} <sep> {graph}"
            f_out.write(combined_line + "\n")

    print("🎉 HOÀN TẤT! Hãy dùng file này để chạy predict Stage 2.")

if __name__ == "__main__":
    main()