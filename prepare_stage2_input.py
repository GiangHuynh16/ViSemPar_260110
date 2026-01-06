import argparse

def merge_inputs(sent_file, skeleton_file, output_file):
    print("⏳ Merging inputs for Stage 2...")
    
    # 1. Đọc file câu gốc (Input Stage 1)
    with open(sent_file, 'r', encoding='utf-8') as f:
        # Lọc lấy câu gốc sạch sẽ
        lines = f.readlines()
        sentences = []
        for line in lines:
            if line.startswith("SENT:") or line.startswith("Input:"):
                clean = line.replace("SENT:", "").replace("Input:", "").strip()
                sentences.append(clean)
            # Nếu file chỉ toàn text trơn thì bỏ if, lấy luôn line.strip()
    
    # 2. Đọc file Skeleton đã predict (Output Stage 1)
    with open(skeleton_file, 'r', encoding='utf-8') as f:
        skeletons = [line.strip() for line in f.readlines()]
        
    # Kiểm tra số lượng dòng
    if len(sentences) != len(skeletons):
        print(f"⚠️ WARNING: Mismatch lengths! Sentences: {len(sentences)}, Skeletons: {len(skeletons)}")
        # Có thể dừng hoặc cắt bớt tùy bạn, ở đây ta lấy min
        min_len = min(len(sentences), len(skeletons))
        sentences = sentences[:min_len]
        skeletons = skeletons[:min_len]

    # 3. Ghép và lưu
    with open(output_file, 'w', encoding='utf-8') as f:
        for sent, skel in zip(sentences, skeletons):
            # Format đúng chuẩn train Stage 2: Input <sep> Skeleton
            f.write(f"{sent} <sep> {skel}\n")
            
    print(f"✅ Done! Saved {len(sentences)} samples to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sent_file", type=str, required=True, help="File chứa câu gốc (Test set)")
    parser.add_argument("--skeleton_file", type=str, required=True, help="File output từ Stage 1")
    parser.add_argument("--output_file", type=str, required=True, help="File input cho Stage 2")
    args = parser.parse_args()
    
    merge_inputs(args.sent_file, args.skeleton_file, args.output_file)