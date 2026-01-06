import argparse
import os

def merge_inputs(sent_file, skeleton_file, output_file):
    print("⏳ Merging inputs for Stage 2...")
    
    # 1. Đọc file câu gốc (Input Stage 1)
    with open(sent_file, 'r', encoding='utf-8') as f:
        raw_lines = [l.strip() for l in f.readlines() if l.strip()]
        
    sentences = []
    # Tự động kiểm tra format
    has_prefix = any(line.startswith("SENT:") or line.startswith("Input:") for line in raw_lines[:5])
    
    if has_prefix:
        print(f"   -> Detected format with prefixes in {sent_file}")
        for line in raw_lines:
            if line.startswith("SENT:"):
                sentences.append(line.replace("SENT:", "").strip())
            elif line.startswith("Input:"):
                sentences.append(line.replace("Input:", "").strip())
    else:
        print(f"   -> Detected RAW TEXT format in {sent_file}")
        sentences = raw_lines

    # 2. Đọc file Skeleton đã predict (Output Stage 1)
    with open(skeleton_file, 'r', encoding='utf-8') as f:
        skeletons = [line.strip() for line in f.readlines()]
        
    # 3. Kiểm tra số lượng
    print(f"📊 Stats: Sentences={len(sentences)} | Skeletons={len(skeletons)}")
    
    if len(sentences) == 0:
        print("❌ ERROR: No sentences found. Check input file path.")
        return

    # Nếu lệch dòng (thường do skeleton bị thiếu hoặc thừa dòng trống), lấy min
    min_len = min(len(sentences), len(skeletons))
    
    # 4. Ghép và lưu
    with open(output_file, 'w', encoding='utf-8') as f:
        for i in range(min_len):
            # Format chuẩn train Stage 2: Input <sep> Skeleton
            f.write(f"{sentences[i]} <sep> {skeletons[i]}\n")
            
    print(f"✅ Done! Saved {min_len} samples to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sent_file", type=str, required=True)
    parser.add_argument("--skeleton_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()
    
    merge_inputs(args.sent_file, args.skeleton_file, args.output_file)