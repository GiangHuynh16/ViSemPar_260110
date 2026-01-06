# merge_stage1.py
# Ghép câu gốc và output stage 1 để làm input cho stage 2
input_text = "data/public_test_input.txt"
input_concept = "evaluation_results/mtup/stage1_output_concepts.txt"
output_merged = "evaluation_results/mtup/stage2_input.txt"

with open(input_text, 'r') as f1, open(input_concept, 'r') as f2:
    texts = f1.readlines()
    concepts = f2.readlines()

assert len(texts) == len(concepts), f"Mismatch lines: {len(texts)} vs {len(concepts)}"

with open(output_merged, 'w') as f_out:
    for t, c in zip(texts, concepts):
        # Format này phải khớp với lúc bạn train Stage 2
        # Giả sử bạn train dạng: "Input: <text> <sep> <concept>"
        # Hoặc đơn giản là nối chuỗi.
        # Ở đây tôi nối bằng token <sep> hoặc xuống dòng tùy cách bạn train.
        # Dựa vào prompt code trên: user_input = f"Input: {text}"
        # Thì text này nên là sự kết hợp.
        
        combined = f"{t.strip()} <sep> {c.strip()}" 
        f_out.write(combined + "\n")

print("Merged data created for Stage 2!")