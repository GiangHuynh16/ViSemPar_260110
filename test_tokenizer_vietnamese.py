#!/usr/bin/env python3
"""Test if Qwen tokenizer handles Vietnamese correctly"""

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    trust_remote_code=True
)

# Test Vietnamese words
test_words = [
    "bi kịch là ở chỗ đó !",
    "bi_kịch",
    "chỗ",
    "đó",
    "(b / bi_kịch :domain(c / chỗ :mod(đ / đó)))"
]

print("=" * 70)
print("TESTING VIETNAMESE TOKENIZATION")
print("=" * 70 + "\n")

for text in test_words:
    # Encode
    token_ids = tokenizer.encode(text, add_special_tokens=False)

    # Decode
    decoded = tokenizer.decode(token_ids, skip_special_tokens=True)

    print(f"Original: {text}")
    print(f"Token IDs: {token_ids}")
    print(f"Decoded:  {decoded}")
    print(f"Match: {'✅' if decoded == text else '❌'}")
    print("-" * 70 + "\n")

# Test generation
print("=" * 70)
print("TESTING GENERATION")
print("=" * 70 + "\n")

test_prompt = """<|im_start|>system
Bạn là chuyên gia phân tích AMR cho tiếng Việt.<|im_end|>
<|im_start|>user
Câu: bi kịch là ở chỗ đó !<|im_end|>
<|im_start|>assistant
"""

inputs = tokenizer(test_prompt, return_tensors="pt")
print(f"Input text has {len(inputs['input_ids'][0])} tokens")
print(f"\nFirst 10 tokens decoded:")
for i in range(min(10, len(inputs['input_ids'][0]))):
    token_id = inputs['input_ids'][0][i].item()
    decoded = tokenizer.decode([token_id])
    print(f"  {i}: {token_id} -> '{decoded}'")

print("\n" + "=" * 70)
print("NOTE: If decoded text doesn't match original, tokenizer may not")
print("handle Vietnamese diacritics correctly!")
print("=" * 70)
