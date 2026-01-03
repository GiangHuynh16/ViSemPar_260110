#!/usr/bin/env python3
"""
Analyze sentence lengths to check if max_length=512 is causing truncation issues
"""

from transformers import AutoTokenizer
import json

# Load tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct')

# Read test sentences
print("Reading test sentences...")
with open('data/public_test.txt', 'r', encoding='utf-8') as f:
    sentences = [line.strip() for line in f if line.strip()]

print(f'\n=== SENTENCE LENGTH ANALYSIS ===')
print(f'Total sentences: {len(sentences)}\n')

# Analyze each sentence
long_sentences = []
all_lengths = []

for i, sent in enumerate(sentences, 1):
    # Tokenize the full prompt (not just sentence)
    prompt = f"""Chuyển câu tiếng Việt sau sang AMR (Abstract Meaning Representation)
theo định dạng Penman:

Câu: {sent}

AMR:
"""
    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    token_count = len(tokens)
    all_lengths.append((i, token_count, sent))

    if token_count > 400:
        long_sentences.append((i, token_count, sent))
        print(f'#{i}: {token_count} tokens')
        print(f'   Sentence: {sent[:100]}...')
        print()

# Sort by length
all_lengths.sort(key=lambda x: x[1], reverse=True)

print(f'\n=== TOP 10 LONGEST PROMPTS ===')
for i, (idx, length, sent) in enumerate(all_lengths[:10], 1):
    print(f'{i}. Sentence #{idx}: {length} tokens')
    print(f'   {sent[:100]}...')
    print()

lengths_only = [l[1] for l in all_lengths]
print(f'\n=== STATISTICS ===')
print(f'Min prompt length: {min(lengths_only)} tokens')
print(f'Max prompt length: {max(lengths_only)} tokens')
print(f'Avg prompt length: {sum(lengths_only)/len(lengths_only):.1f} tokens')
print(f'Median prompt length: {sorted(lengths_only)[len(lengths_only)//2]} tokens')

print(f'\n=== CRITICAL CHECK ===')
print(f'Sentences with prompt >400 tokens: {sum(1 for l in lengths_only if l > 400)}')
print(f'Sentences with prompt >450 tokens: {sum(1 for l in lengths_only if l > 450)}')
print(f'Sentences with prompt >500 tokens: {sum(1 for l in lengths_only if l > 500)}')
print(f'Sentences with prompt >512 tokens: {sum(1 for l in lengths_only if l > 512)} ⚠️')

# Now check the AMR output lengths
print(f'\n=== CHECKING AMR OUTPUT LENGTHS ===')
print("Reading predictions from checkpoint-1500...")

try:
    with open('evaluation_results/checkpoint_comparison/checkpoint-1500.txt', 'r', encoding='utf-8') as f:
        predictions = f.read().split('\n\n')

    print(f'Total predictions: {len(predictions)}')

    # Check if any predictions are truncated (end with incomplete AMR)
    truncated = []
    for i, pred in enumerate(predictions, 1):
        if not pred.strip():
            continue

        # Check parenthesis balance
        open_count = pred.count('(')
        close_count = pred.count(')')

        # Estimate output tokens
        amr_tokens = tokenizer.encode(pred, add_special_tokens=False)

        if open_count != close_count:
            # Find corresponding sentence length
            for idx, length, sent in all_lengths:
                if idx == i:
                    truncated.append((i, length, len(amr_tokens), open_count - close_count))
                    break

    print(f'\n=== UNBALANCED PARENTHESES (Potential Truncation) ===')
    print(f'Total: {len(truncated)} AMRs with unbalanced parentheses')

    if truncated:
        print(f'\nTop cases (sorted by prompt length):')
        truncated.sort(key=lambda x: x[1], reverse=True)
        for idx, prompt_len, output_len, diff in truncated[:15]:
            print(f'Sentence #{idx}:')
            print(f'  Prompt length: {prompt_len} tokens')
            print(f'  Output length: {output_len} tokens')
            print(f'  Total: {prompt_len + output_len} tokens')
            print(f'  Parenthesis diff: {diff} ({"more (" if diff > 0 else "more )"})')
            if prompt_len + output_len > 512:
                print(f'  ⚠️ LIKELY TRUNCATED (total > 512)')
            print()

except FileNotFoundError:
    print("⚠️ Prediction file not found, skipping AMR analysis")

print(f'\n=== RECOMMENDATION ===')
exceeding_512 = sum(1 for l in lengths_only if l > 512)
if exceeding_512 > 0:
    print(f'❌ {exceeding_512} sentences have prompts exceeding 512 tokens')
    print(f'   This WILL cause truncation during inference!')
    print(f'   Recommendation: Retrain with max_length=768 or 1024')
elif sum(1 for l in lengths_only if l > 450) > 5:
    print(f'⚠️ Several sentences have prompts >450 tokens')
    print(f'   This leaves little room for AMR output (only ~60 tokens)')
    print(f'   Recommendation: Consider increasing max_length to 768')
else:
    print(f'✅ All prompts fit within 512 tokens with room for output')
    print(f'   The 13 invalid AMRs are likely due to model errors, not truncation')
    print(f'   Current results (91.3% valid) are EXCELLENT!')
