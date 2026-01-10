#!/usr/bin/env python3
"""
Fix data corruption in train_mtup_unified.txt

The file has encoding issues (UTF-8 displayed as Mojibake) and structural issues.
This script will regenerate the correct training data.

Usage:
    python fix_data_corruption.py
"""

import os
import sys

# Check if original data files exist
ORIGINAL_TRAIN = "data/train.txt"
ORIGINAL_TRAIN_AMR = "data/train.txt.amr"

if not os.path.exists(ORIGINAL_TRAIN) or not os.path.exists(ORIGINAL_TRAIN_AMR):
    print("❌ Error: Original training files not found!")
    print(f"   Looking for: {ORIGINAL_TRAIN} and {ORIGINAL_TRAIN_AMR}")
    print("\nPlease provide the original Vietnamese sentences and AMR files.")
    sys.exit(1)

print("🔧 FIXING DATA CORRUPTION")
print("=" * 70)

# Read original files
print(f"📂 Reading original files...")
with open(ORIGINAL_TRAIN, 'r', encoding='utf-8') as f:
    sentences = [line.strip() for line in f if line.strip()]

with open(ORIGINAL_TRAIN_AMR, 'r', encoding='utf-8') as f:
    amr_graphs = [line.strip() for line in f if line.strip()]

if len(sentences) != len(amr_graphs):
    print(f"❌ Error: Mismatch in number of sentences ({len(sentences)}) and AMR graphs ({len(amr_graphs)})")
    sys.exit(1)

print(f"✅ Found {len(sentences)} training samples")

# System prompt
system_prompt = """Bạn là chuyên gia phân tích AMR (Abstract Meaning Representation) cho tiếng Việt.
Nhiệm vụ: Với mỗi câu tiếng Việt, sinh ra 2 output:

Task 1 - AMR Skeleton: Cấu trúc AMR chỉ có concept và relation, KHÔNG có biến định danh.
Task 2 - Full AMR: AMR hoàn chỉnh với biến theo chuẩn PENMAN.

Quy tắc QUAN TRỌNG cho Task 2:
1. Mỗi concept định nghĩa biến MỘT lần: (t / tôi)
2. Tái sử dụng biến (co-reference): Nếu concept xuất hiện lại, CHỈ dùng tên biến, không viết lại concept
   VD: :ARG0 (t / tôi) ... :ARG1 t  (KHÔNG phải :ARG1 (t / tôi))
3. Biến dùng chữ cái đầu: (t / tôi), (b / bác_sĩ). Nếu trùng thì thêm số: (t2 / tôi)
4. Đảm bảo số ngoặc mở ( bằng số ngoặc đóng )"""


def extract_skeleton(full_amr):
    """
    Extract Task 1 (skeleton) from Task 2 (full AMR).
    Remove all variable definitions (x / concept) -> (concept)
    """
    import re

    # Remove variable definitions: (x / concept) -> (concept)
    skeleton = re.sub(r'\(([a-z0-9]+)\s*/\s*', '(', full_amr)

    # Remove variable references that are standalone (just the variable name)
    # But keep them if they're part of a relation
    # This is a simplification - real skeleton extraction is more complex

    return skeleton


def generate_unified_sample(sentence, full_amr):
    """Generate a unified training sample with Task 1 and Task 2"""

    skeleton = extract_skeleton(full_amr)

    sample = f"""<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
Câu: {sentence}<|im_end|>
<|im_start|>assistant
Task 1: {skeleton}
Task 2: {full_amr}<|im_end|>"""

    return sample


# Generate all samples
print("🔄 Generating unified training samples...")
samples = []
errors = 0

for i, (sentence, amr) in enumerate(zip(sentences, amr_graphs)):
    try:
        sample = generate_unified_sample(sentence, amr)
        samples.append(sample)
    except Exception as e:
        errors += 1
        print(f"⚠️  Error at sample {i}: {e}")

print(f"✅ Generated {len(samples)} samples ({errors} errors)")

# Write to output file
output_file = "data/train_mtup_unified_fixed.txt"
print(f"\n💾 Writing to: {output_file}")

with open(output_file, 'w', encoding='utf-8') as f:
    f.write('\n\n'.join(samples))

print(f"✅ Done! File saved with UTF-8 encoding")

# Verify first sample
print("\n" + "=" * 70)
print("FIRST SAMPLE (VERIFICATION):")
print("=" * 70)
print(samples[0][:500] + "...")
print("=" * 70)

print("\n✅ DATA FIXED!")
print(f"   Original (corrupted): data/train_mtup_unified.txt")
print(f"   Fixed version: {output_file}")
print("\nNext steps:")
print("1. Verify the fixed file looks correct")
print("2. Replace the corrupted file:")
print("   mv data/train_mtup_unified_fixed.txt data/train_mtup_unified.txt")
print("3. Re-run diagnosis to confirm:")
print("   python mtup_v2/scripts/diagnose_model.py --data_path data/train_mtup_unified.txt --adapter_path dummy")
