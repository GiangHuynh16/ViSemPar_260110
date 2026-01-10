#!/usr/bin/env python3
"""
Create MTUP Unified Training Data from train_amr_12.txt

Input: train_amr_12.txt (Full AMR with variables)
Output: train_mtup_unified.txt (Unified prompt with Task 1 and Task 2)

Task 1: Remove variables from AMR (create skeleton)
Task 2: Keep original AMR (with variables)
"""

import os
import re
from pathlib import Path


def parse_amr_file(file_path):
    """
    Parse train_amr_12.txt format:

    #::snt <sentence>
    <AMR graph with variables - multiple lines>

    (blank line separator)
    """
    print(f"📂 Reading: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    samples = []
    blocks = content.strip().split('\n\n')

    for block in blocks:
        if not block.strip():
            continue

        lines = block.strip().split('\n')

        sentence = None
        amr_lines = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line.startswith('#::snt'):
                sentence = line.replace('#::snt', '').strip()
            else:
                amr_lines.append(line)

        if sentence and amr_lines:
            # Join multi-line AMR into single line
            full_amr = ' '.join(amr_lines)
            # Normalize spaces
            full_amr = re.sub(r'\s+', ' ', full_amr)

            samples.append({
                'sentence': sentence,
                'full_amr': full_amr.strip()
            })

    print(f"✅ Parsed {len(samples)} samples")
    return samples


def remove_variables_from_amr(amr_with_vars):
    """
    Remove variables from AMR to create skeleton (Task 1)

    Example:
    Input:  (b / bi_kịch :domain(c / chỗ :mod(đ / đó)))
    Output: (bi_kịch :domain(chỗ :mod(đó)))

    Rules:
    - Remove ALL "x / " patterns (variable definitions)
    - Remove standalone variable references after relations

    Algorithm:
    1. Remove (var / concept) -> (concept)
    2. Remove :relation var -> :relation (when var is standalone without /)
    """
    # Step 1: Remove variable definitions: (var / concept) -> (concept)
    # Use \w to match Unicode word characters (includes Vietnamese chars)
    no_vars = re.sub(r'\([\w]+\s*/\s*', r'(', amr_with_vars)

    # Step 2: Remove standalone variable references
    # Pattern: :relation followed by space and single-letter/number variable
    # Example: ":ARG0 t " -> ":ARG0 "
    # But be careful not to remove concepts like "tôi", only single vars like "t", "c", "đ"
    # Variables are typically 1-3 chars: t, c, đ, t2, c3, etc.

    # This is tricky because we need to distinguish between:
    # - Variable reference: :ARG0 t (should remove "t")
    # - Concept: :ARG0 tôi (should keep "tôi")

    # For now, we'll use a heuristic: if it's 1-3 chars and appears after a relation, it's likely a variable
    # But this is not perfect and may need refinement

    # Actually, a better approach: if there's no "/" after the token,
    # and it's short (<=3 chars), it's likely a variable reference

    # Remove pattern like ":ARG0 x " or ":ARG0 x)" where x is 1-3 chars
    no_vars = re.sub(r'(:[A-Z0-9-]+)\s+([a-z0-9_]{1,3})(?=[\s\)])', r'\1', no_vars)

    return no_vars.strip()


def create_unified_prompt(sample):
    """
    Create unified prompt for MTUP training.
    """
    # Create Task 1: AMR without variables
    task1_amr = remove_variables_from_amr(sample['full_amr'])

    # Task 2: Original AMR with variables
    task2_amr = sample['full_amr']

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

    user_input = f"Câu: {sample['sentence']}"

    assistant_output = f"Task 1: {task1_amr}\nTask 2: {task2_amr}"

    prompt = f"""<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{user_input}<|im_end|>
<|im_start|>assistant
{assistant_output}<|im_end|>"""

    return prompt


def validate_amr_brackets(amr_str):
    """Validate that AMR has balanced brackets"""
    open_count = amr_str.count('(')
    close_count = amr_str.count(')')
    return open_count == close_count


def validate_samples(samples):
    """Validate all samples before training"""
    print("\n🔍 Validating samples...")

    valid_samples = []
    errors = {
        'unbalanced_brackets': 0,
        'empty_amr': 0,
        'other': 0
    }

    for idx, sample in enumerate(samples):
        try:
            # Check full AMR
            if not sample['full_amr'] or len(sample['full_amr']) < 5:
                errors['empty_amr'] += 1
                continue

            if not validate_amr_brackets(sample['full_amr']):
                errors['unbalanced_brackets'] += 1
                print(f"  ❌ Sample {idx}: Unbalanced brackets in full AMR")
                continue

            # Create Task 1 and validate
            task1_amr = remove_variables_from_amr(sample['full_amr'])
            if not validate_amr_brackets(task1_amr):
                errors['unbalanced_brackets'] += 1
                print(f"  ❌ Sample {idx}: Unbalanced brackets in Task 1 AMR")
                continue

            valid_samples.append(sample)

        except Exception as e:
            errors['other'] += 1
            print(f"  ❌ Sample {idx}: {e}")

    print(f"\n✅ Valid samples: {len(valid_samples)}/{len(samples)}")
    if sum(errors.values()) > 0:
        print(f"Errors breakdown:")
        for error_type, count in errors.items():
            if count > 0:
                print(f"  - {error_type}: {count}")

    return valid_samples


def main():
    # Paths
    project_root = Path(__file__).parent.parent.parent
    input_file = project_root / 'data' / 'train_amr_12.txt'
    output_file = project_root / 'data' / 'train_mtup_unified.txt'

    print("=" * 70)
    print("MTUP UNIFIED DATA CREATION FROM train_amr_12.txt")
    print("=" * 70)

    # Parse input
    samples = parse_amr_file(input_file)

    if not samples:
        print("❌ No samples found!")
        return

    # Validate
    valid_samples = validate_samples(samples)

    if not valid_samples:
        print("❌ No valid samples after validation!")
        return

    # Create prompts and write output
    print(f"\n📝 Creating unified prompts...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for idx, sample in enumerate(valid_samples):
            prompt = create_unified_prompt(sample)
            f.write(prompt)
            if idx < len(valid_samples) - 1:
                f.write('\n\n')  # Separator between samples

    print(f"\n✅ Created {len(valid_samples)} training samples")
    print(f"📁 Output: {output_file}")

    # Show example
    print("\n" + "=" * 70)
    print("EXAMPLE (first sample):")
    print("=" * 70)
    example = create_unified_prompt(valid_samples[0])
    print(example)
    print("=" * 70)


if __name__ == '__main__':
    main()
