#!/usr/bin/env python3
"""
Create MTUP Unified Training Data

Merge Stage 1 and Stage 2 data into a single unified format for multi-task learning.
Input: train_amr_mtup_preprocessed.txt
Output: train_mtup_unified.txt

Format:
#::snt <sentence>
#::task1 <AMR no variables>
#::task2 <AMR with variables>
"""

import os
import re
from pathlib import Path


def parse_mtup_preprocessed(file_path):
    """
    Parse train_amr_mtup_preprocessed.txt format:

    #::snt <sentence>
    #::amr-no-vars <amr lines>
    #::amr-with-vars <amr lines>
    """
    print(f"📂 Reading {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    samples = []
    blocks = content.strip().split('\n\n')

    for block in blocks:
        if not block.strip():
            continue

        lines = block.strip().split('\n')

        sentence = None
        no_var_amr = []
        with_var_amr = []
        current_section = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line.startswith('#::snt'):
                sentence = line.replace('#::snt', '').strip()
                current_section = 'snt'
            elif line.startswith('#::amr-no-vars'):
                current_section = 'no-vars'
                # Check if AMR is on same line
                rest = line.replace('#::amr-no-vars', '').strip()
                if rest:
                    no_var_amr.append(rest)
            elif line.startswith('#::amr-with-vars'):
                current_section = 'with-vars'
                # Check if AMR is on same line
                rest = line.replace('#::amr-with-vars', '').strip()
                if rest:
                    with_var_amr.append(rest)
            else:
                # AMR content
                if current_section == 'no-vars':
                    no_var_amr.append(line)
                elif current_section == 'with-vars':
                    with_var_amr.append(line)

        # Validate and add
        if sentence and no_var_amr and with_var_amr:
            # Join multi-line AMR into single line
            no_var_str = ' '.join(no_var_amr).strip()
            with_var_str = ' '.join(with_var_amr).strip()

            # Clean up extra spaces
            no_var_str = re.sub(r'\s+', ' ', no_var_str)
            with_var_str = re.sub(r'\s+', ' ', with_var_str)

            samples.append({
                'sentence': sentence,
                'no_var_amr': no_var_str,
                'with_var_amr': with_var_str
            })
        else:
            print(f"⚠️  Skipped incomplete block: {sentence[:50] if sentence else 'No sentence'}")

    print(f"✅ Parsed {len(samples)} samples")
    return samples


def create_unified_prompt(sample):
    """
    Create unified prompt for MTUP training.

    Template:
    <|im_start|>system
    [System prompt]
    <|im_end|>
    <|im_start|>user
    Câu: {sentence}
    <|im_end|>
    <|im_start|>assistant
    Task 1: {no_var_amr}
    Task 2: {with_var_amr}
    <|im_end|>
    """

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

    assistant_output = f"Task 1: {sample['no_var_amr']}\nTask 2: {sample['with_var_amr']}"

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
            # Check Task 1
            if not sample['no_var_amr'] or len(sample['no_var_amr']) < 5:
                errors['empty_amr'] += 1
                continue

            if not validate_amr_brackets(sample['no_var_amr']):
                errors['unbalanced_brackets'] += 1
                print(f"  ❌ Sample {idx}: Task 1 unbalanced brackets")
                continue

            # Check Task 2
            if not sample['with_var_amr'] or len(sample['with_var_amr']) < 5:
                errors['empty_amr'] += 1
                continue

            if not validate_amr_brackets(sample['with_var_amr']):
                errors['unbalanced_brackets'] += 1
                print(f"  ❌ Sample {idx}: Task 2 unbalanced brackets")
                continue

            valid_samples.append(sample)

        except Exception as e:
            errors['other'] += 1
            print(f"  ❌ Sample {idx}: {e}")

    print(f"\n✅ Valid samples: {len(valid_samples)}/{len(samples)}")
    print(f"Errors breakdown:")
    for error_type, count in errors.items():
        if count > 0:
            print(f"  - {error_type}: {count}")

    return valid_samples


def main():
    # Paths
    project_root = Path(__file__).parent.parent.parent
    input_file = project_root / 'data' / 'train_amr_mtup_preprocessed.txt'
    output_file = project_root / 'data' / 'train_mtup_unified.txt'

    print("=" * 70)
    print("MTUP UNIFIED DATA CREATION")
    print("=" * 70)

    # Parse input
    samples = parse_mtup_preprocessed(input_file)

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
    print(example[:500] + "..." if len(example) > 500 else example)
    print("=" * 70)


if __name__ == '__main__':
    main()
