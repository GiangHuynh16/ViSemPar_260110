#!/usr/bin/env python3
"""
Test extraction logic to see if it's cutting AMRs short
"""

def extract_amr_step2_OLD(generated_text: str) -> str:
    """OLD extraction logic - BUGGY VERSION"""
    # Remove EOS token
    if '<|im_end|>' in generated_text:
        generated_text = generated_text.split('<|im_end|>')[0]

    # Find Step 2 section
    if "AMR chuẩn PENMAN:" in generated_text:
        parts = generated_text.split("AMR chuẩn PENMAN:")
        amr_section = parts[-1]
    else:
        amr_section = generated_text

    # Extract until balanced parentheses
    lines = amr_section.split('\n')
    amr_lines = []

    for line in lines:
        # Skip empty lines at the start
        if not amr_lines and not line.strip():
            continue

        amr_lines.append(line)

        # Check balance on accumulated text
        accumulated = '\n'.join(amr_lines)
        open_count = accumulated.count('(')
        close_count = accumulated.count(')')

        if open_count == close_count > 0:
            break  # Found complete AMR - STOPS HERE!

    # OLD BUG: Only take first line
    result = '\n'.join(amr_lines).strip()
    result = result.split('\n')[0] if result else ""  # ← BUG!
    return result


def extract_amr_step2_NEW(generated_text: str) -> str:
    """NEW extraction logic - FIXED VERSION"""
    # Remove EOS token
    if '<|im_end|>' in generated_text:
        generated_text = generated_text.split('<|im_end|>')[0]

    # Find Step 2 section
    if "AMR chuẩn PENMAN:" in generated_text:
        parts = generated_text.split("AMR chuẩn PENMAN:")
        amr_section = parts[-1]
    else:
        amr_section = generated_text

    # Extract until balanced parentheses
    lines = amr_section.split('\n')
    amr_lines = []

    for line in lines:
        # Skip empty lines at the start
        if not amr_lines and not line.strip():
            continue

        amr_lines.append(line)

        # Check balance on accumulated text
        accumulated = '\n'.join(amr_lines)
        open_count = accumulated.count('(')
        close_count = accumulated.count(')')

        if open_count == close_count > 0:
            break  # Found complete AMR

    # FIXED: Return full multi-line AMR
    return '\n'.join(amr_lines).strip()


# Test cases
test_cases = [
    {
        "name": "Full AMR - Multi-line",
        "generated": """AMR chuẩn PENMAN:
(n / nhớ
:pivot(t / tôi)
:theme(l / lời
:poss(c / chủ_tịch
:mod(a / anh)
:mod(x / xã))))""",
        "expected": """(n / nhớ
:pivot(t / tôi)
:theme(l / lời
:poss(c / chủ_tịch
:mod(a / anh)
:mod(x / xã))))"""
    },
    {
        "name": "Short AMR",
        "generated": """AMR chuẩn PENMAN:
(b / bi_kịch
:domain(c / chỗ
:mod(đ / đó)))""",
        "expected": """(b / bi_kịch
:domain(c / chỗ
:mod(đ / đó)))"""
    },
    {
        "name": "Nested AMR",
        "generated": """AMR chuẩn PENMAN:
(c / contrast-01
:ARG1(q / quay
:frequency(n / năm)
:theme(h / hành_tinh
:mod(n1 / này))
:manner(n2 / nhanh
:degree(h1 / hơn)))
:ARG2(t1 / thay_đổi
:theme(đ / điều_lệnh)
:polarity -))""",
        "expected": """(c / contrast-01
:ARG1(q / quay
:frequency(n / năm)
:theme(h / hành_tinh
:mod(n1 / này))
:manner(n2 / nhanh
:degree(h1 / hơn)))
:ARG2(t1 / thay_đổi
:theme(đ / điều_lệnh)
:polarity -)"""
    }
]


def main():
    print("=" * 80)
    print("TESTING EXTRACTION LOGIC")
    print("=" * 80)
    print()

    for i, test in enumerate(test_cases, 1):
        print(f"Test {i}: {test['name']}")
        print("-" * 80)
        print()

        # Test OLD logic
        old_result = extract_amr_step2_OLD(test['generated'])

        # Test NEW logic
        new_result = extract_amr_step2_NEW(test['generated'])

        print("EXPECTED:")
        print(test['expected'])
        print()

        print("OLD EXTRACTION (BUGGY):")
        print(old_result)
        print()

        print("NEW EXTRACTION (FIXED):")
        print(new_result)
        print()

        # Analysis
        expected_lines = len(test['expected'].split('\n'))
        old_lines = len(old_result.split('\n'))
        new_lines = len(new_result.split('\n'))

        expected_rels = test['expected'].count(':')
        old_rels = old_result.count(':')
        new_rels = new_result.count(':')

        print("ANALYSIS:")
        print(f"  Expected: {expected_lines} lines, {expected_rels} relations")
        print(f"  OLD:      {old_lines} lines, {old_rels} relations",
              "❌ WRONG!" if old_lines < expected_lines else "")
        print(f"  NEW:      {new_lines} lines, {new_rels} relations",
              "✅ CORRECT!" if new_lines == expected_lines else "")
        print()

        if old_result == test['expected']:
            print("  OLD: ✅ Perfect match")
        else:
            print("  OLD: ❌ WRONG - Missing content!")

        if new_result == test['expected']:
            print("  NEW: ✅ Perfect match")
        else:
            print("  NEW: ⚠️  Different (but might be OK)")

        print()
        print("=" * 80)
        print()


if __name__ == '__main__':
    main()
