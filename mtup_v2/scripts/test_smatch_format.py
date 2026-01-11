#!/usr/bin/env python3
"""
Test SMATCH Format - Verify output format matches SMATCH requirements
"""

import re

def extract_task2_from_model_output(text):
    """Extract Task 2 AMR from model output"""
    # Find "Task 2: (amr content)"
    match = re.search(r'Task 2:\s*(\(.+\))', text)
    if match:
        return match.group(1).strip()
    return None

def convert_inline_to_penman(inline_amr):
    """
    Convert inline AMR to PENMAN format with indentation
    Input:  (b / bi_kịch :domain(c / chỗ :mod(đ / đó)))
    Output: (b / bi_kịch
                :domain(c / chỗ
                    :mod(đ / đó)))
    """
    # This is a simplified version - you may need penman library for complex cases
    result = []
    indent = 0
    i = 0

    while i < len(inline_amr):
        char = inline_amr[i]

        if char == '(':
            if i > 0 and inline_amr[i-1] not in ['(', ' ', ':']:
                result.append('\n' + '    ' * indent)
            result.append(char)
            indent += 1
        elif char == ')':
            indent -= 1
            result.append(char)
        elif char == ':':
            # New relation - add newline
            result.append('\n' + '    ' * indent + char)
        else:
            result.append(char)

        i += 1

    return ''.join(result)

def test_format_conversion():
    """Test format conversion"""

    print("=" * 70)
    print("🧪 TESTING SMATCH FORMAT CONVERSION")
    print("=" * 70 + "\n")

    # Example model output
    model_output = "Task 1: (bi_kịch :domain(chỗ :mod(đó)))\nTask 2: (b / bi_kịch :domain(c / chỗ :mod(đ / đó)))"

    print("📝 Model Output:")
    print("-" * 70)
    print(model_output)
    print("-" * 70 + "\n")

    # Extract Task 2
    task2_inline = extract_task2_from_model_output(model_output)

    if task2_inline:
        print("✅ Extracted Task 2 (inline):")
        print(task2_inline)
        print()

        # Convert to PENMAN
        task2_penman = convert_inline_to_penman(task2_inline)

        print("✅ Converted to PENMAN format:")
        print("-" * 70)
        print(task2_penman)
        print("-" * 70 + "\n")

        # Expected ground truth format
        expected = """(b / bi_kịch
    :domain(c / chỗ
        :mod(đ / đó)))"""

        print("📋 Expected Ground Truth format:")
        print("-" * 70)
        print(expected)
        print("-" * 70 + "\n")

        print("✅ FORMAT TEST COMPLETED")
        print("\nNotes:")
        print("1. Model outputs inline format: (b / bi_kịch :domain(c / chỗ :mod(đ / đó)))")
        print("2. SMATCH needs PENMAN format with proper indentation")
        print("3. Need conversion step before SMATCH calculation")
        print("\nRecommendation:")
        print("• Use penman library: pip install penman")
        print("• Parse inline → Convert to PENMAN → Calculate SMATCH")

    else:
        print("❌ Could not extract Task 2 from model output")

def verify_penman_library():
    """Check if penman library is available"""
    print("\n" + "=" * 70)
    print("🔍 CHECKING PENMAN LIBRARY")
    print("=" * 70 + "\n")

    try:
        import penman
        print("✅ penman library is installed")
        print(f"   Version: {penman.__version__}")

        # Test parse
        inline_amr = "(b / bi_kịch :domain(c / chỗ :mod(đ / đó)))"
        graph = penman.decode(inline_amr)
        formatted = penman.encode(graph, indent=4)

        print("\n✅ Test conversion:")
        print("Input (inline):")
        print(f"  {inline_amr}")
        print("\nOutput (PENMAN):")
        print(formatted)

        return True

    except ImportError:
        print("❌ penman library NOT installed")
        print("\nInstall with:")
        print("  pip install penman")
        return False

if __name__ == "__main__":
    test_format_conversion()
    verify_penman_library()

    print("\n" + "=" * 70)
    print("📚 NEXT STEPS")
    print("=" * 70)
    print("\n1. Run quick training test:")
    print("   python mtup_v2/scripts/quick_training_test.py \\")
    print("     --data_path data/train_mtup_unified.txt \\")
    print("     --output_dir outputs/quick_test")
    print("\n2. Test output format with this script")
    print("\n3. If format is correct → Run full training")
    print("\n4. After training → Convert output to PENMAN → Calculate SMATCH")
    print("=" * 70)
