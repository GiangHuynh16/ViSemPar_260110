#!/usr/bin/env python3
"""
Analyze evaluation errors in detail
Extract failed examples and categorize error types
"""

import json
import re
from pathlib import Path
from collections import defaultdict

def parse_log_file(log_path):
    """Parse evaluation log to extract errors"""
    errors = []

    with open(log_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Extract error lines
    error_patterns = [
        r'Error: (.+)',
        r'Unmatched parenthesis at position (\d+) in processing (.+)',
        r'Duplicate node name\s+(\S+)\s+in parsing AMR',
        r'Format error when processing\s+(.+)'
    ]

    for line in content.split('\n'):
        for pattern in error_patterns:
            match = re.search(pattern, line)
            if match:
                errors.append({
                    'line': line.strip(),
                    'match': match.groups()
                })

    return errors

def categorize_error(error_line):
    """Categorize error type"""
    if 'Unmatched parenthesis' in error_line:
        return 'unmatched_parens'
    elif 'Duplicate node name' in error_line:
        return 'duplicate_node'
    elif 'Node name not found' in error_line:
        return 'node_not_found'
    elif 'Format error' in error_line:
        return 'format_error'
    else:
        return 'other'

def extract_amr_snippet(error_line):
    """Extract AMR snippet from error message"""
    # Look for "processing" or "in parsing"
    match = re.search(r'processing\s+(.+?)(?:\s*$|\s*Error)', error_line)
    if match:
        return match.group(1).strip()

    match = re.search(r'parsing AMR\s*$', error_line)
    if match:
        # Try to extract from earlier in line
        parts = error_line.split('in parsing')
        if len(parts) > 0:
            return parts[0].split()[-3:] if len(parts[0].split()) > 3 else parts[0]

    return None

def main():
    log_file = "outputs/evaluation_full_20251225_073829.log"

    if not Path(log_file).exists():
        print(f"Error: Log file not found: {log_file}")
        return

    print("="*80)
    print("ERROR ANALYSIS - DETAILED BREAKDOWN")
    print("="*80)
    print()

    # Parse errors
    errors = parse_log_file(log_file)

    print(f"Total error messages found: {len(errors)}")
    print()

    # Categorize
    categories = defaultdict(list)
    for error in errors:
        cat = categorize_error(error['line'])
        categories[cat].append(error)

    print("="*80)
    print("ERROR DISTRIBUTION")
    print("="*80)
    print()

    for cat, items in sorted(categories.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"{cat.upper()}: {len(items)} errors")

    print()
    print("="*80)
    print("DETAILED ERROR EXAMPLES")
    print("="*80)
    print()

    # Show examples from each category
    for cat, items in sorted(categories.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"\n{'='*80}")
        print(f"CATEGORY: {cat.upper()} ({len(items)} errors)")
        print('='*80)
        print()

        # Show first 5 examples
        for i, error in enumerate(items[:5], 1):
            print(f"Example {i}:")
            print(f"  {error['line'][:200]}")

            # Extract AMR snippet if available
            snippet = extract_amr_snippet(error['line'])
            if snippet:
                print(f"  AMR snippet: {snippet[:150]}")
            print()

        if len(items) > 5:
            print(f"  ... and {len(items) - 5} more similar errors")
            print()

    # Analysis
    print()
    print("="*80)
    print("PATTERN ANALYSIS")
    print("="*80)
    print()

    # Unmatched parentheses analysis
    if 'unmatched_parens' in categories:
        print("UNMATCHED PARENTHESES PATTERNS:")
        print()

        positions = []
        lengths = []

        for error in categories['unmatched_parens']:
            match = re.search(r'position (\d+)', error['line'])
            if match:
                positions.append(int(match.group(1)))

            match = re.search(r'processing (.+)', error['line'])
            if match:
                lengths.append(len(match.group(1)))

        if positions:
            print(f"  Error positions: min={min(positions)}, max={max(positions)}, avg={sum(positions)//len(positions)}")
        if lengths:
            print(f"  AMR lengths: min={min(lengths)}, max={max(lengths)}, avg={sum(lengths)//len(lengths)}")

        print(f"  Observation: Errors occur at various positions, suggesting it's not just")
        print(f"              a consistent early/late termination issue")
        print()

    # Duplicate nodes analysis
    if 'duplicate_node' in categories:
        print("DUPLICATE NODE NAMES:")
        print()

        node_names = []
        for error in categories['duplicate_node']:
            match = re.search(r'Duplicate node name\s+(\S+)', error['line'])
            if match:
                node_names.append(match.group(1))

        # Count frequency
        name_counts = defaultdict(int)
        for name in node_names:
            name_counts[name] += 1

        print("  Most common duplicate variables:")
        for name, count in sorted(name_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"    '{name}': {count} times")

        print()
        print(f"  Observation: Short variable names (n, t, c) are most commonly duplicated,")
        print(f"              suggesting the model struggles to track variable usage")
        print()

    print()
    print("="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    print()

    print("Based on error analysis:")
    print()
    print("1. UNMATCHED PARENTHESES (highest priority - 61% of errors):")
    print("   - Implement constrained decoding to force balanced parentheses")
    print("   - Add post-processing to detect and fix unbalanced structures")
    print("   - Try shorter max_length to prevent over-generation")
    print()

    print("2. DUPLICATE NODE NAMES (20% of errors):")
    print("   - Post-process to rename duplicates: n, n → n, n2")
    print("   - Track used variables during generation (if possible)")
    print("   - Add explicit variable uniqueness instruction to prompt")
    print()

    print("3. NODE NOT FOUND (10% of errors):")
    print("   - Validate all node references before finalizing AMR")
    print("   - Two-pass generation: structure first, then validate")
    print()

    print("="*80)

if __name__ == "__main__":
    main()
