#!/usr/bin/env python3
"""
Fix duplicate node names in AMR output for SMATCH evaluation
"""

import sys
import re
from collections import Counter

def fix_duplicate_nodes(amr_text):
    """Rename duplicate node variables"""
    
    # Find all node declarations (variable / concept)
    pattern = r'\((\w+)\s*/\s*[\w_\-]+'
    nodes = re.findall(pattern, amr_text)
    
    # Find duplicates
    node_counts = Counter(nodes)
    duplicates = {node for node, count in node_counts.items() if count > 1}
    
    if not duplicates:
        return amr_text
    
    # Rename duplicates
    fixed_amr = amr_text
    seen = {}
    
    for dup in duplicates:
        # Find all occurrences
        matches = list(re.finditer(rf'\({dup}\s*/', fixed_amr))
        
        # Rename from second occurrence onwards
        for i, match in enumerate(matches[1:], start=2):
            new_name = f"{dup}{i}"
            # Replace this specific occurrence
            start, end = match.span()
            fixed_amr = (fixed_amr[:start+1] + new_name + 
                        fixed_amr[start+1+len(dup):])
    
    return fixed_amr

def process_file(input_file, output_file):
    """Process AMR file and fix duplicates"""
    
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by #::snt
    parts = content.split('#::snt ')
    
    fixed_parts = [parts[0]]  # Keep header if any
    fixed_count = 0
    
    for part in parts[1:]:
        lines = part.split('\n', 1)
        if len(lines) < 2:
            fixed_parts.append(part)
            continue
            
        sentence = lines[0]
        amr = lines[1] if len(lines) > 1 else ''
        
        # Extract just the AMR portion (before next #::snt or end)
        amr_only = amr.split('#::snt')[0].strip()
        
        # Fix duplicates
        fixed_amr = fix_duplicate_nodes(amr_only)
        
        if fixed_amr != amr_only:
            fixed_count += 1
        
        # Reconstruct
        fixed_parts.append(sentence + '\n' + fixed_amr + '\n')
    
    # Write output
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('#::snt '.join(fixed_parts))
    
    print(f"Fixed {fixed_count} AMRs with duplicate nodes")
    print(f"Output: {output_file}")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python fix_duplicate_nodes.py <input_file> <output_file>")
        sys.exit(1)
    
    process_file(sys.argv[1], sys.argv[2])
