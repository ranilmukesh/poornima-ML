#!/usr/bin/env python3
"""
Script to fix indentation in stack_ridge.py to properly nest all processing 
code inside the dataset loop.
"""

def fix_indentation():
    with open('stack_ridge.py', 'r', encoding='utf-8') as f:
        lines = f.lines()
    
    # Find the start of the dataset loop
    loop_start = None
    for i, line in enumerate(lines):
        if 'for dataset_idx, (active_name, dataset_info) in enumerate(loaded.items()):' in line:
            loop_start = i
            break
    
    if loop_start is None:
        print("Could not find dataset loop!")
        return
    
    # Find where the loop body ends (need to add proper indentation)
    # The sections that need indenting start at "# ADVANCED FEATURE ENGINEERING"
    feature_eng_start = None
    for i in range(loop_start, len(lines)):
        if '# ADVANCED FEATURE ENGINEERING' in lines[i] and lines[i].strip().startswith('#'):
            feature_eng_start = i - 1  # The line with "# ===="
            break
    
    if feature_eng_start is None:
        print("Could not find feature engineering section!")
        return
    
    # Find the end of all processing (before the next major section or EOF)
    # Look for sections that should NOT be indented (e.g., final summary)
    processing_end = len(lines)
    
    # Add 4 spaces of indentation to all lines from feature_eng_start to processing_end
    # that currently start with 4 spaces (making them 8 spaces)
    fixed_lines = lines[:feature_eng_start]
    
    for i in range(feature_eng_start, processing_end):
        line = lines[i]
        # If line starts with exactly 4 spaces (not already indented to 8+)
        if line.startswith('    ') and not line.startswith('        '):
            # Add 4 more spaces
            fixed_lines.append('    ' + line)
        else:
            fixed_lines.append(line)
    
    # Write back
    with open('stack_ridge_fixed.py', 'w', encoding='utf-8') as f:
        f.writelines(fixed_lines)
    
    print("✅ Fixed indentation saved to stack_ridge_fixed.py")
    print("Please review the file and rename it if correct.")

if __name__ == '__main__':
    fix_indentation()
