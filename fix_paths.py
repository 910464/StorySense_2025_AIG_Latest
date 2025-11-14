#!/usr/bin/env python3
"""
Script to fix test method names and file paths in test files
"""
import re

def fix_test_file(file_path):
    """Fix method names and file paths to match actual implementation"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix method names
    method_fixes = {
        '_process_pdf_file': '_process_pdf',
        '_process_txt_file': '_process_text',
        '_process_docx_file': '_process_word',
        '_process_xlsx_file': '_process_excel',
        '_process_pptx_file': '_process_presentation',
    }
    
    for old_name, new_name in method_fixes.items():
        content = content.replace(old_name, new_name)
    
    # Pattern to match file_path = '/path/to/file.ext'
    pattern = r"(\s+file_path\s*=\s*)'([^']+)'"
    
    # Replace with Path object
    def replace_with_path(match):
        indent = match.group(1)
        file_path_str = match.group(2)
        return f"{indent}Path('{file_path_str}')"
    
    content = re.sub(pattern, replace_with_path, content)
    
    # Also need to add Path import if not present
    if "from pathlib import Path" not in content:
        # Find existing imports section
        lines = content.split('\n')
        import_line_idx = 0
        for i, line in enumerate(lines):
            if line.startswith('from ') or line.startswith('import '):
                import_line_idx = i
        
        # Insert Path import after last import
        if import_line_idx > 0:
            lines.insert(import_line_idx + 1, "from pathlib import Path")
        else:
            lines.insert(0, "from pathlib import Path")
        
        content = '\n'.join(lines)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Fixed method names and file paths in {file_path}")

if __name__ == "__main__":
    test_file = "tests/context_handler/context_file_handler/test_enhanced_context_processor.py"
    fix_test_file(test_file)