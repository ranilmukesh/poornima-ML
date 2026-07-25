#!/usr/bin/env python3
"""
Simple script to export all workspace files to a single Markdown file.
"""
import os
from pathlib import Path
from datetime import datetime

# Configuration
WORKSPACE_DIR = Path(__file__).parent
OUTPUT_FILE = WORKSPACE_DIR / "workspace_export.md"
EXCLUDE_DIRS = {'.git', '__pycache__', 'tmp', 'paper-files', 'node_modules', '.venv', 'venv', 'env'}
EXCLUDE_FILES = {'.env', '.gitignore', 'workspace_export.md', 'export_to_md.py'}
EXCLUDE_EXTS = {'.pkl', '.pkl', '.png', '.jpg', '.jpeg', '.gif', '.csv', '.pkl', '.bat', '.pkl'}

# File extensions to include as code blocks (only .js and .html)
CODE_EXTS = {'.js', '.html'}

def should_include(file_path: Path) -> bool:
    """Check if file should be included in export."""
    # Skip excluded directories
    for part in file_path.parts:
        if part in EXCLUDE_DIRS:
            return False
    
    # Skip excluded files
    if file_path.name in EXCLUDE_FILES:
        return False
    
    # Skip excluded extensions
    if file_path.suffix.lower() in EXCLUDE_EXTS:
        return False
    
    # Only include files with allowed extensions (.js and .html)
    if file_path.suffix.lower() not in CODE_EXTS:
        return False
    
    # Only include files (not directories)
    if not file_path.is_file():
        return False
    
    return True

def get_language(file_path: Path) -> str:
    """Get language identifier for code block."""
    ext = file_path.suffix.lower()
    lang_map = {
        '.py': 'python',
        '.js': 'javascript',
        '.ts': 'typescript',
        '.html': 'html',
        '.css': 'css',
        '.json': 'json',
        '.md': 'markdown',
        '.txt': 'text',
        '.yml': 'yaml',
        '.yaml': 'yaml',
        '.toml': 'toml',
        '.ini': 'ini',
        '.cfg': 'ini',
        '.conf': 'ini',
        '.sh': 'bash',
        '.bat': 'batch',
        '.ps1': 'powershell',
        '.sql': 'sql',
        '.xml': 'xml',
        '.csv': 'csv',
    }
    return lang_map.get(ext, 'text')

def read_file_content(file_path: Path) -> str:
    """Read file content with error handling."""
    try:
        return file_path.read_text(encoding='utf-8')
    except UnicodeDecodeError:
        try:
            return file_path.read_text(encoding='latin-1')
        except Exception as e:
            return f"[Error reading file: {e}]"
    except Exception as e:
        return f"[Error reading file: {e}]"

def main():
    print(f"Exporting workspace: {WORKSPACE_DIR}")
    print(f"Output file: {OUTPUT_FILE}")
    
    # Collect all files
    files = []
    for file_path in WORKSPACE_DIR.rglob('*'):
        if should_include(file_path):
            rel_path = file_path.relative_to(WORKSPACE_DIR)
            files.append((rel_path, file_path))
    
    # Sort by path
    files.sort(key=lambda x: str(x[0]))
    
    print(f"Found {len(files)} files to export")
    
    # Write markdown
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        # Header
        f.write(f"# Workspace Export\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Workspace:** {WORKSPACE_DIR}\n")
        f.write(f"**Files exported:** {len(files)}\n\n")
        f.write("---\n\n")
        
        # Table of contents
        f.write("## Table of Contents\n\n")
        for rel_path, _ in files:
            anchor = str(rel_path).replace('/', '-').replace('.', '-').replace(' ', '-').lower()
            f.write(f"- [{rel_path}](#{anchor})\n")
        f.write("\n---\n\n")
        
        # File contents
        for rel_path, abs_path in files:
            anchor = str(rel_path).replace('/', '-').replace('.', '-').replace(' ', '-').lower()
            f.write(f"## {rel_path} {{#{anchor}}}\n\n")
            
            lang = get_language(abs_path)
            content = read_file_content(abs_path)
            
            if lang != 'text' or abs_path.suffix.lower() in CODE_EXTS:
                f.write(f"```{lang}\n{content}\n```\n\n")
            else:
                f.write(f"{content}\n\n")
            
            f.write("---\n\n")
    
    print(f"Export complete: {OUTPUT_FILE}")
    print(f"File size: {OUTPUT_FILE.stat().st_size / 1024:.1f} KB")

if __name__ == '__main__':
    main()