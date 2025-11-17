#!/usr/bin/env python3
"""
Standalone test for inline editing functionality
Tests core methods without requiring RAG dependencies
"""

import sys
import tempfile
import shutil
import difflib
import ast
from pathlib import Path

def backup_file(file_path: Path, backup_dir: Path) -> Path:
    """Create a backup of a file before editing."""
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"{file_path.stem}_{timestamp}{file_path.suffix}"
    backup_path = backup_dir / backup_name
    shutil.copy2(file_path, backup_path)
    return backup_path

def create_diff(old_content: str, new_content: str, filename: str = "file") -> str:
    """Create a unified diff between old and new content."""
    old_lines = old_content.splitlines(keepends=True)
    new_lines = new_content.splitlines(keepends=True)
    
    diff = difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile=f"--- {filename}",
        tofile=f"+++ {filename}",
        lineterm='',
        n=3
    )
    
    return ''.join(diff)

def find_function_or_class(file_path: Path, name: str):
    """Find a function or class by name in a Python file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    tree = ast.parse(content)
    
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name == name:
                code = ast.get_source_segment(content, node)
                return {
                    'type': 'function' if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else 'class',
                    'name': node.name,
                    'start_line': node.lineno,
                    'end_line': node.end_lineno if hasattr(node, 'end_lineno') else node.lineno,
                    'code': code or '',
                    'full_content': content
                }
    
    return None

def test_backup():
    """Test backup functionality."""
    print("Testing backup functionality...")
    
    backup_dir = Path(".test_backups")
    backup_dir.mkdir(exist_ok=True)
    
    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write("def test():\n    pass\n")
        temp_file = Path(f.name)
    
    try:
        backup_path = backup_file(temp_file, backup_dir)
        assert backup_path.exists(), "Backup file should exist"
        print(f"✅ Backup created: {backup_path.name}")
        
        # Verify content matches
        with open(temp_file) as f1, open(backup_path) as f2:
            assert f1.read() == f2.read(), "Backup content should match original"
        print("✅ Backup content verified")
        
        # Cleanup
        backup_path.unlink()
    finally:
        temp_file.unlink()
        if backup_dir.exists():
            shutil.rmtree(backup_dir, ignore_errors=True)

def test_diff():
    """Test diff creation."""
    print("\nTesting diff creation...")
    
    old_content = "def test():\n    return 1\n"
    new_content = "def test():\n    return 2\n"
    
    diff = create_diff(old_content, new_content, "test.py")
    
    assert "--- test.py" in diff, "Diff should contain file name"
    assert "+++ test.py" in diff, "Diff should contain file name"
    assert "-    return 1" in diff or "+    return 2" in diff, "Diff should show changes"
    print("✅ Diff creation works")
    print(f"   Sample diff:\n{diff}")

def test_find_function():
    """Test function finding."""
    print("\nTesting function finding...")
    
    # Create a test file
    test_content = """from typing import List

def longestBalanced(nums: List[int]) -> int:
    \"\"\"Test function.\"\"\"
    return 0

def helper():
    pass
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(test_content)
        temp_file = Path(f.name)
    
    try:
        # Test finding function
        result = find_function_or_class(temp_file, "longestBalanced")
        assert result is not None, "Should find function"
        assert result['name'] == "longestBalanced", "Should find correct function"
        assert result['type'] == 'function', "Should identify as function"
        assert result['start_line'] == 3, "Should have correct start line"
        print(f"✅ Function found: {result['name']} (lines {result['start_line']}-{result['end_line']})")
        print(f"   Code preview: {result['code'][:50]}...")
        
        # Test finding non-existent function
        result2 = find_function_or_class(temp_file, "nonexistent")
        assert result2 is None, "Should not find non-existent function"
        print("✅ Non-existent function correctly returns None")
        
    finally:
        temp_file.unlink()

def test_replace_logic():
    """Test the logic for replacing code in a file."""
    print("\nTesting replace logic...")
    
    original_content = """def test():
    return 1

def other():
    pass
"""
    
    # Simulate replacing test() function
    func_info = {
        'start_line': 1,
        'end_line': 2,
        'code': 'def test():\n    return 1\n'
    }
    
    new_code = """def test():
    return 2
"""
    
    lines = original_content.splitlines(keepends=True)
    start_idx = func_info['start_line'] - 1
    end_idx = func_info['end_line']
    new_code_lines = new_code.splitlines(keepends=True)
    
    new_lines = lines[:start_idx] + new_code_lines + lines[end_idx:]
    new_content = ''.join(new_lines)
    
    expected = """def test():
    return 2

def other():
    pass
"""
    
    assert new_content == expected, "Replacement should work correctly"
    print("✅ Replace logic works correctly")
    print(f"   Original: {original_content.splitlines()[0]}")
    print(f"   New: {new_content.splitlines()[0]}")

def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Inline Edit Core Functionality")
    print("=" * 60)
    
    try:
        test_backup()
        test_diff()
        test_find_function()
        test_replace_logic()
        
        print("\n" + "=" * 60)
        print("✅ All core functionality tests passed!")
        print("=" * 60)
        print("\nNote: Full RAG functionality requires chromadb and langchain.")
        print("These tests verify the inline editing logic works correctly.")
        return 0
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())

