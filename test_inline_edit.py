#!/usr/bin/env python3
"""
Test script for Inline_Edit_Rag.py core functionality
Tests backup, diff, and function finding without requiring full RAG setup
"""

import sys
import tempfile
import shutil
from pathlib import Path
from Inline_Edit_Rag import InlineEditRAG

def test_backup():
    """Test backup functionality."""
    print("Testing backup functionality...")
    
    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write("def test():\n    pass\n")
        temp_file = Path(f.name)
    
    try:
        # Create a minimal RAG instance (without full initialization)
        rag = InlineEditRAG.__new__(InlineEditRAG)
        rag.backup_dir = Path(".backups")
        rag.backup_dir.mkdir(exist_ok=True)
        
        # Test backup
        backup_path = rag.backup_file(temp_file)
        assert backup_path is not None, "Backup should succeed"
        assert backup_path.exists(), "Backup file should exist"
        print(f"✅ Backup created: {backup_path.name}")
        
        # Cleanup
        backup_path.unlink()
    finally:
        temp_file.unlink()
        if rag.backup_dir.exists():
            shutil.rmtree(rag.backup_dir, ignore_errors=True)

def test_diff():
    """Test diff creation."""
    print("\nTesting diff creation...")
    
    old_content = "def test():\n    return 1\n"
    new_content = "def test():\n    return 2\n"
    
    rag = InlineEditRAG.__new__(InlineEditRAG)
    diff = rag.create_diff(old_content, new_content, "test.py")
    
    assert "--- test.py" in diff, "Diff should contain file name"
    assert "+++ test.py" in diff, "Diff should contain file name"
    assert "-    return 1" in diff or "+    return 2" in diff, "Diff should show changes"
    print("✅ Diff creation works")
    print(f"   Sample diff:\n{diff[:200]}...")

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
        rag = InlineEditRAG.__new__(InlineEditRAG)
        
        # Test finding function
        result = rag.find_function_or_class(temp_file, "longestBalanced")
        assert result is not None, "Should find function"
        assert result['name'] == "longestBalanced", "Should find correct function"
        assert result['type'] == 'function', "Should identify as function"
        assert result['start_line'] == 3, "Should have correct start line"
        print(f"✅ Function found: {result['name']} (lines {result['start_line']}-{result['end_line']})")
        
        # Test finding non-existent function
        result2 = rag.find_function_or_class(temp_file, "nonexistent")
        assert result2 is None, "Should not find non-existent function"
        print("✅ Non-existent function correctly returns None")
        
    finally:
        temp_file.unlink()

def test_extract_code_blocks():
    """Test code block extraction."""
    print("\nTesting code block extraction...")
    
    rag = InlineEditRAG.__new__(InlineEditRAG)
    
    text = """
Here is some code:
```python
def test():
    return 1
```

And more:
```javascript
console.log("test");
```
"""
    
    blocks = rag.extract_code_blocks(text)
    assert len(blocks) == 2, "Should extract 2 code blocks"
    assert blocks[0][0] == "python", "First block should be Python"
    assert "def test()" in blocks[0][1], "Should contain function code"
    print(f"✅ Extracted {len(blocks)} code blocks")
    print(f"   Block 1: {blocks[0][0]} ({len(blocks[0][1])} chars)")
    print(f"   Block 2: {blocks[1][0]} ({len(blocks[1][1])} chars)")

def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Inline_Edit_Rag.py Core Functionality")
    print("=" * 60)
    
    try:
        test_backup()
        test_diff()
        test_find_function()
        test_extract_code_blocks()
        
        print("\n" + "=" * 60)
        print("✅ All core functionality tests passed!")
        print("=" * 60)
        return 0
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())

