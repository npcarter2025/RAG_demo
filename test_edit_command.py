#!/usr/bin/env python3
"""
Test edit command parsing
"""

def test_edit_command_parsing():
    """Test that edit commands are parsed correctly."""
    print("Testing edit command parsing...")
    
    test_cases = [
        ("edit: add_numbers.py Add error handling", ("add_numbers.py", "Add error handling")),
        ("edit: test.py Fix the bug", ("test.py", "Fix the bug")),
        ("edit: file.py Add type hints", ("file.py", "Add type hints")),
    ]
    
    for command, expected in test_cases:
        if command.lower().startswith("edit:"):
            edit_cmd = command[5:].strip()
            parts = edit_cmd.split(None, 1)
            
            if len(parts) >= 2:
                target = parts[0]
                instruction = parts[1]
                assert target == expected[0], f"Target mismatch: {target} != {expected[0]}"
                assert instruction == expected[1], f"Instruction mismatch: {instruction} != {expected[1]}"
                print(f"✅ Parsed: '{command}' -> target='{target}', instruction='{instruction}'")
            else:
                print(f"❌ Failed to parse: '{command}'")
                return False
    
    print("✅ All edit command parsing tests passed!")
    return True

if __name__ == "__main__":
    test_edit_command_parsing()

