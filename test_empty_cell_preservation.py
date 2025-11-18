"""
Test empty cell preservation in markdown table extraction
"""
from src.data_preprocess.step2_hugging_github_extract import detect_and_extract_markdown_tables

test_cases = [
    # Test 1: Leading empty cells (the original problem)
    {
        "name": "Leading empty cells",
        "markdown": """
|||TruthfulQA|Toxigen|
|---|---|---|
|Llama 1|7B|27.42|23.00|
""",
        "expected": {
            "header_cells": ["", "", "TruthfulQA", "Toxigen"],
            "data_cells_first_row": ["Llama 1", "7B", "27.42", "23.00"]
        }
    },
    
    # Test 2: Middle empty cells
    {
        "name": "Middle empty cells",
        "markdown": """
|A|B|C|D|
|---|---|---|---|
|1|2||4|
|5||7|8|
""",
        "expected": {
            "header_cells": ["A", "B", "C", "D"],
            "data_cells_first_row": ["1", "2", "", "4"],
            "data_cells_second_row": ["5", "", "7", "8"]
        }
    },
    
    # Test 3: Trailing empty cells
    {
        "name": "Trailing empty cells",
        "markdown": """
|Name|Age|City||
|---|---|---|---|
|John|30|NYC||
|Jane|25|LA||
""",
        "expected": {
            "header_cells": ["Name", "Age", "City", ""],
            "data_cells_first_row": ["John", "30", "NYC", ""]
        }
    },
    
    # Test 4: All empty cells in a row
    # Note: The filter in HTML table extraction will skip all-empty rows
    # So this test expects the all-empty row to be removed
    {
        "name": "All empty cells row (filtered)",
        "markdown": """
|A|B|C|
|---|---|---|---|
||||
|1|2|3|
""",
        "expected": {
            "header_cells": ["A", "B", "C"],
            "data_cells_first_row": ["1", "2", "3"],
            # All-empty row is filtered out in HTML processing
        }
    },
    
    # Test 5: Normal table without empty cells
    {
        "name": "Normal table (no empty cells)",
        "markdown": """
|Name|Age|
|---|---|
|John|30|
|Jane|25|
""",
        "expected": {
            "header_cells": ["Name", "Age"],
            "data_cells_first_row": ["John", "30"]
        }
    },
    
    # Test 6: Mixed leading, middle, and trailing empty cells
    {
        "name": "Mixed empty cells",
        "markdown": """
|||Middle||
|---|---|---|---|
|A||C||
""",
        "expected": {
            "header_cells": ["", "", "Middle", ""],
            "data_cells_first_row": ["A", "", "C", ""]
        }
    },
    
    # Test 7: Single empty cell
    {
        "name": "Single empty cell",
        "markdown": """
|A|B|
|---|---|
||X|
""",
        "expected": {
            "header_cells": ["A", "B"],
            "data_cells_first_row": ["", "X"]
        }
    },
    
    # Test 8: Empty header
    {
        "name": "Empty header cells",
        "markdown": """
|||Col3|
|---|---|---|
|Val1|Val2|Val3|
""",
        "expected": {
            "header_cells": ["", "", "Col3"],
            "data_cells_first_row": ["Val1", "Val2", "Val3"]
        }
    },
]

def test_case(name, markdown, expected):
    """Test a single case and return pass/fail"""
    found, tables = detect_and_extract_markdown_tables(markdown)
    
    if not found or len(tables) == 0:
        return False, "No table extracted"
    
    table = tables[0]
    lines = [l for l in table.split('\n') if l.strip()]
    
    if len(lines) < 2:
        return False, f"Too few lines: {len(lines)}"
    
    # Parse header
    header_line = lines[0]
    header_cells = [c.strip() for c in header_line.split('|')[1:-1]]
    
    # Parse data rows (skip separator line with ---)
    data_rows = []
    for line in lines[1:]:
        if not line.strip().startswith('|'):
            continue
        # Skip separator lines
        if all(c in ['-', ':', ' ', '|'] for c in line):
            continue
        cells = [c.strip() for c in line.split('|')[1:-1]]
        data_rows.append(cells)
    
    # Check header
    if 'header_cells' in expected:
        if header_cells != expected['header_cells']:
            return False, f"Header mismatch: got {header_cells}, expected {expected['header_cells']}"
    
    # Check first data row
    if 'data_cells_first_row' in expected and data_rows:
        if data_rows[0] != expected['data_cells_first_row']:
            return False, f"First row mismatch: got {data_rows[0]}, expected {expected['data_cells_first_row']}"
    
    # Check second data row
    if 'data_cells_second_row' in expected and len(data_rows) > 1:
        if data_rows[1] != expected['data_cells_second_row']:
            return False, f"Second row mismatch: got {data_rows[1]}, expected {expected['data_cells_second_row']}"
    
    return True, "Pass"

print("="*80)
print("Testing Empty Cell Preservation")
print("="*80)

passed = 0
failed = 0

for case in test_cases:
    success, message = test_case(**case)
    status = "✅" if success else "❌"
    print(f"\n{status} {case['name']}")
    print(f"   {message}")
    
    if success:
        passed += 1
    else:
        failed += 1

print("\n" + "="*80)
print(f"Summary: {passed} passed, {failed} failed")
print("="*80)

