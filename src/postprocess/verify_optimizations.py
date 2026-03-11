"""
Quick verification that optimizations work correctly
"""
from src.data_preprocess.step2_hugging_github_extract import detect_and_extract_markdown_tables
import time

test_cases = [
    ("No HTML (should be fast)", "| A | B |\n|---|---|\n| 1 | 2 |", 1),
    ("With HTML", "<table><tr><th>A</th></tr><tr><td>1</td></tr></table>", 1),
    ("Both formats", "| A | B |\n|---|---|\n| 1 | 2 |\n<table><tr><th>C</th></tr><tr><td>3</td></tr></table>", 2),
    ("Large no HTML", "No tables here. " * 10000, 0),
]

print("="*80)
print("VERIFICATION: Optimizations work correctly")
print("="*80)

all_passed = True

for name, content, expected_count in test_cases:
    start = time.perf_counter()
    try:
        found, tables = detect_and_extract_markdown_tables(content)
        elapsed = time.perf_counter() - start
        
        if len(tables) == expected_count:
            status = "✅ PASS"
        else:
            status = f"❌ FAIL (got {len(tables)}, expected {expected_count})"
            all_passed = False
        
        print(f"\n{name}: {status}")
        print(f"  Time: {elapsed*1000:.2f} ms")
        
    except Exception as e:
        print(f"\n{name}: ❌ ERROR - {e}")
        all_passed = False

print("\n" + "="*80)
if all_passed:
    print("✅ ALL TESTS PASSED - Optimizations working correctly!")
else:
    print("❌ SOME TESTS FAILED - Check the code")
print("="*80)

