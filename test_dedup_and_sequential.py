"""
Test deduplication and sequential numbering
"""
from src.data_preprocess.step2_gitcard_tab import detect_and_extract_markdown_tables
import re

# Simulate extraction with duplicates and empty tables
test_content = """
## Table 1 (markdown)

| Model | Score |
|-------|-------|
| BERT  | 92.4  |

## Table 2 (HTML - same as table 1)

<table>
  <tr><th>Model</th><th>Score</th></tr>
  <tr><td>BERT</td><td>92.4</td></tr>
</table>

## Table 3 (different markdown)

| Name | Value |
|------|-------|
| A    | 1     |

## Table 4 (empty - will fail CSV conversion)

| Col1 | Col2 |
|------|------|

## Table 5 (another real table)

| Method | Accuracy |
|--------|----------|
| SGD    | 0.95     |
"""

print("="*80)
print("TEST: Deduplication and Sequential Numbering")
print("="*80)

print("\nInput content has:")
print("  - Table 1: Markdown (BERT/Score)")
print("  - Table 2: HTML (same BERT/Score) - should be DEDUPLICATED")
print("  - Table 3: Markdown (Name/Value)")
print("  - Table 4: Empty table - will FAIL CSV conversion")
print("  - Table 5: Markdown (Method/Accuracy)")

# Extract
found, tables = detect_and_extract_markdown_tables(test_content)

print(f"\n{'='*80}")
print("EXTRACTION RESULT:")
print(f"{'='*80}")
print(f"Tables extracted: {len(tables)}")

# Simulate the deduplication logic from saving part
def normalize_for_dedup(table_str):
    """Normalize table content for duplicate detection."""
    lines = [ln.strip() for ln in table_str.split('\n') if ln.strip()]
    # Remove separators
    data = [ln for ln in lines if not re.fullmatch(r'^\|?\s*[:\-\| ]+\|?\s*$', ln.strip())]
    # Normalize pipes and spaces
    normalized = []
    for ln in data:
        ln = re.sub(r'\s*\|\s*', '|', ln)
        ln = re.sub(r'\s+', ' ', ln)
        normalized.append(ln.strip())
    return '||'.join(normalized)

# Deduplicate
seen_signatures = set()
unique_tables = []
for table_content in tables:
    sig = normalize_for_dedup(table_content)
    if sig not in seen_signatures:
        unique_tables.append(table_content)
        seen_signatures.add(sig)
    else:
        print(f"\n  ⚠️  Duplicate detected and removed!")
        print(f"      Signature: {sig[:50]}...")

print(f"\nAfter deduplication: {len(unique_tables)} unique tables")

# Show which tables remain
print(f"\n{'='*80}")
print("UNIQUE TABLES:")
print(f"{'='*80}")
for i, table in enumerate(unique_tables, 1):
    lines = table.split('\n')
    first_line = lines[0] if lines else ""
    print(f"\nTable {i}: {first_line[:60]}...")
    
    # Check content
    if 'BERT' in table and 'Score' in table:
        print(f"  -> BERT/Score table")
    elif 'Name' in table and 'Value' in table:
        print(f"  -> Name/Value table")
    elif 'Method' in table and 'Accuracy' in table:
        print(f"  -> Method/Accuracy table")
    elif not any(c.isalpha() for c in table):
        print(f"  -> Empty table (will fail CSV)")

# Simulate sequential numbering with failed saves
print(f"\n{'='*80}")
print("SEQUENTIAL NUMBERING SIMULATION:")
print(f"{'='*80}")
print("Simulating CSV conversion and numbering...")

table_counter = 0
saved_files = []

for j, table in enumerate(unique_tables, start=1):
    # Simulate CSV conversion (table 4 would fail)
    is_empty = not any(c.isalnum() for line in table.split('\n')[2:] for c in line)
    
    if not is_empty:
        table_counter += 1
        filename = f"hash_table{table_counter}.csv"
        saved_files.append(filename)
        print(f"  Original index {j} → Sequential number {table_counter} → {filename} ✅")
    else:
        print(f"  Original index {j} → Failed (empty) ❌")

print(f"\n{'='*80}")
print("FINAL RESULT:")
print(f"{'='*80}")
print(f"Input: 5 tables (with 1 duplicate)")
print(f"After dedup: {len(unique_tables)} tables")
print(f"After CSV conversion: {table_counter} files saved")
print(f"File names: {', '.join(saved_files)}")

if len(saved_files) == 3 and saved_files == ['hash_table1.csv', 'hash_table2.csv', 'hash_table3.csv']:
    print("\n✅ PERFECT!")
    print("   - Duplicates removed")
    print("   - Empty tables skipped")
    print("   - Sequential numbering (no gaps)")
else:
    print(f"\n⚠️  Result: {saved_files}")

