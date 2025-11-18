"""
Test parallel saving with sequential numbering
"""
import pandas as pd
import os
import tempfile
import shutil
from src.data_preprocess.step2_hugging_github_extract import (
    detect_and_extract_markdown_tables,
    sanitize_markdown_table_separators,
    generate_csv_path_for_dedup,
    normalize_table_for_dedup
)
from src.data_ingestion.readme_parser import MarkdownHandler

# Create test data
test_data = {
    'hash1': """
| Model | Score |
|-------|-------|
| BERT  | 92.4  |
| GPT   | 89.1  |

| Name | Value |
|------|-------|
| A    | 1     |
| B    | 2     |
""",
    'hash2': """
<table>
  <tr><th>Method</th><th>Acc</th></tr>
  <tr><td>SGD</td><td>0.95</td></tr>
</table>

| Loss | Epoch |
|------|-------|
| 0.1  | 10    |
""",
    'hash3': """
| Category | Count |
|----------|-------|
| A        | 100   |
"""
}

print("="*80)
print("TEST: Parallel Saving with Sequential Numbering")
print("="*80)

# Create temp directory
temp_dir = tempfile.mkdtemp(prefix="test_parallel_")
print(f"\nTemp dir: {temp_dir}")

# Process each hash
results = []
for hash_key, content in test_data.items():
    print(f"\n--- Processing {hash_key} ---")
    
    # Extract tables
    found, tables = detect_and_extract_markdown_tables(content)
    print(f"  Extracted: {len(tables)} tables")
    
    # Deduplicate
    seen_signatures = set()
    unique_tables = []
    for table_content in tables:
        sig = normalize_table_for_dedup(table_content)
        if sig not in seen_signatures:
            unique_tables.append(table_content)
            seen_signatures.add(sig)
    
    if len(unique_tables) != len(tables):
        print(f"  Dedup: {len(tables)} → {len(unique_tables)} unique")
    
    # Sanitize
    sanitized_tables = [sanitize_markdown_table_separators(t) for t in unique_tables]
    
    # Convert all
    conversion_results = []
    for j, table_content in enumerate(sanitized_tables, start=1):
        temp_out_path = generate_csv_path_for_dedup(hash_key, j, temp_dir)
        tmp_csv_path = MarkdownHandler.markdown_to_csv(table_content, temp_out_path, verbose=False)
        conversion_results.append({
            'original_index': j,
            'success': tmp_csv_path is not None,
            'temp_path': temp_out_path if tmp_csv_path else None
        })
    
    # Assign sequential numbers
    table_counter = 0
    for result in conversion_results:
        if result['success']:
            table_counter += 1
            result['final_index'] = table_counter
    
    print(f"  Conversion: {table_counter}/{len(sanitized_tables)} succeeded")
    
    # Rename to sequential
    csv_list = []
    for result in conversion_results:
        if result['success']:
            temp_path = result['temp_path']
            final_index = result['final_index']
            final_out_path = generate_csv_path_for_dedup(hash_key, final_index, temp_dir)
            
            if temp_path != final_out_path:
                os.rename(temp_path, final_out_path)
            
            csv_list.append(os.path.basename(final_out_path))
    
    print(f"  Saved: {csv_list}")
    results.append((hash_key, csv_list))

# Verify
print(f"\n{'='*80}")
print("VERIFICATION:")
print(f"{'='*80}")

all_files = sorted(os.listdir(temp_dir))
print(f"\nAll files in temp dir ({len(all_files)}):")
for f in all_files:
    print(f"  - {f}")

print(f"\n{'='*80}")
print("CHECK: Sequential numbering")
print(f"{'='*80}")

for hash_key, csv_list in results:
    print(f"\n{hash_key}: {len(csv_list)} files")
    # Check if numbering is sequential
    numbers = []
    for csv_name in csv_list:
        # Extract table number from filename like "hash_table3.csv"
        import re
        match = re.search(r'_table(\d+)\.csv', csv_name)
        if match:
            numbers.append(int(match.group(1)))
    
    numbers.sort()
    expected = list(range(1, len(numbers) + 1))
    
    if numbers == expected:
        print(f"  ✅ Sequential: {numbers}")
    else:
        print(f"  ❌ NOT sequential: {numbers} (expected: {expected})")

# Cleanup
shutil.rmtree(temp_dir)
print(f"\n✅ Test completed, temp dir cleaned")

