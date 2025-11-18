"""
Test if HTML and markdown tables might be duplicated
"""
from src.data_preprocess.step2_hugging_github_extract import detect_and_extract_markdown_tables

# Test case: Content with BOTH markdown and HTML versions of the same table
mixed_content = """
## Markdown Table

| Model | Score |
|-------|-------|
| BERT  | 92.4  |
| GPT-2 | 89.1  |

## HTML Table (same content)

<table>
  <tr>
    <th>Model</th>
    <th>Score</th>
  </tr>
  <tr>
    <td>BERT</td>
    <td>92.4</td>
  </tr>
  <tr>
    <td>GPT-2</td>
    <td>89.1</td>
  </tr>
</table>

## Another table

| Name | Value |
|------|-------|
| A    | 1     |
"""

print("="*80)
print("Testing duplicate detection")
print("="*80)
print("\nContent has:")
print("  - 1 markdown table (Model/Score)")
print("  - 1 HTML table (same Model/Score)")
print("  - 1 different markdown table (Name/Value)")

found, tables = detect_and_extract_markdown_tables(mixed_content)

print(f"\nExtraction result:")
print(f"  Found: {found}")
print(f"  Tables: {len(tables)}")

if len(tables) == 2:
    print("\n✅ PERFECT: Extracted 2 tables (deduplicated successfully)")
elif len(tables) == 3:
    print("\n⚠️  WARNING: Extracted 3 tables (Model/Score duplicated)")
    print("\nLet's check if they are duplicates:")
    for i, table in enumerate(tables, 1):
        print(f"\n--- TABLE {i} ---")
        print(table[:200])
        # Normalize for comparison
        normalized = ' '.join(table.split())
        if 'Model' in table and 'Score' in table and 'BERT' in table:
            print("  -> Contains Model/Score/BERT (potential duplicate)")
else:
    print(f"\n❓ Extracted {len(tables)} tables")

for i, table in enumerate(tables, 1):
    print(f"\n{'='*80}")
    print(f"TABLE {i}:")
    print(f"{'='*80}")
    print(table)

