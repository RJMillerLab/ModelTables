"""
Test if markdown extraction and HTML extraction might extract the same table twice
"""
from src.data_preprocess.step2_hugging_github_extract import detect_and_extract_markdown_tables

# Test case 1: Pure HTML table (no markdown)
html_only = """
Some text before

<table>
  <tr>
    <th>Model</th>
    <th>Score</th>
  </tr>
  <tr>
    <td>BERT</td>
    <td>92.4</td>
  </tr>
</table>

Some text after
"""

# Test case 2: HTML table with pipes inside (worst case)
html_with_pipes = """
<table>
  <tr>
    <td>Model | Version</td>
    <td>Score</td>
  </tr>
  <tr>
    <td>BERT | v1</td>
    <td>92.4</td>
  </tr>
</table>
"""

# Test case 3: Both formats in same content
both_formats = """
Markdown table:

| Model | Score |
|-------|-------|
| BERT  | 92.4  |

HTML table (different data):

<table>
  <tr>
    <th>Name</th>
    <th>Value</th>
  </tr>
  <tr>
    <td>A</td>
    <td>1</td>
  </tr>
</table>
"""

print("="*80)
print("TEST 1: Pure HTML table (no markdown pipes)")
print("="*80)
found, tables = detect_and_extract_markdown_tables(html_only)
print(f"Extracted {len(tables)} table(s)")
for i, t in enumerate(tables, 1):
    print(f"\nTable {i}:")
    print(t[:150])
    if '<table>' in t or '<tr>' in t or '<td>' in t:
        print("  ⚠️  Contains HTML tags (HTML extraction)")
    else:
        print("  ✅ Pure markdown format")

print("\n" + "="*80)
print("TEST 2: HTML with pipes inside cells")
print("="*80)
found, tables = detect_and_extract_markdown_tables(html_with_pipes)
print(f"Extracted {len(tables)} table(s)")
for i, t in enumerate(tables, 1):
    print(f"\nTable {i}:")
    print(t[:150])

if len(tables) == 1:
    print("\n✅ Good: Only extracted once (either markdown OR HTML)")
elif len(tables) == 2:
    print("\n❌ BAD: Extracted twice (once as markdown, once as HTML)")

print("\n" + "="*80)
print("TEST 3: Both markdown and HTML (different content)")
print("="*80)
found, tables = detect_and_extract_markdown_tables(both_formats)
print(f"Extracted {len(tables)} table(s)")
for i, t in enumerate(tables, 1):
    print(f"\nTable {i} preview:")
    lines = t.split('\n')[:3]
    for line in lines:
        print(f"  {line[:80]}")

print("\n" + "="*80)
print("ANALYSIS")
print("="*80)
print("""
Current flow:
1. Lines 52-145: Markdown extraction (scan for |...|)
2. Lines 199-255: HTML extraction (BeautifulSoup find <table>)
3. Lines 216-252: Deduplication (signature comparison)

Question: Will markdown extraction pick up HTML tables?
- If HTML has no | symbols → NO (markdown won't extract it)
- If HTML cells contain | → MAYBE (might be partially extracted)

Current dedup scope: WITHIN same card only
- existing_signatures is created fresh for each card
- Compares HTML tables against markdown tables from SAME card
- Does NOT compare across different cards

Is this correct? YES!
- Same card: deduplicated ✅
- Different cards: separate ✅
- Different sources (HF/GitHub/ArXiv): separate ✅
""")

