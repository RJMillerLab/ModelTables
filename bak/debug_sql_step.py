#!/usr/bin/env python
"""
Debug SQL step by step
"""
import os
import duckdb

# -------- paths / config --------
RAW_DIR = os.path.expanduser("~/Repo/CitationLake/data/raw")
PARQUET_GLOB = os.path.join(RAW_DIR, "train-*-of-00004.parquet")

con = duckdb.connect()

# Test the SQL conditions step by step
query = f"""
    SELECT modelId, card,
           card LIKE '---\\n%' as starts_with_yaml,
           position('\\n---\\n' IN card) as yaml_end_pos,
           position('\\n---\\n' IN card) > 0 as has_yaml_end,
           substring(card FROM position('\\n---\\n' IN card) + 5) as raw_extraction,
           trim(substring(card FROM position('\\n---\\n' IN card) + 5)) as trimmed_extraction,
           length(trim(substring(card FROM position('\\n---\\n' IN card) + 5))) as extraction_len
    FROM read_parquet('{PARQUET_GLOB}')
    WHERE modelId IN ('DackJan/minigpt-test', 'WenjunJi/Hello')
"""

results = con.execute(query).fetchall()

for row in results:
    print(f"=== {row[0]} ===")
    print(f"starts_with_yaml: {row[2]}")
    print(f"yaml_end_pos: {row[3]}")
    print(f"has_yaml_end: {row[4]}")
    print(f"extraction_len: {row[7]}")
    
    raw_extract = row[5]
    trimmed_extract = row[6]
    
    print(f"Raw extraction first 100 chars: {repr(raw_extract[:100]) if raw_extract else 'None'}")
    print(f"Trimmed extraction first 100 chars: {repr(trimmed_extract[:100]) if trimmed_extract else 'None'}")
    
    print("="*50)

# Test simpler approach
print("\nTesting manual extraction:")
card_query = f"""
    SELECT modelId, card
    FROM read_parquet('{PARQUET_GLOB}')
    WHERE modelId = 'DackJan/minigpt-test'
"""

result = con.execute(card_query).fetchone()
if result:
    card = result[1]
    print(f"Full card length: {len(card)}")
    
    # Find YAML end position manually
    lines = card.split('\n')
    yaml_end = -1
    for i, line in enumerate(lines):
        if i > 0 and line.strip() == '---':  # Skip first ---
            yaml_end = i
            break
    
    if yaml_end > 0:
        body_lines = lines[yaml_end + 1:]
        body = '\n'.join(body_lines).strip()
        print(f"Manually extracted body length: {len(body)}")
        print(f"Body starts with: {repr(body[:100])}")