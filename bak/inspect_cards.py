#!/usr/bin/env python
"""
Inspect the actual card content format
"""
import os
import duckdb

# -------- paths / config --------
RAW_DIR = os.path.expanduser("~/Repo/CitationLake/data/raw")
PARQUET_GLOB = os.path.join(RAW_DIR, "train-*-of-00004.parquet")

con = duckdb.connect()

# Get the specific cards
query = f"""
    SELECT modelId, card
    FROM read_parquet('{PARQUET_GLOB}')
    WHERE modelId IN ('DackJan/minigpt-test', 'WenjunJi/Hello')
    LIMIT 2
"""

results = con.execute(query).fetchall()

for model_id, card in results:
    print(f"=== {model_id} ===")
    print(f"Card length: {len(card)}")
    print("First 300 characters:")
    print(repr(card[:300]))
    
    # Show where the YAML ends
    lines = card.split('\n')
    print(f"\nFirst 10 lines:")
    for i, line in enumerate(lines[:10]):
        print(f"  {i:2}: {repr(line)}")
    
    # Find YAML end manually
    yaml_end = -1
    yaml_started = False
    for i, line in enumerate(lines):
        if line.strip() == '---':
            if not yaml_started:
                yaml_started = True
                print(f"YAML starts at line {i}")
            else:
                yaml_end = i
                print(f"YAML ends at line {i}")
                break
    
    if yaml_end > 0:
        body = '\n'.join(lines[yaml_end + 1:])
        print(f"Extracted body length: {len(body)}")
        print(f"Body starts with: {repr(body[:100])}")
    
    print("\n" + "="*50 + "\n")