#!/usr/bin/env python
"""
Test specific template matches we found earlier
"""
import os
import duckdb
from pathlib import Path

# -------- paths / config --------
RAW_DIR = os.path.expanduser("~/Repo/CitationLake/data/raw")
PARQUET_GLOB = os.path.join(RAW_DIR, "train-*-of-00004.parquet")

# Load template
TEMPLATE_PATH = Path("modelcard_template.md")
if TEMPLATE_PATH.exists():
    official_template = TEMPLATE_PATH.read_text(encoding="utf-8")
    
    # Extract template body without YAML header
    lines = official_template.split('\n')
    yaml_end_idx = -1
    yaml_started = False
    for i, line in enumerate(lines):
        if line.strip() == '---':
            if not yaml_started:
                yaml_started = True
            else:
                yaml_end_idx = i
                break
    
    if yaml_end_idx > 0:
        template_body_only = '\n'.join(lines[yaml_end_idx + 1:]).strip()
    else:
        template_body_only = official_template.strip()

    print(f"Template body length: {len(template_body_only)} characters")
else:
    print("Template file not found!")
    exit(1)

# -------- duckdb connection --------
con = duckdb.connect()

# Test specific models we know should match
test_models = ["DackJan/minigpt-test", "WenjunJi/Hello"]

print("\nTesting specific models that should match template:")

# Filter conditions
empty_card_conditions = [
    "card IS NOT NULL",
    "trim(card) <> ''",
    "card <> 'None'",
    "card <> 'none'",
    "card <> 'Null'", 
    "card <> 'null'",
    "card <> 'N/A'",
    "card <> 'n/a'",
    "card <> 'Entry not found'",
    "length(card) > 10"
]

BASE = f"FROM read_parquet('{PARQUET_GLOB}')"
non_empty_query = f"SELECT * {BASE} WHERE {' AND '.join(empty_card_conditions)}"
no_policy_query = f"SELECT * FROM ({non_empty_query}) WHERE card <> 'Invalid username or password.'"

for model_id in test_models:
    print(f"\nChecking {model_id}...")
    
    # Find the specific model
    model_query = f"""
        SELECT modelId, author, card, length(card) as card_len
        FROM ({no_policy_query})
        WHERE modelId = '{model_id}'
        LIMIT 1
    """
    
    result = con.execute(model_query).fetchall()
    
    if result:
        model_data = result[0]
        print(f"  Found: {model_data[0]} by {model_data[1]}")
        print(f"  Card length: {model_data[3]} characters")
        
        card_content = model_data[2]
        
        # Test different matching approaches
        # 1. Direct comparison
        direct_match = card_content.strip() == template_body_only
        print(f"  Direct match: {direct_match}")
        
        # 2. Normalize newlines
        normalized_card = card_content.replace('\r\n', '\n').replace('\r', '\n').strip()
        normalized_template = template_body_only.replace('\r\n', '\n').replace('\r', '\n').strip()
        newline_match = normalized_card == normalized_template
        print(f"  Newline normalized match: {newline_match}")
        
        # 3. Show first 200 chars for comparison
        print(f"  Card preview: {repr(card_content[:200])}")
        print(f"  Template preview: {repr(template_body_only[:200])}")
        
        # 4. Compare lengths
        print(f"  Card length: {len(card_content)}, Template length: {len(template_body_only)}")
        
        if not newline_match:
            # Find where they differ
            min_len = min(len(normalized_card), len(normalized_template))
            for i in range(min_len):
                if normalized_card[i] != normalized_template[i]:
                    print(f"  First difference at position {i}:")
                    print(f"    Card: {repr(normalized_card[max(0,i-10):i+10])}")
                    print(f"    Template: {repr(normalized_template[max(0,i-10):i+10])}")
                    break
        
    else:
        print(f"  Model {model_id} not found in dataset")

print(f"\nDone testing specific models.")