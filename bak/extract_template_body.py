#!/usr/bin/env python
"""
Extract and compare template body from specific cards
"""
import os
import duckdb
from pathlib import Path

def extract_body_from_card(card_content):
    """Extract the body content after YAML header"""
    lines = card_content.split('\n')
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
        return '\n'.join(lines[yaml_end_idx + 1:]).strip()
    else:
        return card_content.strip()

# -------- paths / config --------
RAW_DIR = os.path.expanduser("~/Repo/CitationLake/data/raw")
PARQUET_GLOB = os.path.join(RAW_DIR, "train-*-of-00004.parquet")

# Load template
TEMPLATE_PATH = Path("modelcard_template.md")
if TEMPLATE_PATH.exists():
    official_template = TEMPLATE_PATH.read_text(encoding="utf-8")
    template_body_only = extract_body_from_card(official_template)
    print(f"Template body length: {len(template_body_only)} characters")
else:
    print("Template file not found!")
    exit(1)

# -------- duckdb connection --------
con = duckdb.connect()

# Test specific models
test_models = ["DackJan/minigpt-test", "WenjunJi/Hello"]

print("\nExtracting and comparing template bodies:")

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
    print(f"\n=== {model_id} ===")
    
    model_query = f"""
        SELECT modelId, author, card
        FROM ({no_policy_query})
        WHERE modelId = '{model_id}'
        LIMIT 1
    """
    
    result = con.execute(model_query).fetchall()
    
    if result:
        card_content = result[0][2]
        
        # Extract body from card
        card_body = extract_body_from_card(card_content)
        
        print(f"Card body length: {len(card_body)} characters")
        print(f"Template body length: {len(template_body_only)} characters")
        
        # Compare bodies
        body_match = card_body.strip() == template_body_only.strip()
        print(f"Body match: {body_match}")
        
        if body_match:
            print("✓ EXACT MATCH FOUND!")
        else:
            # Show first difference
            card_norm = card_body.strip()
            template_norm = template_body_only.strip()
            min_len = min(len(card_norm), len(template_norm))
            
            for i in range(min_len):
                if card_norm[i] != template_norm[i]:
                    print(f"First difference at position {i}:")
                    print(f"  Card:     {repr(card_norm[max(0,i-20):i+20])}")
                    print(f"  Template: {repr(template_norm[max(0,i-20):i+20])}")
                    break
            
            if len(card_norm) != len(template_norm):
                print(f"Length difference: card={len(card_norm)}, template={len(template_norm)}")
        
        # Show first 300 chars of each
        print(f"\nCard body preview:")
        print(repr(card_body[:300]))
        print(f"\nTemplate body preview:")
        print(repr(template_body_only[:300]))
        
    else:
        print(f"Model {model_id} not found")

print("\nDone.")