#!/usr/bin/env python
"""
Find cards that exactly match the official HuggingFace model card template.
This will show how many cards are literally just the unmodified template.
"""
import os
import duckdb
import pandas as pd
import json
import requests
from pathlib import Path

# -------- paths / config --------
RAW_DIR = os.path.expanduser("~/Repo/CitationLake/data/raw")
PARQUET_GLOB = os.path.join(RAW_DIR, "train-*-of-00004.parquet")

# Download the official template
TEMPLATE_PATH = Path("official_modelcard_template.md")
TEMPLATE_URL = "https://raw.githubusercontent.com/huggingface/huggingface_hub/main/src/huggingface_hub/templates/modelcard_template.md"

print("Downloading official HuggingFace model card template...")
try:
    response = requests.get(TEMPLATE_URL, timeout=30)
    response.raise_for_status()
    official_template = response.text
    TEMPLATE_PATH.write_text(official_template, encoding='utf-8')
    print(f"Template downloaded successfully ({len(official_template)} characters)")
    print(f"Saved to: {TEMPLATE_PATH}")
except Exception as e:
    print(f"Error downloading template: {e}")
    exit(1)

# Also try common variations of the template (with different YAML headers)
print(f"\nOfficial template preview (first 500 chars):")
print(repr(official_template[:500]))

# -------- duckdb connection --------
con = duckdb.connect()

# Apply the same filtering as the original script
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

print("\nLoading data to find exact template matches...")
query = f"""
SELECT 
    modelId,
    author, 
    card,
    likes,
    downloads,
    length(card) as card_length
FROM ({no_policy_query})
ORDER BY likes DESC, downloads DESC
"""

df = con.execute(query).fetchdf()
print(f"Total cards to check: {len(df)}")

# Find exact matches with the official template
print(f"\nSearching for cards that exactly match the official template...")

# Normalize whitespace for comparison (in case of minor formatting differences)
def normalize_content(text):
    # Keep original for exact match, but also try normalized version
    return text.strip()

official_template_normalized = normalize_content(official_template)

# Check for exact matches
exact_matches = df[df['card'] == official_template]
exact_matches_normalized = df[df['card'].apply(normalize_content) == official_template_normalized]

print(f"Exact matches (character-perfect): {len(exact_matches)}")
print(f"Exact matches (normalized whitespace): {len(exact_matches_normalized)}")

# Also check for matches without YAML front matter (template body only)
# Extract just the markdown content after YAML
template_lines = official_template.split('\n')
yaml_end_idx = -1
yaml_started = False

for i, line in enumerate(template_lines):
    if line.strip() == '---':
        if not yaml_started:
            yaml_started = True
        else:
            yaml_end_idx = i
            break

if yaml_end_idx > 0:
    template_body = '\n'.join(template_lines[yaml_end_idx + 1:]).strip()
    print(f"\nTemplate body without YAML header ({len(template_body)} chars):")
    print(repr(template_body[:200]))
    
    # Check for cards that match just the template body
    def extract_body(card_text):
        lines = card_text.split('\n')
        yaml_end = -1
        yaml_start = False
        for i, line in enumerate(lines):
            if line.strip() == '---':
                if not yaml_start:
                    yaml_start = True
                else:
                    yaml_end = i
                    break
        if yaml_end > 0:
            return '\n'.join(lines[yaml_end + 1:]).strip()
        return card_text.strip()
    
    body_matches = df[df['card'].apply(extract_body) == template_body]
    print(f"Cards matching template body (ignoring YAML): {len(body_matches)}")
else:
    body_matches = pd.DataFrame()
    print("Could not extract template body (no YAML found)")

# Combine all types of matches
all_matches = pd.concat([exact_matches, exact_matches_normalized, body_matches]).drop_duplicates()
print(f"\nTotal template matches (all types): {len(all_matches)}")

if len(all_matches) > 0:
    print("\nTop 20 models using the official template:")
    for idx, (_, row) in enumerate(all_matches.head(20).iterrows(), 1):
        print(f"\n{idx}. {row['modelId']}")
        print(f"   Author: {row['author']}")
        print(f"   Likes: {row['likes']}")
        print(f"   Downloads: {row['downloads']}")
        print(f"   Card length: {row['card_length']} chars")
        
        # Determine match type
        match_type = "unknown"
        if row['card'] == official_template:
            match_type = "exact"
        elif normalize_content(row['card']) == official_template_normalized:
            match_type = "normalized"
        elif len(body_matches) > 0 and row['modelId'] in body_matches['modelId'].values:
            match_type = "body_only"
        print(f"   Match type: {match_type}")

    # Save results
    results = {
        "official_template_length": len(official_template),
        "total_cards_checked": len(df),
        "exact_matches": len(exact_matches),
        "normalized_matches": len(exact_matches_normalized),
        "body_matches": len(body_matches),
        "total_template_matches": len(all_matches),
        "template_match_percentage": (len(all_matches) / len(df)) * 100,
        "official_template_content": official_template,
        "template_matches": []
    }
    
    for idx, (_, row) in enumerate(all_matches.head(50).iterrows(), 1):
        match_info = {
            "rank": idx,
            "modelId": row['modelId'],
            "author": row['author'],
            "likes": int(row['likes']),
            "downloads": int(row['downloads']),
            "card_length": int(row['card_length']),
            "match_type": "exact" if row['card'] == official_template else 
                         "normalized" if normalize_content(row['card']) == official_template_normalized else "body_only"
        }
        results["template_matches"].append(match_info)
    
    with open("official_template_matches.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: official_template_matches.json")
    print(f"Template match rate: {(len(all_matches) / len(df)) * 100:.4f}%")

else:
    print("\nNo cards found that exactly match the official template!")
    print("This suggests the template detection logic is working on different criteria.")