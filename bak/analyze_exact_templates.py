#!/usr/bin/env python
"""
Find cards with exactly identical template content to verify template detection.
Shows value counts of completely duplicate template cards.
"""
import os
import duckdb
import pandas as pd
import json
from collections import Counter

# -------- paths / config --------
RAW_DIR = os.path.expanduser("~/Repo/CitationLake/data/raw")
PARQUET_GLOB = os.path.join(RAW_DIR, "train-*-of-00004.parquet")

# -------- duckdb connection --------
con = duckdb.connect()

# Apply the same filtering as the original script up to deduplication
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

# Get data BEFORE deduplication to see actual template duplicates
print("Loading data before deduplication to find exact template matches...")

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
print(f"Total cards before deduplication: {len(df)}")

# Count exact duplicates by card content
card_counts = df['card'].value_counts()
print(f"Unique card contents: {len(card_counts)}")

# Find cards that appear multiple times (exact duplicates)
duplicate_cards = card_counts[card_counts > 1]
print(f"Cards with exact duplicates: {len(duplicate_cards)}")
print(f"Total instances of duplicate cards: {duplicate_cards.sum()}")

if len(duplicate_cards) > 0:
    print("\nTop 20 most duplicated card contents:")
    top_duplicates = duplicate_cards.head(20)
    
    for idx, (card_content, count) in enumerate(top_duplicates.items(), 1):
        print(f"\n=== Rank {idx}: {count} identical copies ===")
        print(f"Card length: {len(card_content)} characters")
        print(f"First 300 characters:")
        print(repr(card_content[:300]) + "...")
        
        # Show some examples of models with this exact card
        examples = df[df['card'] == card_content].head(5)
        print(f"Example models with this card:")
        for _, row in examples.iterrows():
            print(f"  - {row['modelId']} (author: {row['author']}, likes: {row['likes']}, downloads: {row['downloads']})")

    # Save detailed analysis
    output_file = "exact_template_duplicates.json"
    analysis_results = {
        "total_cards_before_dedup": len(df),
        "unique_card_contents": len(card_counts),
        "cards_with_duplicates": len(duplicate_cards),
        "total_duplicate_instances": int(duplicate_cards.sum()),
        "top_20_duplicates": []
    }
    
    for idx, (card_content, count) in enumerate(top_duplicates.items(), 1):
        examples = df[df['card'] == card_content].head(5)
        duplicate_info = {
            "rank": idx,
            "duplicate_count": int(count),
            "card_length": len(card_content),
            "card_content_preview": card_content[:500],
            "card_content_full": card_content,  # Include full content for analysis
            "example_models": []
        }
        
        for _, row in examples.iterrows():
            duplicate_info["example_models"].append({
                "modelId": row['modelId'],
                "author": row['author'],
                "likes": int(row['likes']),
                "downloads": int(row['downloads'])
            })
        
        analysis_results["top_20_duplicates"].append(duplicate_info)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, indent=2, ensure_ascii=False)
    
    print(f"\nDetailed analysis saved to: {output_file}")
    
else:
    print("No exact duplicate cards found.")

# Additional analysis: Show statistics
print(f"\nDuplication statistics:")
print(f"- Cards appearing exactly once: {len(card_counts[card_counts == 1]):,}")
print(f"- Cards appearing 2-5 times: {len(card_counts[(card_counts >= 2) & (card_counts <= 5)]):,}")
print(f"- Cards appearing 6-10 times: {len(card_counts[(card_counts >= 6) & (card_counts <= 10)]):,}")  
print(f"- Cards appearing >10 times: {len(card_counts[card_counts > 10]):,}")