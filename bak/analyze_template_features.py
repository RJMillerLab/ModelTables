#!/usr/bin/env python
"""
Analyze template feature distribution and examine top template cards
to verify if they truly contain only template information.
"""
import os
import duckdb
import pandas as pd
import json

# -------- paths / config --------
RAW_DIR = os.path.expanduser("~/Repo/CitationLake/data/raw")
PARQUET_GLOB = os.path.join(RAW_DIR, "train-*-of-00004.parquet")

# -------- duckdb connection --------
con = duckdb.connect()

# Template features from original script
template_features = [
    "[More Information Needed]",
    "<!-- Provide a quick summary of what the model is/does. -->", 
    "<!-- Provide a longer summary of what this model is. -->",
    "Use the code below to get started with the model",
    "Carbon emissions can be estimated using the [Machine Learning Impact calculator]"
]

# First, apply the same filtering as the original script up to deduplication
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

# Deduplicated data
dedup_query = f"""
    SELECT * FROM ({no_policy_query})
    QUALIFY row_number() OVER (PARTITION BY card ORDER BY likes DESC) = 1
"""

# Create feature scoring query
feature_checks = []
for i, feature in enumerate(template_features):
    feature_sql = feature.replace("'", "''")
    feature_checks.append(f"CASE WHEN card LIKE '%{feature_sql}%' THEN 1 ELSE 0 END AS feature_{i}")

feature_select = ", ".join(feature_checks)
template_score_expr = " + ".join([f"feature_{i}" for i in range(len(template_features))])

# Query to get all cards with their template feature scores
analysis_query = f"""
SELECT 
    modelId,
    author,
    card,
    likes,
    downloads,
    {feature_select},
    ({template_score_expr}) as template_score
FROM ({dedup_query})
ORDER BY template_score DESC, likes DESC
"""

print("Loading data and calculating template scores...")
df = con.execute(analysis_query).fetchdf()

print(f"\nTotal deduplicated cards: {len(df)}")

# Analyze template score distribution
score_counts = df['template_score'].value_counts().sort_index()
print("\nTemplate score distribution:")
for score, count in score_counts.items():
    print(f"  Score {score}: {count:,} cards")

# Analyze individual feature frequency
print("\nIndividual feature frequencies:")
for i, feature in enumerate(template_features):
    feature_count = df[f'feature_{i}'].sum()
    print(f"  Feature {i}: {feature_count:,} cards - '{feature[:50]}...'")

# Get cards with high template scores (likely templates)
high_score_cards = df[df['template_score'] >= 3].copy()
print(f"\nCards with template score >= 3: {len(high_score_cards)}")

if len(high_score_cards) > 0:
    # Show top 10 by score and likes
    top_templates = high_score_cards.head(10)
    
    print("\nTop 10 template cards:")
    for idx, row in top_templates.iterrows():
        print(f"\nRank {idx+1}:")
        print(f"  ModelID: {row['modelId']}")
        print(f"  Author: {row['author']}")
        print(f"  Template Score: {row['template_score']}")
        print(f"  Likes: {row['likes']}")
        print(f"  Downloads: {row['downloads']}")
        print(f"  Card length: {len(row['card'])} characters")
        print(f"  Card preview (first 200 chars): {row['card'][:200]}...")
        
        # Show which features matched
        matched_features = []
        for i in range(len(template_features)):
            if row[f'feature_{i}'] == 1:
                matched_features.append(f"'{template_features[i][:30]}...'")
        print(f"  Matched features: {', '.join(matched_features)}")

# Save detailed analysis
output_file = "template_feature_analysis.json"
analysis_results = {
    "total_cards": len(df),
    "score_distribution": score_counts.to_dict(),
    "feature_frequencies": {
        template_features[i]: int(df[f'feature_{i}'].sum()) 
        for i in range(len(template_features))
    },
    "high_score_cards_count": len(high_score_cards),
    "top_10_template_cards": []
}

if len(high_score_cards) > 0:
    for idx, row in high_score_cards.head(10).iterrows():
        card_info = {
            "rank": idx + 1,
            "modelId": row['modelId'],
            "author": row['author'],
            "template_score": int(row['template_score']),
            "likes": int(row['likes']),
            "downloads": int(row['downloads']),
            "card_length": len(row['card']),
            "card_preview": row['card'][:500],
            "matched_features": [
                template_features[i] for i in range(len(template_features))
                if row[f'feature_{i}'] == 1
            ]
        }
        analysis_results["top_10_template_cards"].append(card_info)

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(analysis_results, f, indent=2, ensure_ascii=False)

print(f"\nDetailed analysis saved to: {output_file}")