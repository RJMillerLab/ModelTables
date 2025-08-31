#!/usr/bin/env python
"""hf_models_analysis.py – fast statistics using DuckDB from modellake_pro.db and modelcard_step4.parquet.
Outputs hf_models_analysis.png plus JSON counts to stdout.
COMPLETE VALIDATION: No sampling, checks all ~380k models to ensure models with tables have valid modelcards.
"""
import os, json, re, textwrap
import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

# -------- paths / config --------
DB_PATH = "modellake_pro.duckdb"  # DuckDB database with raw_modelcard
STEP4_PATH = "data/processed/modelcard_step4.parquet"  # Final deduped table data

# -------- duckdb connections --------
# Connect to the main database for raw_modelcard
con_main = duckdb.connect(DB_PATH)
# Connect to a separate connection for parquet files
con_parquet = duckdb.connect()

# helper to run and fetch single value from main DB
def q_main(sql: str) -> int:
    return con_main.execute(sql).fetchone()[0]

# helper to run and fetch single value from parquet
def q_parquet(sql: str) -> int:
    return con_parquet.execute(sql).fetchone()[0]

# ----- counts dict -----
counts = {}

# 1) All models in HF (from raw_modelcard)
print("Counting all models...")
counts["All Models"] = q_main("SELECT COUNT(*) FROM raw_modelcard")

# 2) Models with cards (non-empty, not 'Entry not found')
print("Counting models with valid cards...")
counts["Models w/ Cards"] = q_main("""
    SELECT COUNT(*) FROM raw_modelcard 
    WHERE card IS NOT NULL AND card <> '' AND card <> 'Entry not found'
""")

# 3) Models with tables (from step4 parquet) - ANY source
print("Counting models with tables from any source...")
counts["Models w/ Any Table"] = q_parquet(f"""
    SELECT COUNT(*) FROM read_parquet('{STEP4_PATH}')
    WHERE (hugging_table_list_dedup IS NOT NULL AND array_length(hugging_table_list_dedup) > 0)
       OR (html_table_list_mapped_dedup IS NOT NULL AND array_length(html_table_list_mapped_dedup) > 0)
       OR (llm_table_list_mapped_dedup IS NOT NULL AND array_length(llm_table_list_mapped_dedup) > 0)
       OR (github_table_list_dedup IS NOT NULL AND array_length(github_table_list_dedup) > 0)
""")

# 4) Models with HuggingFace tables specifically
print("Counting models with HuggingFace tables...")
counts["Models w/ Hugging Tables"] = q_parquet(f"""
    SELECT COUNT(*) FROM read_parquet('{STEP4_PATH}')
    WHERE hugging_table_list_dedup IS NOT NULL AND array_length(hugging_table_list_dedup) > 0
""")

# ---------- COMPLETE VALIDATION: Double Check all models with tables ----------
print("\n=== COMPLETE VALIDATION ===")
print("Validating that ALL models with tables have valid modelcards...")

# Get all models with tables and their modelcard status
validation_query = f"""
    WITH models_with_tables AS (
        SELECT DISTINCT modelId
        FROM read_parquet('{STEP4_PATH}')
        WHERE (hugging_table_list_dedup IS NOT NULL AND array_length(hugging_table_list_dedup) > 0)
           OR (html_table_list_mapped_dedup IS NOT NULL AND array_length(html_table_list_mapped_dedup) > 0)
           OR (llm_table_list_mapped_dedup IS NOT NULL AND array_length(llm_table_list_mapped_dedup) > 0)
           OR (github_table_list_dedup IS NOT NULL AND array_length(github_table_list_dedup) > 0)
    )
    SELECT 
        mwt.modelId,
        CASE 
            WHEN rmc.card IS NULL OR rmc.card = '' OR rmc.card = 'Entry not found' 
            THEN 'NO_CARD' 
            ELSE 'HAS_CARD' 
        END as card_status,
        rmc.card as card_content
    FROM models_with_tables mwt
    LEFT JOIN raw_modelcard rmc ON mwt.modelId = rmc.modelId
    ORDER BY mwt.modelId
"""

validation_results = con_main.execute(validation_query).fetchdf()

# Count validation results
models_with_tables_total = len(validation_results)
models_with_tables_and_cards = len(validation_results[validation_results['card_status'] == 'HAS_CARD'])
models_with_tables_no_cards = len(validation_results[validation_results['card_status'] == 'NO_CARD'])

print(f"Total models with tables: {models_with_tables_total:,}")
print(f"Models with tables AND valid cards: {models_with_tables_and_cards:,}")
print(f"Models with tables BUT NO valid cards: {models_with_tables_no_cards:,}")

# Show examples of models with tables but no cards (if any exist)
if models_with_tables_no_cards > 0:
    print(f"\n⚠️  WARNING: Found {models_with_tables_no_cards} models with tables but NO valid modelcards!")
    print("Examples of models with tables but no cards:")
    examples_no_cards = validation_results[validation_results['card_status'] == 'NO_CARD'].head(10)
    for _, row in examples_no_cards.iterrows():
        print(f"  - {row['modelId']}: {row['card_status']}")
else:
    print("✅ SUCCESS: All models with tables have valid modelcards!")

# ---------- CROSS ANALYSIS ----------
print("\n=== CROSS ANALYSIS ===")

# Get models with cards vs without cards
models_with_cards = counts["Models w/ Cards"]
models_without_cards = counts["All Models"] - models_with_cards

print(f"\nModelCards Analysis:")
print(f"Models WITH modelcards: {models_with_cards:,}")
print(f"Models WITHOUT modelcards: {models_without_cards:,}")
print(f"Difference: {models_with_cards - models_without_cards:,}")

# Get models with tables vs without tables
models_with_tables = counts["Models w/ Any Table"]
models_without_tables = counts["All Models"] - models_with_tables

print(f"\nTables Analysis:")
print(f"Models WITH tables: {models_with_tables:,}")
print(f"Models WITHOUT tables: {models_without_tables:,}")
print(f"Difference: {models_with_tables - models_without_tables:,}")

# Cross analysis: models with both cards and tables
models_with_both = models_with_tables_and_cards  # From our validation above

# Models with cards but no tables
models_with_cards_no_tables = models_with_cards - models_with_both

# Models with tables but no cards (this should be 0 if validation passed)
models_with_tables_no_cards = models_with_tables - models_with_both

print(f"\nCross Analysis (ModelCards vs Tables):")
print(f"Models with BOTH cards and tables: {models_with_both:,}")
print(f"Models with cards but NO tables: {models_with_cards_no_tables:,}")
print(f"Models with tables but NO cards: {models_with_tables_no_cards:,}")
print(f"Models with NEITHER: {counts['All Models'] - models_with_cards - models_with_tables + models_with_both:,}")

# ---------- print JSON ----------
print("\n=== MAIN STATISTICS ===")
print("HuggingFace Models Analysis:")
print(json.dumps(counts, indent=2))

# ---------- detailed breakdown ----------
print("\nDetailed breakdown:")
print(f"Total models: {counts['All Models']:,}")
print(f"Models with cards: {counts['Models w/ Cards']:,}")
print(f"Models with tables: {counts['Models w/ Any Table']:,}")
print(f"Models with HuggingFace tables: {counts['Models w/ Hugging Tables']:,}")

# Calculate percentages
total = counts['All Models']
print(f"\nPercentages:")
print(f"Models with cards: {counts['Models w/ Cards']/total*100:.1f}%")
print(f"Models with tables: {counts['Models w/ Any Table']/total*100:.1f}%")
print(f"Models with HuggingFace tables: {counts['Models w/ Hugging Tables']/total*100:.1f}%")

# ---------- table source distribution ----------
print("\nTable source distribution:")
table_sources = con_parquet.execute(f"""
    SELECT 
        COUNT(*) as total_models_with_tables,
        COUNT(CASE WHEN hugging_table_list_dedup IS NOT NULL AND array_length(hugging_table_list_dedup) > 0 THEN 1 END) as hugging_source,
        COUNT(CASE WHEN html_table_list_mapped_dedup IS NOT NULL AND array_length(html_table_list_mapped_dedup) > 0 THEN 1 END) as html_source,
        COUNT(CASE WHEN llm_table_list_mapped_dedup IS NOT NULL AND array_length(llm_table_list_mapped_dedup) > 0 THEN 1 END) as llm_source,
        COUNT(CASE WHEN github_table_list_dedup IS NOT NULL AND array_length(github_table_list_dedup) > 0 THEN 1 END) as github_source
    FROM read_parquet('{STEP4_PATH}')
    WHERE (hugging_table_list_dedup IS NOT NULL AND array_length(hugging_table_list_dedup) > 0)
       OR (html_table_list_mapped_dedup IS NOT NULL AND array_length(html_table_list_mapped_dedup) > 0)
       OR (llm_table_list_mapped_dedup IS NOT NULL AND array_length(llm_table_list_mapped_dedup) > 0)
       OR (github_table_list_dedup IS NOT NULL AND array_length(github_table_list_dedup) > 0)
""").fetchdf()

print(f"Total models with tables: {table_sources.iloc[0]['total_models_with_tables']:,}")
print(f"From HuggingFace cards: {table_sources.iloc[0]['hugging_source']:,}")
print(f"From HTML/PDF: {table_sources.iloc[0]['html_source']:,}")
print(f"From LLM processing: {table_sources.iloc[0]['llm_source']:,}")
print(f"From GitHub READMEs: {table_sources.iloc[0]['github_source']:,}")

# ---------- plot main statistics (4 columns) ----------
plt.figure(figsize=(14, 8))
bars = plt.bar(counts.keys(), counts.values(), color=plt.cm.Blues(np.linspace(0.8, 0.4, len(counts))))

# Put numeric labels on top of each bar (with thousand separators)
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        f"{int(height):,}",
        ha="center",
        va="bottom",
        fontsize=14,
        fontweight='bold'
    )

plt.ylabel("Number of Models", fontsize=16)
plt.xlabel("Step by Step Filtering", fontsize=16)
plt.xticks(rotation=0, ha="center", fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.savefig("hf_models_analysis.pdf", format='pdf', dpi=300, bbox_inches='tight')
plt.savefig("hf_models_analysis.png", format='png', dpi=300, bbox_inches='tight')
print("\nsave fig to hf_models_analysis.pdf and hf_models_analysis.png")
plt.close()

# ---------- plot cross analysis ----------
plt.figure(figsize=(15, 10))

# Create subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Subplot 1: ModelCards vs No ModelCards
labels1 = ['With ModelCards', 'Without ModelCards']
values1 = [models_with_cards, models_without_cards]
colors1 = ['#2ca02c', '#d62728']
bars1 = ax1.bar(labels1, values1, color=colors1)
ax1.set_title('Models with vs without ModelCards', fontsize=14, fontweight='bold')
ax1.set_ylabel('Number of Models', fontsize=12)

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height):,}', ha='center', va='bottom', fontweight='bold')

# Subplot 2: Tables vs No Tables
labels2 = ['With Tables', 'Without Tables']
values2 = [models_with_tables, models_without_tables]
colors2 = ['#1f77b4', '#ff7f0e']
bars2 = ax2.bar(labels2, values2, color=colors2)
ax2.set_title('Models with vs without Tables', fontsize=14, fontweight='bold')
ax2.set_ylabel('Number of Models', fontsize=12)

# Add value labels on bars
for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height):,}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig("hf_cross_analysis.pdf", format='pdf', dpi=300, bbox_inches='tight')
plt.savefig("hf_cross_analysis.png", format='png', dpi=300, bbox_inches='tight')
print("save cross analysis fig to hf_cross_analysis.pdf and hf_cross_analysis.png")
plt.close()

# Close connections
con_main.close()
con_parquet.close()
