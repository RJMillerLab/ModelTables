#!/usr/bin/env python3
"""Analyze card content patterns for empty and template detection"""
import os
import duckdb
import pandas as pd
from collections import Counter

RAW_DIR = os.path.expanduser("~/Repo/CitationLake/data/raw")
PARQUET_GLOB = os.path.join(RAW_DIR, "train-*-of-00004.parquet")

con = duckdb.connect()

print("=== Analyzing card content patterns ===\n")

# 1. Check for empty-like patterns
print("1. Empty-like card patterns:")
empty_query = f"""
SELECT 
    CASE 
        WHEN card IS NULL THEN 'NULL'
        WHEN card = '' THEN 'empty_string'
        WHEN card = 'None' THEN 'string_None'
        WHEN card = 'none' THEN 'string_none'  
        WHEN card = 'Null' THEN 'string_Null'
        WHEN card = 'null' THEN 'string_null'
        WHEN card = 'N/A' THEN 'string_N/A'
        WHEN card = 'n/a' THEN 'string_n/a'
        WHEN trim(card) = '' THEN 'whitespace_only'
        WHEN length(card) <= 3 THEN 'very_short'
        ELSE 'other'
    END as pattern,
    COUNT(*) as count
FROM read_parquet('{PARQUET_GLOB}')
WHERE card IS NULL 
   OR card = '' 
   OR card = 'None' 
   OR card = 'none'
   OR card = 'Null' 
   OR card = 'null'
   OR card = 'N/A'
   OR card = 'n/a'
   OR trim(card) = ''
   OR length(card) <= 3
GROUP BY pattern
ORDER BY count DESC
"""

empty_result = con.execute(empty_query).fetchdf()
print(empty_result.to_string(index=False))

# 2. Sample very short cards (potential empty indicators)
print("\n2. Sample of short card content (length <= 20):")
short_query = f"""
SELECT card, length(card) as len, COUNT(*) as count
FROM read_parquet('{PARQUET_GLOB}')
WHERE card IS NOT NULL 
  AND card <> 'Entry not found'
  AND length(card) <= 20
GROUP BY card, length(card)
ORDER BY count DESC, len
LIMIT 15
"""

short_result = con.execute(short_query).fetchdf()
print(short_result.to_string(index=False))

# 3. Check 'Entry not found' pattern
print("\n3. 'Entry not found' pattern count:")
invalid_query = f"""
SELECT COUNT(*) as entry_not_found_count
FROM read_parquet('{PARQUET_GLOB}')
WHERE card = 'Entry not found'
"""

invalid_result = con.execute(invalid_query).fetchdf()
print(f"'Entry not found' count: {invalid_result.iloc[0, 0]}")

# 4. Sample template-like content (first 200 chars of common patterns)
print("\n4. Sample potential template patterns:")
template_query = f"""
SELECT 
    substring(card, 1, 100) as card_start,
    COUNT(*) as count
FROM read_parquet('{PARQUET_GLOB}')
WHERE card IS NOT NULL 
  AND card <> 'Entry not found'
  AND length(card) > 50
  AND (
    lower(card) LIKE '%model description%'
    OR lower(card) LIKE '%this model%'
    OR lower(card) LIKE '%---\ntags:%'
    OR lower(card) LIKE '%license:%'
  )
GROUP BY card_start
ORDER BY count DESC
LIMIT 10
"""

template_result = con.execute(template_query).fetchdf()
print(template_result.to_string(index=False))

con.close()