#!/usr/bin/env python3
"""Debug the count discrepancy"""
import os
import duckdb

RAW_DIR = os.path.expanduser("~/Repo/CitationLake/data/raw")
PARQUET_GLOB = os.path.join(RAW_DIR, "train-*-of-00004.parquet")

con = duckdb.connect()

print("=== Debugging count discrepancy ===\n")

# My script's first step count
query1 = f"""
SELECT COUNT(*) as count
FROM read_parquet('{PARQUET_GLOB}')
WHERE card IS NOT NULL AND card <> 'Entry not found'
"""
my_count = con.execute(query1).fetchone()[0]
print(f"My script count (card IS NOT NULL AND card <> 'Entry not found'): {my_count:,}")

# Direct verification
query2 = f"""
SELECT COUNT(*) as count  
FROM read_parquet('{PARQUET_GLOB}')
WHERE card <> 'Entry not found'
"""
direct_count = con.execute(query2).fetchone()[0]
print(f"Direct count (card <> 'Entry not found'): {direct_count:,}")

# Check for any NULL cards
query3 = f"""
SELECT COUNT(*) as count
FROM read_parquet('{PARQUET_GLOB}')
WHERE card IS NULL
"""
null_count = con.execute(query3).fetchone()[0]
print(f"NULL cards: {null_count:,}")

# The difference
print(f"Difference: {my_count - direct_count}")

con.close()