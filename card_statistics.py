#!/usr/bin/env python
"""card_statistics.py – fast step-by-step stats using DuckDB.
Outputs step_by_step_filtering_statistics.png plus JSON counts to stdout.
"""
import os, json, re, textwrap
import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

# -------- paths / config --------
RAW_DIR = os.path.expanduser("~/Repo/CitationLake/data/raw")  # parquet shards
# Pattern to match all relevant parquet shards (e.g., train-00000-of-00004.parquet … train-00003-of-00004.parquet)
PARQUET_GLOB = os.path.join(RAW_DIR, "train-*-of-00004.parquet")


# ---------- validity condition ----------
VALID_CARD_COND = "card IS NOT NULL AND card <> '' AND card <> 'Entry not found'"

# -------- duckdb connection --------
con = duckdb.connect()

# helper to run and fetch single value
def q(sql: str) -> int:
    return con.execute(sql).fetchone()[0]

# base where clause (exclude entry not found)
BASE = f"FROM read_parquet('{PARQUET_GLOB}') WHERE {VALID_CARD_COND}"

# check columns
# Retrieve column names by executing a zero-row SELECT (faster than loading full data)
cols = set(
    con.execute(f"SELECT * FROM read_parquet('{PARQUET_GLOB}') LIMIT 0").fetchdf().columns
)

# ----- counts dict -----
counts = {}

# 1) all repositories
counts["All"] = q(f"SELECT COUNT(*) FROM read_parquet('{PARQUET_GLOB}')")

# 2) non-empty model cards (after filtering out 'Entry not found')
counts["Non-empty model cards"] = q(f"SELECT COUNT(*) {BASE}")

# 3) unique model cards – keep row with highest likes per card  
DEDUP_QUERY = f"""
    SELECT * {BASE}
    QUALIFY row_number() OVER (PARTITION BY card ORDER BY likes DESC) = 1
"""
counts["Unique model cards"] = q(f"SELECT COUNT(*) FROM ({DEDUP_QUERY})")

# 4) downloads > 0
counts["Downloads > 0"] = q(
    f"SELECT COUNT(*) FROM ({DEDUP_QUERY}) WHERE downloads > 0"
)

# ---------- print JSON ----------
print(json.dumps(counts, indent=2))

# ---------- preview first two records ----------
sample_df = con.execute(f"SELECT * {BASE} LIMIT 2").fetchdf()
print("\nPreview of first two rows (truncated):")
with pd.option_context('display.max_columns', None, 'display.max_colwidth', 120):
    print(sample_df)

# ---------- plot ----------
plt.figure(figsize=(12, 8))
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
    )
plt.ylabel("Number of Model Repositories", fontsize=16)
plt.xlabel("Filtering Steps", fontsize=16)
plt.xticks(rotation=20, ha="right", fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.savefig("step_by_step_filtering_statistics.pdf", format='pdf', dpi=300, bbox_inches='tight')
print("save fig to step_by_step_filtering_statistics.pdf")
plt.close()
