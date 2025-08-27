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
import requests

# -------- paths / config --------
RAW_DIR = os.path.expanduser("~/Repo/CitationLake/data/raw")  # parquet shards
# Pattern to match all relevant parquet shards (e.g., train-00000-of-00004.parquet … train-00003-of-00004.parquet)
PARQUET_GLOB = os.path.join(RAW_DIR, "train-*-of-00004.parquet")

# -------- model card template (for "Not template" filtering) --------
# Allow user to set custom path via env var; otherwise try to locate or download
DEFAULT_TEMPLATE_PATH = Path(__file__).with_name("modelcard_template.md")
TEMPLATE_PATH = Path(os.getenv("MODEL_CARD_TEMPLATE_PATH", DEFAULT_TEMPLATE_PATH))

if not TEMPLATE_PATH.exists():
    try:
        print(f"Template file {TEMPLATE_PATH} not found – downloading from HuggingFace Hub …")
        url = (
            "https://raw.githubusercontent.com/huggingface/huggingface_hub/main/"
            "src/huggingface_hub/templates/modelcard_template.md"
        )
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        TEMPLATE_PATH.write_text(resp.text, encoding="utf-8")
        print(f"Saved template to {TEMPLATE_PATH}")
    except Exception as e:
        print(f"Warning: failed to download template: {e}")

# Use a distinctive paragraph from the template (first non-empty line after YAML front-matter)
template_text = ""
if TEMPLATE_PATH.exists():
    raw_lines = TEMPLATE_PATH.read_text(encoding="utf-8").splitlines()
    # Skip YAML header if present (---)
    content_lines = []
    skip = False
    for line in raw_lines:
        if line.strip() == "---":
            skip = not skip  # toggle once at start and end of header
            continue
        if skip:
            continue
        if line.strip():
            content_lines.append(line.strip())
        if len(content_lines) >= 3:
            break
    template_text = " ".join(content_lines).lower()

# Fallback phrase if template not found
if not template_text:
    template_text = "## model description"  # appears in default template

# Escape single quotes for SQL literal
template_sql_literal = template_text.replace("'", "''")
# Build LIKE pattern with wildcards
template_like_pattern = f"%{template_sql_literal}%"

# ---------- validity condition ----------
VALID_CARD_COND = "card IS NOT NULL AND card <> '' AND card <> 'Entry not found'"

# -------- duckdb connection --------
con = duckdb.connect()

# helper to run and fetch single value
def q(sql: str) -> int:
    return con.execute(sql).fetchone()[0]

# base where clause (exclude entry not found)
BASE = f"FROM read_parquet('{PARQUET_GLOB}')"

# check columns
# Retrieve column names by executing a zero-row SELECT (faster than loading full data)
cols = set(
    con.execute(f"SELECT * FROM read_parquet('{PARQUET_GLOB}') LIMIT 0").fetchdf().columns
)

# ----- counts dict -----
counts = {}

# -1) total repos
counts["All repositories"] = q(f"SELECT COUNT(*) {BASE}")

# 0) invalid card (Entry not found)
invalid_cond = "card = 'Entry not found'"
counts["Invalid card"] = q(f"SELECT COUNT(*) {BASE} WHERE {invalid_cond}")

# 1) non-empty card (valid)
counts["Non-empty card"] = q(f"SELECT COUNT(*) {BASE} WHERE {VALID_CARD_COND}")

# 2) unique card – keep row with highest likes per card
DEDUP_QUERY = f"""
    SELECT * FROM read_parquet('{PARQUET_GLOB}')
    WHERE {VALID_CARD_COND}
    QUALIFY row_number() OVER (PARTITION BY card ORDER BY likes DESC) = 1
"""
counts["Unique card"] = q(f"SELECT COUNT(*) FROM ({DEDUP_QUERY})")

# 3) not template (card does NOT contain template snippet)
non_template_cond = f"lower(card) NOT LIKE '{template_like_pattern}'"
counts["Not template"] = q(
    f"SELECT COUNT(*) FROM ({DEDUP_QUERY}) WHERE {non_template_cond}"
)

# 4) downloads > 0 (after previous filters)
counts["downloads > 0"] = q(
    f"SELECT COUNT(*) FROM ({DEDUP_QUERY}) WHERE {non_template_cond} AND downloads > 0"
)

# ---------- print JSON ----------
print(json.dumps(counts, indent=2))

# ---------- preview first two records ----------
sample_df = con.execute(f"SELECT * {BASE} LIMIT 2").fetchdf()
print("\nPreview of first two rows (truncated):")
with pd.option_context('display.max_columns', None, 'display.max_colwidth', 120):
    print(sample_df)

# ---------- plot ----------
plt.figure(figsize=(9, 5))
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
        fontsize=9,
    )
plt.ylabel("# Models")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig("step_by_step_filtering_statistics.png")
print("save fig to step_by_step_filtering_statistics.png")
plt.close()
