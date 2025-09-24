#!/usr/bin/env python
"""
Debug SQL query for template matching
"""
import os
import duckdb
from pathlib import Path

# -------- paths / config --------
RAW_DIR = os.path.expanduser("~/Repo/CitationLake/data/raw")
PARQUET_GLOB = os.path.join(RAW_DIR, "train-*-of-00004.parquet")

# Load template
TEMPLATE_PATH = Path("modelcard_template.md")
official_template = TEMPLATE_PATH.read_text(encoding="utf-8")

# Extract template body
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

# -------- duckdb connection --------
con = duckdb.connect()

# Prepare the query
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

DEDUP_QUERY = f"""
    SELECT * FROM ({no_policy_query})
    QUALIFY row_number() OVER (PARTITION BY card ORDER BY likes DESC) = 1
"""

# Escape quotes for SQL
template_body_sql = template_body_only.replace("'", "''")

print(f"Template body length: {len(template_body_only)}")
print(f"SQL escaped template length: {len(template_body_sql)}")

# Test the specific models we know should match
test_query = f"""
    SELECT modelId, author, 
           length(card) as card_len,
           CASE 
               WHEN card LIKE '---\\n%' AND position('\\n---\\n' IN card) > 0 THEN
                   trim(substring(card FROM position('\\n---\\n' IN card) + 5))
               ELSE
                   trim(card)
           END as extracted_body,
           length(CASE 
               WHEN card LIKE '---\\n%' AND position('\\n---\\n' IN card) > 0 THEN
                   trim(substring(card FROM position('\\n---\\n' IN card) + 5))
               ELSE
                   trim(card)
           END) as body_len
    FROM ({DEDUP_QUERY}) 
    WHERE modelId IN ('DackJan/minigpt-test', 'WenjunJi/Hello')
"""

print("\nTesting specific models:")
results = con.execute(test_query).fetchall()

for row in results:
    model_id, author, card_len, extracted_body, body_len = row
    print(f"\n{model_id}:")
    print(f"  Full card length: {card_len}")
    print(f"  Extracted body length: {body_len}")
    print(f"  Template body length: {len(template_body_only)}")
    print(f"  Match: {extracted_body == template_body_only}")
    
    if extracted_body != template_body_only:
        # Find first difference
        min_len = min(len(extracted_body), len(template_body_only))
        for i in range(min_len):
            if extracted_body[i] != template_body_only[i]:
                print(f"  First diff at pos {i}: {repr(extracted_body[i])} vs {repr(template_body_only[i])}")
                break

# Try the actual SQL matching query
print(f"\nTesting SQL equality condition...")
sql_match_query = f"""
    SELECT modelId, author,
           CASE 
               WHEN card LIKE '---\\n%' AND position('\\n---\\n' IN card) > 0 THEN
                   trim(substring(card FROM position('\\n---\\n' IN card) + 5))
               ELSE
                   trim(card)
           END = '{template_body_sql}' as is_match
    FROM ({DEDUP_QUERY}) 
    WHERE modelId IN ('DackJan/minigpt-test', 'WenjunJi/Hello')
"""

results = con.execute(sql_match_query).fetchall()
for row in results:
    print(f"{row[0]}: SQL match = {row[2]}")

print(f"\nActual count query:")
count_query = f"""
    SELECT COUNT(*) FROM ({DEDUP_QUERY}) 
    WHERE 
        CASE 
            WHEN card LIKE '---\\n%' AND position('\\n---\\n' IN card) > 0 THEN
                trim(substring(card FROM position('\\n---\\n' IN card) + 5))
            ELSE
                trim(card)
        END = '{template_body_sql}'
"""

count_result = con.execute(count_query).fetchone()[0]
print(f"Total matches found: {count_result}")