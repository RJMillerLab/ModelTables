"""
Author: Zhengyuan Dong
Created: 2025-04-03
Last Modified: 2025-04-04
Description: Compute the overlap rate of citation and citing papers based on paperId, and save:
- pairwise overlap scores
- thresholded related paper pairs
- direct citation links (if paperId appears in references of another)
- groundtruth benchmark table paths from related pairs using modelId mapping
"""

import os
import json
import pandas as pd
import pickle
from tqdm import tqdm
from collections import defaultdict

# === Configuration ===
RELATED_PICKLE = "data/processed/modelcard_citation_overlap_by_paperId_related.pickle"
FINAL_INTEGRATION_PARQUET = "data/processed/final_integration_with_paths.parquet"
ALL_TITLE_PATH = "data/processed/modelcard_all_title_list.parquet"
STEP4_PARQUET = "data/processed/modelcard_step4.parquet"
OUTPUT_GT_PATH = "data/gt/scilakeUnionBenchmark_by_ids.pickle"
DISCOUNT_RATE = 0.5
IS_STRICT_MODE = True
USE_SYMLINK = False


def gather_tables_with_discount(row, use_symlink: bool, is_strict: bool, discount_rate: float):
    if use_symlink:
        hugging_col = "hugging_table_list_sym"
        github_col = "github_table_list_sym"
        html_col = "html_table_list_sym"
        llm_col = "llm_table_list_sym"
    else:
        hugging_col = "hugging_table_list"
        github_col = "github_table_list"
        html_col = "html_table_list_mapped"
        llm_col = "llm_table_list_mapped"

    hugging_list = row.get(hugging_col, [])
    github_list = row.get(github_col, [])
    html_list = row.get(html_col, [])
    llm_list = row.get(llm_col, [])

    discount_val = 1.0 if is_strict else discount_rate
    result = {}
    for path in hugging_list:
        result[path] = result.get(path, 0.0) + 1.0
    for path in github_list + html_list + llm_list:
        result[path] = result.get(path, 0.0) + discount_val

    return result


def main():
    print(f"Loading related paperId graph from: {RELATED_PICKLE}")
    with open(RELATED_PICKLE, "rb") as f:
        paperid_relationships = pickle.load(f)

    print(f"Loading metadata and title mapping...")
    df_metadata = pd.read_parquet(FINAL_INTEGRATION_PARQUET, columns=["corpusid", "query"])
    df_titles = pd.read_parquet(ALL_TITLE_PATH, columns=["modelId", "all_title_list"])
    df_titles_exploded = df_titles.explode("all_title_list")

    # title -> modelIds map
    title_to_modelIds = defaultdict(set)
    for _, row in df_titles_exploded.iterrows():
        title = row["all_title_list"]
        if pd.notna(title):
            title_to_modelIds[title.strip()].add(row["modelId"])

    # paperId -> modelIds map
    paperId_to_modelIds = defaultdict(set)
    for _, row in df_metadata.iterrows():
        pid = row.get("corpusid")
        title = row.get("query")
        if pd.notna(pid) and pd.notna(title):
            model_ids = title_to_modelIds.get(title.strip(), set())
            for mid in model_ids:
                paperId_to_modelIds[str(pid)].add(mid)

    print("Loading table paths from step4 metadata...")
    df_step4 = pd.read_parquet(STEP4_PARQUET)
    df_step4_indexed = df_step4.set_index("modelId", drop=False)

    groundtruth_tables = {}

    for pid, related_pids in tqdm(paperid_relationships.items()):
        model_ids = paperId_to_modelIds.get(pid, set())
        related_model_ids = set()
        for rpid in related_pids:
            related_model_ids.update(paperId_to_modelIds.get(rpid, set()))

        all_model_ids = model_ids.union(related_model_ids)

        combined_table = {}
        for mid in all_model_ids:
            if mid not in df_step4_indexed.index:
                continue
            row = df_step4_indexed.loc[mid]
            table_dict = gather_tables_with_discount(
                row,
                use_symlink=USE_SYMLINK,
                is_strict=IS_STRICT_MODE,
                discount_rate=DISCOUNT_RATE
            )
            for path, wt in table_dict.items():
                combined_table[path] = combined_table.get(path, 0.0) + wt

        for mid in model_ids:
            groundtruth_tables[mid] = combined_table

    os.makedirs(os.path.dirname(OUTPUT_GT_PATH), exist_ok=True)
    with open(OUTPUT_GT_PATH, "wb") as f:
        pickle.dump(groundtruth_tables, f)

    print(f"✅ Groundtruth saved to {OUTPUT_GT_PATH}")


if __name__ == "__main__":
    main()
