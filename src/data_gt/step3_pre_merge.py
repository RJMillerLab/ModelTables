"""
Author: Zhengyuan Dong
Created: 2025-03-11
Last Modified: 2025-04-02
Description: Merge tables from df2 and df based on query and title.
Usage:
    python -m src.data_gt.step3_pre_merge
"""

import ast  ########
import os
import json
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

FINAL_INTEGRATION_PARQUET   = "data/processed/final_integration_with_paths.parquet"
ALL_TITLE_PATH              = "data/processed/modelcard_all_title_list.parquet"
MERGE_PATH                  = "data/processed/modelcard_step3_merged.parquet"
SIDE_PATH                   = "data/processed/modelcard_step2.parquet" 

def _combine_lists(series):
    """
    Helper to combine lists while dropping NaN/None.
    """
    all_items = []
    for x in series.dropna():
        if isinstance(x, (list, tuple, np.ndarray)):
            all_items.extend(x)
        else:
            pass
    return list(set(all_items))

def _safe_parse_list(val):
    """Parse list-like strings into actual list, or return as-is."""
    if isinstance(val, str) and val.startswith("[") and val.endswith("]"):
        try:
            parsed = ast.literal_eval(val.replace('\n', '').replace('\r', ''))
            if isinstance(parsed, (list, tuple, np.ndarray)):
                return list(parsed)
        except Exception:
            return []
    elif isinstance(val, (list, tuple, np.ndarray)):
        return list(val)
    else:
        return []

def populate_hugging_table_list(df_merged, processed_base_path):
    """
    Populate 'hugging_table_list' using 'hugging_deduped_mapping.json'
    """
    hugging_map_json_path = os.path.join(processed_base_path, "hugging_deduped_mapping.json")
    with open(hugging_map_json_path, 'r', encoding='utf-8') as jf:
        hash_to_csv_map = json.load(jf)
    df_merged['hugging_table_list'] = [[] for _ in range(len(df_merged))]
    for i, row in tqdm(df_merged.iterrows(), total=len(df_merged), desc="Populating Hugging table list"):
        hval = row['readme_hash']
        if not isinstance(hval, str):
            continue
        deduped_csv_list = hash_to_csv_map.get(hval, [])
        df_merged.at[i, 'hugging_table_list'] = [p[p.index("data/processed/"):] if "data/processed/" in p else p 
                                                  for p in deduped_csv_list]
    return df_merged

def populate_github_table_list(df_merged, processed_base_path):
    """
    Populate 'github_table_list' using 'md_to_csv_mapping.json'
    """
    with open(os.path.join(processed_base_path, "deduped_github_csvs", "md_to_csv_mapping.json"), 'r', encoding='utf-8') as jf:
        md_to_csv_mapping = json.load(jf)
    df_merged['github_table_list'] = [[] for _ in range(len(df_merged))]
    for i, row in tqdm(df_merged.iterrows(), total=len(df_merged), desc="Populating GitHub table list"):
        readme_paths = row['readme_path']
        if isinstance(readme_paths, str):
            readme_paths = [readme_paths]
        combined_csvs = []
        for md_file in readme_paths:
            md_basename = os.path.basename(md_file).replace(".md", "")
            value = md_to_csv_mapping.get(md_basename)
            if value not in [None, []]:
                combined_csvs.extend(value)
        combined_csvs = list(set(combined_csvs))
        full_csv_paths = []
        for csv_basename in combined_csvs:
            csv_full_path = os.path.join(processed_base_path, "deduped_github_csvs", csv_basename)
            full_csv_paths.append(csv_full_path)
        df_merged.at[i, "github_table_list"] = full_csv_paths
    return df_merged

def map_tables_by_dict(df2, df):
    """
    Accelerate the process by using a dictionary mapping:
    1) Build a mapping from query to (html_table_list, llm_table_list)
    2) For each row in df2, look up and merge results from all_title_list
    """
    # turn comment into english
    # Turn stringified lists into actual lists
    df['html_table_list'] = df['html_table_list'].apply(_safe_parse_list)
    df['llm_table_list']  = df['llm_table_list'].apply(_safe_parse_list)
    # Build lookup: dict[query] = (html_table_list, llm_table_list)
    df_lookup = {}
    for row in df.itertuples(index=False):
        # row: query, html_table_list, llm_table_list
        df_lookup[row.query] = (
            row.html_table_list if isinstance(row.html_table_list, list) else [],
            row.llm_table_list  if isinstance(row.llm_table_list,  list) else []
        )
    # add col to df2：html_table_list_mapped, llm_table_list_mapped
    df2["html_table_list_mapped"] = [[] for _ in range(len(df2))]
    df2["llm_table_list_mapped"]  = [[] for _ in range(len(df2))]
    # loop through df2, for each row, check all_title_list
    for i, row in df2.iterrows():
        title_list = row["all_title_list"]
        if not isinstance(title_list, list):
            continue
        combined_html = []
        combined_llm  = []
        for title in title_list:
            if title in df_lookup:
                hlist, llist = df_lookup[title]
                combined_html.extend(hlist)
                combined_llm.extend(llist)
        # deduplicate
        combined_html = list(set(combined_html))
        combined_llm  = list(set(combined_llm))
        df2.at[i, "html_table_list_mapped"] = combined_html
        df2.at[i, "llm_table_list_mapped"]  = combined_llm
    return df2

def merge_table_list_to_df2():
    df = pd.read_parquet(FINAL_INTEGRATION_PARQUET, columns=['query', 'html_table_list', 'llm_table_list']) # , 'corpusid'
    print(f"  df loaded with shape: {df.shape}")

    # Clean stringified lists ########
    df['html_table_list'] = df['html_table_list'].apply(_safe_parse_list)  ########
    df['llm_table_list'] = df['llm_table_list'].apply(_safe_parse_list)  ########

    df2 = pd.read_parquet(ALL_TITLE_PATH)
    print(f"  df2 loaded with shape: {df2.shape}")
    print("\nStep 1: Expanding df2 to match df (on df2.all_title_list vs df.query)...")

    mode = 'normal' # accelerated, normal
    if mode=='accelerated':
        df2_merged = map_tables_by_dict(df2, df)
    else:
        all_title_list_key = "all_title_list"
        df2[all_title_list_key] = df2[all_title_list_key].apply(lambda x: list(dict.fromkeys(x)) if isinstance(x, (list, tuple, np.ndarray)) else x)
        df2_exploded = df2.explode(all_title_list_key).rename(columns={all_title_list_key: 'explode_title'})
        merged = pd.merge(
            df2_exploded,
            df,
            how='left',
            left_on='explode_title',
            right_on='query'
        )
        print("Step 2: Grouping & assembling lists back by modelId...")
        grouped = merged.groupby('modelId').agg({
            'html_table_list': lambda x: _combine_lists(x),
            'llm_table_list': lambda x: _combine_lists(x),
        }).reset_index()
        grouped.rename(columns={
            'html_table_list': 'html_table_list_mapped',
            'llm_table_list': 'llm_table_list_mapped'
        }, inplace=True)

        print("Step 3: Merging the grouped columns back into df2...")
        df2_merged = pd.merge(df2, grouped, on='modelId', how='left')
        df2_merged['html_table_list_mapped'] = df2_merged['html_table_list_mapped'].apply(
            lambda v: v if isinstance(v, (list, tuple, np.ndarray)) else []
        )
        df2_merged['llm_table_list_mapped'] = df2_merged['llm_table_list_mapped'].apply(
            lambda v: v if isinstance(v, (list, tuple, np.ndarray)) else []
        )
    # load side data and merge to df with modelId
    side_df = pd.read_parquet(SIDE_PATH, columns=['modelId', 'readme_path', 'readme_hash'])
    side_df = populate_hugging_table_list(side_df, os.path.dirname(SIDE_PATH))
    side_df = populate_github_table_list(side_df, os.path.dirname(SIDE_PATH))
    df_final = pd.merge(df2_merged, side_df[['modelId', 'github_table_list', 'hugging_table_list']], on='modelId', how='left')
    df_final.to_parquet(MERGE_PATH, compression='zstd', engine='pyarrow', index=False)
    return df_final

if __name__ == "__main__":
    print(f"Merging all tables list...")
    merge_table_list_to_df2()
    print(f"\n🎉 All tables merged and saved to {MERGE_PATH}.")
