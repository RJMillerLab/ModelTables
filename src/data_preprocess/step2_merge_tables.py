"""
Author: Zhengyuan Dong
Created: 2025-03-11
Last Modified: 2025-09-21
Description: Merge tables from df2 and df based on query and title.
Usage:
    python -m src.data_preprocess.step2_merge_tables
"""

import ast  ########
import os
import argparse
import json
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from src.utils import to_parquet, load_config, is_list_like, to_list_safe

FINAL_INTEGRATION_PARQUET   = "data/processed/final_integration_with_paths_v2.parquet"
ALL_TITLE_PATH              = "data/processed/modelcard_all_title_list.parquet"
MERGE_PATH                  = "data/processed/modelcard_step3_merged_v2.parquet"
SIDE_PATH                   = "data/processed/modelcard_step2_v2.parquet"  # v1

def _combine_lists(series):
    """
    Helper to combine lists while dropping NaN/None.
    """
    all_items = []
    for x in series.dropna():
        if is_list_like(x):
            all_items.extend(to_list_safe(x))
        else:
            pass
    return list(set(all_items))

def _safe_parse_list(val):
    """Parse list-like strings into actual list, or return as-is."""
    if isinstance(val, str) and val.startswith("[") and val.endswith("]"):
        try:
            parsed = ast.literal_eval(val.replace('\n', '').replace('\r', ''))
            if is_list_like(parsed):
                return to_list_safe(parsed)
        except Exception:
            return []
    elif is_list_like(val):
        return to_list_safe(val)
    else:
        return []

def populate_hugging_table_list(df_merged, processed_base_path, tag=None):
    """
    Populate 'hugging_table_list' using 'hugging_deduped_mapping.json' (v1) or 'hugging_deduped_mapping_v2.json' (v2)
    """
    suffix = f"_{tag}" if tag else ""
    # Try v2 with tag first, then v2 without tag, then v1
    hugging_map_json_path_v2_tag = os.path.join(processed_base_path, f"hugging_deduped_mapping_v2{suffix}.json")
    hugging_map_json_path_v2 = os.path.join(processed_base_path, "hugging_deduped_mapping_v2.json")
    hugging_map_json_path_v1 = os.path.join(processed_base_path, "hugging_deduped_mapping.json")
    
    if tag and os.path.exists(hugging_map_json_path_v2_tag):
        print(f"📦 Using HuggingFace mapping v2 with tag: {hugging_map_json_path_v2_tag}")
        hugging_map_json_path = hugging_map_json_path_v2_tag
    elif os.path.exists(hugging_map_json_path_v2):
        print(f"📦 Using HuggingFace mapping v2: {hugging_map_json_path_v2}")
        hugging_map_json_path = hugging_map_json_path_v2
    else:
        print(f"📦 Using HuggingFace mapping v1: {hugging_map_json_path_v1}")
        hugging_map_json_path = hugging_map_json_path_v1
    
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

def populate_github_table_list(df_merged, processed_base_path, tag=None):
    """
    Populate 'github_table_list' using 'md_to_csv_mapping.json' (v1) or v2 version
    """
    suffix = f"_{tag}" if tag else ""
    # Try v2 with tag first, then v2 without tag, then v1
    github_csvs_v2_tag = os.path.join(processed_base_path, f"deduped_github_csvs_v2{suffix}")
    github_csvs_v2 = os.path.join(processed_base_path, "deduped_github_csvs_v2")
    github_csvs_v1 = os.path.join(processed_base_path, "deduped_github_csvs")
    github_mapping_v2_tag = os.path.join(github_csvs_v2_tag, "md_to_csv_mapping.json")
    github_mapping_v2 = os.path.join(github_csvs_v2, "md_to_csv_mapping.json")
    github_mapping_v1 = os.path.join(github_csvs_v1, "md_to_csv_mapping.json")
    
    if tag and os.path.exists(github_mapping_v2_tag):
        print(f"📦 Using GitHub mapping v2 with tag: {github_mapping_v2_tag}")
        github_csvs_folder = github_csvs_v2_tag
        github_mapping_path = github_mapping_v2_tag
    elif os.path.exists(github_mapping_v2):
        print(f"📦 Using GitHub mapping v2: {github_mapping_v2}")
        github_csvs_folder = github_csvs_v2
        github_mapping_path = github_mapping_v2
    else:
        print(f"📦 Using GitHub mapping v1: {github_mapping_v1}")
        github_csvs_folder = github_csvs_v1
        github_mapping_path = github_mapping_v1
    
    with open(github_mapping_path, 'r', encoding='utf-8') as jf:
        md_to_csv_mapping = json.load(jf)
    df_merged['github_table_list'] = [[] for _ in range(len(df_merged))]
    for i, row in tqdm(df_merged.iterrows(), total=len(df_merged), desc="Populating GitHub table list"):
        readme_paths = row['readme_path']
        # Handle different types: str, list, tuple, numpy.ndarray, or None
        # Check numpy array first to avoid ValueError with pd.isna() on empty arrays
        if pd.isna(readme_paths):
            readme_paths = []
        elif isinstance(readme_paths, str):
            readme_paths = [readme_paths]
        elif is_list_like(readme_paths):
            readme_paths = to_list_safe(readme_paths)
        else:
            readme_paths = []
        combined_csvs = []
        for md_file in readme_paths:
            if not md_file or not isinstance(md_file, str):
                continue
            md_basename = os.path.basename(md_file).replace(".md", "")
            value = md_to_csv_mapping.get(md_basename)
            if value not in [None, []]:
                combined_csvs.extend(value)
        combined_csvs = list(set(combined_csvs))
        full_csv_paths = []
        for csv_basename in combined_csvs:
            csv_full_path = os.path.join(github_csvs_folder, csv_basename)
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
        # Handle both list and numpy.ndarray types
        html_list = to_list_safe(row.html_table_list) if is_list_like(row.html_table_list) else []
        llm_list = to_list_safe(row.llm_table_list) if is_list_like(row.llm_table_list) else []
        df_lookup[row.query] = (html_list, llm_list)
    # add col to df2：html_table_list_mapped, llm_table_list_mapped
    df2["html_table_list_mapped"] = [[] for _ in range(len(df2))]
    df2["llm_table_list_mapped"]  = [[] for _ in range(len(df2))]
    # loop through df2, for each row, check all_title_list
    for i, row in df2.iterrows():
        title_list = row["all_title_list"]
        if not is_list_like(title_list):
            continue
        title_list = to_list_safe(title_list)
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

def merge_table_list_to_df2(final_integration_path, all_title_path, merge_path, side_path, tag=None):
    df = pd.read_parquet(final_integration_path, columns=['query', 'html_table_list', 'llm_table_list']) # , 'corpusid'
    print(f"  df loaded with shape: {df.shape}")

    # Clean stringified lists ########
    df['html_table_list'] = df['html_table_list'].apply(_safe_parse_list)  ########
    df['llm_table_list'] = df['llm_table_list'].apply(_safe_parse_list)  ########

    df2 = pd.read_parquet(all_title_path, columns=['modelId', 'all_title_list'])
    print(f"  df2 loaded with shape: {df2.shape}")
    print("\nStep 1: Expanding df2 to match df (on df2.all_title_list vs df.query)...")

    mode = 'normal' # accelerated, normal
    if mode=='accelerated':
        df2_merged = map_tables_by_dict(df2, df)
    else:
        all_title_list_key = "all_title_list"
        df2[all_title_list_key] = df2[all_title_list_key].apply(lambda x: list(dict.fromkeys(to_list_safe(x))) if is_list_like(x) else x)
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
            lambda v: to_list_safe(v) if is_list_like(v) else []
        )
        df2_merged['llm_table_list_mapped'] = df2_merged['llm_table_list_mapped'].apply(
            lambda v: to_list_safe(v) if is_list_like(v) else []
        )
    # load side data and merge to df with modelId
    # Try v2 first, fallback to v1
    print(f"📦 Loading side data from: {side_path}")
    side_df = pd.read_parquet(side_path, columns=['modelId', 'readme_path', 'readme_hash'])
    processed_base_path = os.path.dirname(side_path)
    side_df = populate_hugging_table_list(side_df, processed_base_path, tag=tag)
    side_df = populate_github_table_list(side_df, processed_base_path, tag=tag)
    df_final = pd.merge(df2_merged, side_df[['modelId', 'github_table_list', 'hugging_table_list']], on='modelId', how='left')
    df_final.drop(columns=['card_tags', 'downloads', 'github_link', 'pdf_link'], inplace=True, errors='ignore')
    to_parquet(df_final, merge_path)
    return df_final

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge all table lists into a unified model ID file")
    parser.add_argument('--tag', dest='tag', default=None,
                        help='Tag suffix for versioning (e.g., 251117). Enables versioning mode.')
    parser.add_argument('--input-integration', dest='input_integration', default=None,
                        help='Path to final_integration_with_paths parquet (default: auto-detect from tag)')
    parser.add_argument('--input-title-list', dest='input_title_list', default=None,
                        help='Path to modelcard_all_title_list parquet (default: auto-detect from tag)')
    parser.add_argument('--input-step2', dest='input_step2', default=None,
                        help='Path to modelcard_step2 parquet (default: auto-detect from tag)')
    parser.add_argument('--output', dest='output', default=None,
                        help='Path to modelcard_step3_merged parquet (default: auto-detect from tag)')
    args = parser.parse_args()
    
    config = load_config('config.yaml')
    base_path = config.get('base_path', 'data')
    processed_base_path = os.path.join(base_path, 'processed')
    tag = args.tag
    suffix = f"_{tag}" if tag else ""
    
    # Determine input/output paths based on tag
    final_integration_path = args.input_integration or os.path.join(processed_base_path, f"final_integration_with_paths_v2{suffix}.parquet")
    all_title_path = args.input_title_list or os.path.join(processed_base_path, f"modelcard_all_title_list{suffix}.parquet")
    side_path = args.input_step2 or os.path.join(processed_base_path, f"modelcard_step2_v2{suffix}.parquet")
    merge_path = args.output or os.path.join(processed_base_path, f"modelcard_step3_merged_v2{suffix}.parquet")
    
    print("📁 Paths in use:")
    print(f"   Final integration:  {final_integration_path}")
    print(f"   All title list:      {all_title_path}")
    print(f"   Step2 side data:     {side_path}")
    print(f"   Output merged:       {merge_path}")
    
    print(f"\nMerging all tables list...")
    merge_table_list_to_df2(final_integration_path, all_title_path, merge_path, side_path, tag=tag)
    print(f"\n🎉 All tables merged and saved to {merge_path}.")
