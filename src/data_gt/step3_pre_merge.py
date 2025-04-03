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
MERGE_PATH                  = "data/processed/modelcard_step3_merged.parquet" ########

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

def _safe_parse_list(val):  ########
    """Parse list-like strings into actual list, or return as-is."""
    if isinstance(val, str) and val.startswith("[") and val.endswith("]"):
        try:
            parsed = ast.literal_eval(val.replace('\n', '').replace('\r', ''))  ########
            if isinstance(parsed, (list, tuple, np.ndarray)):
                return list(parsed)
        except Exception:
            return []
    elif isinstance(val, (list, tuple, np.ndarray)):
        return list(val)
    else:
        return []

def merge_table_list_to_df2():
    df = pd.read_parquet(FINAL_INTEGRATION_PARQUET, columns=['query', 'html_table_list', 'llm_table_list']) # , 'corpusid'
    print(f"  df loaded with shape: {df.shape}")

    # Clean stringified lists ########
    df['html_table_list'] = df['html_table_list'].apply(_safe_parse_list)  ########
    df['llm_table_list'] = df['llm_table_list'].apply(_safe_parse_list)  ########

    df2 = pd.read_parquet(ALL_TITLE_PATH)
    print(f"  df2 loaded with shape: {df2.shape}")
    print("\nStep 1: Expanding df2 to match df (on df2.all_title_list vs df.query)...")

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

    df2_merged.to_parquet(MERGE_PATH, index=False)
    return df2_merged

if __name__ == "__main__":
    print(f"Merging all tables list...")
    merge_table_list_to_df2()
    print(f"\n🎉 All tables merged and saved to {MERGE_PATH}.")
