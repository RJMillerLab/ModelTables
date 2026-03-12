"""
Author: Zhengyuan Dong
Created: 2026-03-11
Last Modified: 2026-03-11
Description: Merge title2arxiv and html_parsing_results_v2 to get title2htmltab.
Usage: python -m src.data_preprocess.step2_merge_tables_simplify --tag v2_251117
"""

import os
import re
import argparse
import pandas as pd

from src.utils import load_config, to_parquet
from src.data_preprocess.step2_integration_s2orc_llm import (normalize_title, preprocess_title, parse_arxiv_id, convert_to_list)

def main():
    parser = argparse.ArgumentParser(description="Integrate HTML/PDF/annotation tables and prepare LLM inputs")
    parser.add_argument('--tag', dest='tag', default=None, help='Tag suffix for versioning (e.g., 251117). Enables versioning mode.')
    parser.add_argument('--v2_mode', dest='v2_mode', action='store_true', help='Use v2 mode.')
    args = parser.parse_args()

    config = load_config('config.yaml')
    base_path = config.get('base_path', 'data')
    suffix = f"_{args.tag}" if args.tag else ""
    v2_suffix = "_v2" if args.v2_mode else ""

    TITLE_PARQUET = os.path.join(base_path, 'processed', f"s2orc_rerun{suffix}.parquet")
    TITLE2ARXIV_PARQUET = os.path.join(base_path, 'processed', f"title2arxiv_cache{suffix}.parquet")
    HTML_TABLE_PARQUET = os.path.join(base_path, 'processed', f"html_parsing_results{v2_suffix}{suffix}.parquet")
    #FINAL_OUTPUT_PARQUET = os.path.join(base_path, 'processed', f"title2htmltab{suffix}.parquet")

    print("📁 Paths in use:")
    print(f"   Query titles:        {TITLE_PARQUET}")
    print(f"   Title→arxiv cache:  {TITLE2ARXIV_PARQUET}")
    print(f"   HTML table list:      {HTML_TABLE_PARQUET}")

    # --- Step 1: Load extracted annotations ---
    df_title = pd.read_parquet(TITLE_PARQUET, columns=['query_title', 'retrieved_title'])
    df_title["norm_title"] = df_title["retrieved_title"].apply(normalize_title)
    df_title["preproc_title"] = df_title["retrieved_title"].apply(preprocess_title)
    print("📝 df_title shape:", df_title.shape)

    # --- Step 2: Load title2arxiv mapping (title -> arxiv_id) ---
    df_cache = pd.read_parquet(TITLE2ARXIV_PARQUET, columns=["title", "arxiv_id", "norm_title"])
    df_title2arxiv = df_cache[df_cache["arxiv_id"].notna() & (df_cache["arxiv_id"].astype(str).str.strip() != "")][["title", "arxiv_id", "norm_title"]].copy()
    df_title2arxiv = df_title2arxiv.rename(columns={"title": "retrieved_title"}).drop_duplicates(subset=["retrieved_title"], keep="first")
    df_title2arxiv["preproc_title"] = df_title2arxiv["retrieved_title"].apply(preprocess_title)
    print("📝 df_title2arxiv shape:", df_title2arxiv.shape)

    # --- Step 3: Merge df_html with df_title2arxiv based on arxiv id pure version ---
    print(f"📦 Loading HTML tables: {HTML_TABLE_PARQUET}")
    df_html = pd.read_parquet(HTML_TABLE_PARQUET) # Columns: [paper_id, html_path, page_type, csv_paths]
    if 'csv_paths' in df_html.columns and 'table_list' not in df_html.columns:
        df_html['table_list'] = df_html['csv_paths'].apply(convert_to_list)
    print("📝 df_html shape:", df_html.shape)
    ########### Merge df based on arxiv id pure version
    df_html[['arxiv_id_pure', 'arxiv_id_version']] = df_html['paper_id'].apply(parse_arxiv_id)
    # keep the latest version
    df_html = df_html.sort_values('arxiv_id_version', ascending=False).drop_duplicates('arxiv_id_pure', keep='first')
    df_title2arxiv['arxiv_id_pure'] = df_title2arxiv['arxiv_id'].apply(lambda x: parse_arxiv_id(x)[0])
    df_html_merged = pd.merge(df_title2arxiv, df_html,left_on="arxiv_id_pure", right_on="arxiv_id_pure",how="left")
    df_html_merged.rename(columns={"html_path": "html_html_path", "page_type": "html_page_type", "table_list": "html_table_list", "paper_id": "html_paper_id"}, inplace=True)
    del df_html
    print("📝 df_html_merged shape:", df_html_merged.shape)

    # --- Step 4: Merge html info to title, try multiple processed titles---
    df_merged = pd.merge(df_title, df_html_merged, on="retrieved_title", how="left", suffixes=("", "_temp")) # main key: query title
    del df_html_merged, df_title
    # try multiple processed titles to map title to arxiv id
    keys_to_try = [("norm_title", "norm_title"), ("preproc_title", "preproc_title")]
    for left_key, right_key in keys_to_try:
        mask_missing = df_merged["arxiv_id"].isna()
        if not mask_missing.any():
            break
        df_missing = df_merged[mask_missing].copy()
        df_missing2 = pd.merge(df_missing.drop(columns=["arxiv_id"]), df_title2arxiv[[right_key, "arxiv_id"]], left_on=left_key, right_on=right_key, how="left")
        df_merged.loc[mask_missing, "arxiv_id"] = df_missing2["arxiv_id"].values
    df_merged.drop(columns=["norm_title", "preproc_title"], inplace=True)
    print("📝 After merging title mapping, shape:", df_merged.shape)
    #to_parquet(df_merged, FINAL_OUTPUT_PARQUET)

    # --- Step 5: Merge back to modelid level, borrow logic from step2_merge_tables---
    from src.data_preprocess.step2_merge_tables import map_tables_by_title, populate_hugging_table_list, populate_github_table_list, _safe_parse_list
    
    # Determine input/output paths: tag is full suffix (no _v2 in template)
    modelid2titles_path = os.path.join(base_path, 'processed', f"modelcard_all_title_list{suffix}.parquet")
    modelid2readmeinfo_path = os.path.join(base_path, 'processed', f"modelcard_step2{v2_suffix}{suffix}.parquet")
    modelid2tablist_path = os.path.join(base_path, 'processed', f"modelcard_step3_merged{v2_suffix}{suffix}.parquet")

    hugging_map_json_path = os.path.join(base_path, 'processed', f"hugging_deduped_mapping{v2_suffix}{suffix}.json")
    github_csvs_folder = os.path.join(base_path, 'processed', f"deduped_github_csvs{v2_suffix}{suffix}")
    github_mapping_path = os.path.join(github_csvs_folder, f"md_to_csv_mapping.json")
    
    print(f"\nMerging all tables list...")
    # Build query_to_tablist_df directly from df_merged
    query_to_tablist_df = df_merged.rename(columns={"query_title": "query"})[["query", "html_table_list"]]
    # llm_table_list: all-empty list column (each row is [])
    query_to_tablist_df["llm_table_list"] = [[] for _ in range(len(query_to_tablist_df))]
    # Clean stringified lists ########
    query_to_tablist_df['html_table_list'] = query_to_tablist_df['html_table_list'].apply(_safe_parse_list)  ########
    query_to_tablist_df['llm_table_list'] = query_to_tablist_df['llm_table_list'].apply(_safe_parse_list)  ########

    model_level_df = pd.read_parquet(modelid2titles_path, columns=['modelId', 'all_title_list'])
    print(f"  modelid2titles loaded with shape: {model_level_df.shape}")
    print("\nStep 1: Expanding modelid2titles to match df (on modelid2titles.all_title_list vs df.query)...")
    model_w_html_llm = map_tables_by_title(model_level_df, query_to_tablist_df)
    
    # load side data and merge to df with modelId
    side_df = pd.read_parquet(modelid2readmeinfo_path, columns=['modelId', 'readme_path', 'readme_hash'])
    side_df = populate_hugging_table_list(side_df, hugging_map_json_path) # readme_hash -> hugging_table_list
    side_df = populate_github_table_list(side_df, github_csvs_folder, github_mapping_path) # readme_path -> github_table_list
    model_w_all_tab = pd.merge(side_df, model_w_html_llm, on='modelId', how='left')
    to_parquet(model_w_all_tab[['modelId', 'hugging_table_list', 'github_table_list', 'html_table_list_mapped', 'llm_table_list_mapped']], modelid2tablist_path)
    print(f"\n🎉 All tables merged and saved to {modelid2tablist_path}.")

if __name__ == "__main__":
    main()
