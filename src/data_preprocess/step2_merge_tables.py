"""
Author: Zhengyuan Dong
Created: 2025-03-11
Last Modified: 2026-03-11
Description: Map from (paper) title back to model-level ID. Joins final_integration (title→html/llm table paths)
             with modelcard_all_title_list (modelId→all_title_list). Links HuggingFace/GitHub tables via
             readme_hash and readme_path from modelcard_step2. Output: modelId + html/llm/hugging/github table lists.
Usage:
    python -m src.data_preprocess.step2_merge_tables --tag v2_251117
"""

import ast  ########
import os
import argparse
import pandas as pd
from src.utils import to_parquet, load_config, is_list_like, to_list_safe

def _safe_parse_list(val):
    """Parse list-like strings into actual list, or return as-is."""
    if isinstance(val, str) and val.strip().startswith("[") and val.strip().endswith("]"):
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

def populate_hugging_table_list(tmp_df, hugging_map_json_path):
    """
    Populate 'hugging_table_list' using hugging_deduped_mapping.
    """
    print(f"📦 Using HuggingFace mapping: {hugging_map_json_path}")
    mapping_s = pd.read_json(hugging_map_json_path, typ="series")
    def normalize_path_list(paths):
        if not is_list_like(paths):
            return []
        paths = to_list_safe(paths)
        return [
            p[p.index("data/processed/"):] if isinstance(p, str) and "data/processed/" in p else p
            for p in paths
        ]
    map_dict = mapping_s.apply(normalize_path_list).to_dict()
    tmp_df = tmp_df.copy()
    tmp_df["hugging_table_list"] = tmp_df["readme_hash"].map(
        lambda x: map_dict.get(x, []) if isinstance(x, str) else []
    )
    return tmp_df

def populate_github_table_list(tmp_df, github_csvs_folder, github_mapping_path):
    """
    Populate 'github_table_list' using md_to_csv_mapping.json.
    """
    print(f"📦 Using GitHub mapping: {github_mapping_path}")

    mapping_s = pd.read_json(github_mapping_path, typ="series")

    def normalize_readme_paths(readme_paths):
        if readme_paths is None:
            return []
        if isinstance(readme_paths, str):
            return [readme_paths]
        elif is_list_like(readme_paths):
            return to_list_safe(readme_paths)
        else:
            try:
                if pd.isna(readme_paths):
                    return []
                return []
            except Exception:
                return []

    def build_github_table_list(readme_paths):
        readme_paths = normalize_readme_paths(readme_paths)

        combined_csvs = []
        seen = set()

        for md_file in readme_paths:
            if not md_file or not isinstance(md_file, str):
                continue
            md_basename = os.path.basename(md_file).replace(".md", "")
            value = mapping_s.get(md_basename)
            if value in [None, []]:
                continue

            for csv_basename in value:
                if csv_basename not in seen:
                    seen.add(csv_basename)
                    combined_csvs.append(os.path.join(github_csvs_folder, csv_basename))

        return combined_csvs

    tmp_df = tmp_df.copy()
    tmp_df["github_table_list"] = tmp_df["readme_path"].apply(build_github_table_list)
    return tmp_df

def map_tables_by_title(model_level_df, query_to_tablist_df):
    """
    Map model all_title_list to html/llm table lists via query lookup.
    Content-equivalent to the other implementations, ignoring order.
    """
    html_lookup = {}
    llm_lookup = {}

    for row in query_to_tablist_df.itertuples(index=False):
        html_lookup[row.query] = to_list_safe(row.html_table_list) if is_list_like(row.html_table_list) else []
        llm_lookup[row.query] = to_list_safe(row.llm_table_list) if is_list_like(row.llm_table_list) else []

    def normalize_title_list(title_list):
        return list(dict.fromkeys(to_list_safe(title_list))) if is_list_like(title_list) else []

    def collect_tables(title_list):
        title_list = normalize_title_list(title_list)
        combined_html = []
        combined_llm = []

        for title in title_list:
            combined_html.extend(html_lookup.get(title, []))
            combined_llm.extend(llm_lookup.get(title, []))

        return (
            list(dict.fromkeys(combined_html)),
            list(dict.fromkeys(combined_llm)),
        )

    out = model_level_df.copy()
    mapped = out["all_title_list"].apply(collect_tables)
    out["html_table_list_mapped"] = mapped.str[0]
    out["llm_table_list_mapped"] = mapped.str[1]
    return out


def main():
    parser = argparse.ArgumentParser(description="Merge all table lists into a unified model ID file")
    parser.add_argument('--tag', dest='tag', default=None, help='Tag suffix for versioning (e.g., 251117). Enables versioning mode.')
    parser.add_argument('--v2_mode', dest='v2_mode', action='store_true', help='Use v2 mode.')
    args = parser.parse_args()
    
    config = load_config('config.yaml')
    base_path = config.get('base_path', 'data')
    suffix = f"_{args.tag}" if args.tag else ""
    v2_suffix = "_v2" if args.v2_mode else ""
    
    # Determine input/output paths: tag is full suffix (no _v2 in template)
    query_to_tablist_path = os.path.join(base_path, 'processed', f"final_integration_with_paths{v2_suffix}{suffix}.parquet")
    modelid2titles_path = os.path.join(base_path, 'processed', f"modelcard_all_title_list{suffix}.parquet")
    modelid2readmeinfo_path = os.path.join(base_path, 'processed', f"modelcard_step2{v2_suffix}{suffix}.parquet")
    modelid2tablist_path = os.path.join(base_path, 'processed', f"modelcard_step3_merged{v2_suffix}{suffix}.parquet")
    
    hugging_map_json_path = os.path.join(base_path, 'processed', f"hugging_deduped_mapping{v2_suffix}{suffix}.json")
    github_csvs_folder = os.path.join(base_path, 'processed', f"deduped_github_csvs{v2_suffix}{suffix}")
    github_mapping_path = os.path.join(github_csvs_folder, f"md_to_csv_mapping.json")
    
    print(f"\nMerging all tables list...")
    query_to_tablist_df = pd.read_parquet(query_to_tablist_path, columns=['query', 'html_table_list', 'llm_table_list']) # , 'corpusid'
    print(f"  query_to_tablist_df loaded with shape: {query_to_tablist_df.shape}")
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