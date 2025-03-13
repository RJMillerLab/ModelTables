"""
Author : Zhengyuan Dong
Created : 2025-03-12
Last Modified : 2025-03-12
Description : Recreate symlinks for HuggingFace deduped CSVs

Usage:
    python -m src.data_preprocess.step2_recreate_symlinks
"""

import os
import json
import pandas as pd
from tqdm import tqdm
from src.utils import load_config

def recreate_symlinks_hugging(config_path="config.yaml"):
    config = load_config(config_path)
    processed_base_path = os.path.join(config.get('base_path'), 'processed')
    data_type = 'modelcard'

    step2_parquet = os.path.join(processed_base_path, f"{data_type}_step2.parquet")
    print(f"Loading parquet from {step2_parquet}")
    df_merged = pd.read_parquet(step2_parquet)
    print(f"Loaded DataFrame with {len(df_merged)} rows.")

    hugging_map_json_path = os.path.join(processed_base_path, "hugging_deduped_mapping.json")  ########
    print(f"Loading HuggingFace deduped mapping from {hugging_map_json_path}")
    with open(hugging_map_json_path, 'r', encoding='utf-8') as jf:
        hash_to_csv_map = json.load(jf)

    output_folder_hugging = os.path.join(processed_base_path, "cleaned_markdown_csvs_hugging")
    os.makedirs(output_folder_hugging, exist_ok=True)

    if 'hugging_csv_files' not in df_merged.columns:
        df_merged['hugging_csv_files'] = [[] for _ in range(len(df_merged))]
    else:
        df_merged['hugging_csv_files'] = [[] for _ in range(len(df_merged))]

    print("Creating symlinks for HuggingFace deduped CSVs...")
    for i, row in tqdm(df_merged.iterrows(), total=len(df_merged), desc="Linking HuggingFace CSVs"):
        hval = row.get('readme_hash')
        if not hval:
            continue
        deduped_csv_list = hash_to_csv_map.get(hval, [])
        symlink_paths = []
        if deduped_csv_list:
            model_id = row['modelId']
            model_id_sanitized = model_id.replace('/', '_')
            for idx_table, csv_real_path in enumerate(deduped_csv_list, start=1):
                link_filename = f"{model_id_sanitized}_hugging_table{idx_table}.csv"  ########
                link_path = os.path.join(output_folder_hugging, link_filename)
                if os.path.lexists(link_path):
                    os.remove(link_path)
                try:
                    os.symlink(csv_real_path, link_path)  ########
                except Exception as e:
                    print(f"❌ Error creating symlink: {link_path} -> {csv_real_path}: {e}")
                symlink_paths.append(link_path)
        df_merged.at[i, 'hugging_csv_files'] = symlink_paths

    df_merged.to_parquet(step2_parquet, index=False)
    print("✅ HuggingFace symlinks recreated and DataFrame updated.")

def recreate_symlinks_github(config_path="config.yaml"):
    """
    Recreate symlinks for GitHub CSV files.
    1. 从 cleaned_markdown_csvs_github 文件夹中加载 md_to_csv_mapping.json（由原始 GitHub 处理生成）。
    2. 根据映射中的 CSV 文件名，在 symlinked_github_csvs 文件夹下创建符号链接。
    """
    config = load_config(config_path)
    processed_base_path = os.path.join(config.get('base_path'), 'processed')

    github_csv_folder = os.path.join(processed_base_path, "cleaned_markdown_csvs_github")  ########
    mapping_json_path = os.path.join(github_csv_folder, "md_to_csv_mapping.json")  ########
    print(f"Loading GitHub mapping from {mapping_json_path}")
    with open(mapping_json_path, 'r', encoding='utf-8') as f:
        md_to_csv_mapping = json.load(f)

    output_folder_symlinks = os.path.join(processed_base_path, "symlinked_github_csvs")  ########
    os.makedirs(output_folder_symlinks, exist_ok=True)

    symlink_mapping = {}
    print("Creating symlinks for GitHub CSVs...")
    for md_basename, csv_list in md_to_csv_mapping.items():
        if not csv_list:
            symlink_mapping[md_basename] = []
            continue
        symlinked_csv_paths = []
        for csv_basename in csv_list:
            csv_full_path = os.path.join(github_csv_folder, csv_basename)  ########
            if not os.path.exists(csv_full_path):
                print(f"Warning: CSV file {csv_full_path} not found, skipping.")
                continue
            symlink_path = os.path.join(output_folder_symlinks, csv_basename)  ########
            if os.path.lexists(symlink_path):
                os.remove(symlink_path)
            try:
                os.symlink(os.path.abspath(csv_full_path), symlink_path)
            except Exception as e:
                print(f"❌ Error creating symlink for {csv_full_path}: {e}")
            symlinked_csv_paths.append(symlink_path)
        symlink_mapping[md_basename] = symlinked_csv_paths

    mapping_out_path = os.path.join(output_folder_symlinks, "symlinked_github_mapping.json")  ########
    with open(mapping_out_path, 'w', encoding='utf-8') as f:
        json.dump(symlink_mapping, f, indent=2)
    print("✅ GitHub symlinks recreated.")

def main():
    config_path = "config.yaml"
    recreate_symlinks_hugging(config_path)  ########
    recreate_symlinks_github(config_path)  ########

if __name__ == "__main__":
    main()
