"""
Author: Zhengyuan Dong
Created: 2025-03-11
Last Modified: 2025-04-02
Description: 
Usage:
    python -m src.data_gt.step3_create_symlinks
"""

import os, re, time, logging, hashlib, json
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np
from shutil import copytree
import shutil
from src.data_ingestion.readme_parser import MarkdownHandler
from src.utils import load_data, load_config


def create_symlinks(md_to_csv_mapping, output_folder_symlinks):
    """
    Create symbolic links for GitHub processed CSVs in a separate directory to avoid data duplication.
    """
    os.makedirs(output_folder_symlinks, exist_ok=True)
    symlink_mapping = {}  # Store symlinked paths for later use
    for md_basename, csv_paths in md_to_csv_mapping.items():
        if not csv_paths:
            symlink_mapping[md_basename] = []
            continue
        symlinked_csv_paths = []
        for csv_basename in csv_paths:
            csv_full_path = os.path.join(output_folder_symlinks.replace("sym_github_csvs", "deduped_github_csvs"), csv_basename)
            if not os.path.exists(csv_full_path):
                # If we don't find the actual CSV in "deduped_github_csvs", skip
                # (or you can adapt logic if you want to locate them differently)
                continue
            symlink_path = os.path.join(output_folder_symlinks, os.path.basename(csv_basename))
            try:
                if os.path.exists(symlink_path) or os.path.islink(symlink_path):
                    os.unlink(symlink_path)  # Remove existing symlink if it exists
                os.symlink(os.path.abspath(csv_full_path), symlink_path)  # Create symlink
                symlinked_csv_paths.append(symlink_path)
            except Exception as e:
                print(f"❌ Error creating symlink for {csv_full_path}: {e}")
        symlink_mapping[md_basename] = symlinked_csv_paths
    return symlink_mapping

def create_symlink_hugging(df_merged, processed_base_path):
    # load the deduped mapping
    hugging_map_json_path = os.path.join(processed_base_path, "hugging_deduped_mapping.json")  ########
    with open(hugging_map_json_path, 'r', encoding='utf-8') as jf:
        hash_to_csv_map = json.load(jf)
    # create the output folder
    output_folder_hugging = os.path.join(processed_base_path, "sym_hugging_csvs")
    os.makedirs(output_folder_hugging, exist_ok=True)
    # We'll track readme_hash -> list of "master" CSV paths
    df_merged['hugging_table_list'] = [[] for _ in range(len(df_merged))]
    df_merged['hugging_table_list_sym'] = [[] for _ in range(len(df_merged))]
    print("⚠️Step: Creating symlinks to 'sym_hugging_csvs' ...")
    for i, row in tqdm(df_merged.iterrows(), total=len(df_merged), desc="Linking CSVs ..."):
        hval = row['readme_hash']
        if not isinstance(hval, str):
            continue
        deduped_csv_list = hash_to_csv_map.get(hval, [])
        hugging_csv_list = []
        hugging_csv_list_sym = []
        if deduped_csv_list:
            model_id = row['modelId']
            for idx_table, csv_real_path in enumerate(deduped_csv_list, start=1):
                hugging_csv_list.append(csv_real_path)
                # e.g.: <processed_base_path>/sym_hugging_csvs/<modelId>_hugging_table{idx_table}.csv
                model_id_sanitized = model_id.replace('/', '_')
                link_filename = f"{model_id_sanitized}_hugging_table{idx_table}.csv"  ########
                link_path = os.path.join(output_folder_hugging, link_filename)
                if os.path.lexists(link_path):
                    os.remove(link_path)
                try:
                    os.symlink(csv_real_path, link_path)  ########
                except Exception as e:
                    print(f"❌ Error creating symlink: {link_path} -> {csv_real_path}: {e}")
                hugging_csv_list_sym.append(link_path)
        hugging_csv_list = [
            p[p.index("data/processed/"):] if "data/processed/" in p else p for p in hugging_csv_list
        ]
        hugging_csv_list_sym = [
            p[p.index("data/processed/"):] if "data/processed/" in p else p for p in hugging_csv_list_sym
        ]
        df_merged.at[i, 'hugging_table_list'] = hugging_csv_list
        df_merged.at[i, 'hugging_table_list_sym'] = hugging_csv_list_sym
    return df_merged

def create_symlink_github(df_merged, processed_base_path):
    ########
    # Instead of the old create_symlinks() call, we replicate the approach used above for hugging:
    ########
    with open(os.path.join(processed_base_path, "deduped_github_csvs", "md_to_csv_mapping.json"), 'r', encoding='utf-8') as jf:
        md_to_csv_mapping = json.load(jf)
    output_folder_github_sym = os.path.join(processed_base_path, "sym_github_csvs")  ########
    os.makedirs(output_folder_github_sym, exist_ok=True)                            ########
    df_merged['github_table_list'] = [[] for _ in range(len(df_merged))]            ########
    df_merged['github_table_list_sym'] = [[] for _ in range(len(df_merged))]        ########
    print("⚠️Step: Creating symlinks to 'sym_github_csvs' ...")                     ########
    for i, row in tqdm(df_merged.iterrows(), total=len(df_merged), desc="Linking CSVs ..."):
        readme_paths = row['readme_path']
        if isinstance(readme_paths, str):
            readme_paths = [readme_paths]
        model_id = row['modelId']
        model_id_sanitized = model_id.replace('/', '_')                              ########
        # Gather all CSVs from the readme(s)
        combined_csvs = []
        for md_file in readme_paths:
            md_basename = os.path.basename(md_file).replace(".md", "")
            #combined_csvs.extend(md_to_csv_mapping.get(md_basename, []))
            value = md_to_csv_mapping.get(md_basename)
            if value not in [None, []]:
                combined_csvs.extend(value)
        combined_csvs = list(set(combined_csvs))
        github_csv_list = []
        github_csv_list_sym = []
        for idx_table, csv_basename in enumerate(combined_csvs, start=1):           ########
            csv_full_path = os.path.join(processed_base_path, "deduped_github_csvs", csv_basename)
            if not os.path.exists(csv_full_path):
                print('csv_full_path: ', csv_full_path)
                print('skipping non-exist csv')
                continue
            # e.g. <processed_base_path>/sym_github_csvs/<modelId>_github_table{n}.csv
            link_filename = f"{model_id_sanitized}_github_table{idx_table}.csv"     ########
            link_path = os.path.join(output_folder_github_sym, link_filename)       ########
            if os.path.lexists(link_path):
                os.remove(link_path)
            try:
                os.symlink(csv_full_path, link_path)                                ########
            except Exception as e:
                print(f"❌ Error creating symlink for {csv_full_path}: {e}")
            github_csv_list.append(csv_full_path)
            github_csv_list_sym.append(link_path)
        # only keep paths starting from data/processed
        github_csv_list = [
            p[p.index("data/processed/"):] if "data/processed/" in p else p for p in github_csv_list
        ]
        github_csv_list_sym = [
            p[p.index("data/processed/"):] if "data/processed/" in p else p for p in github_csv_list_sym
        ]
        df_merged.at[i, "github_table_list"] = github_csv_list
        df_merged.at[i, "github_table_list_sym"] = github_csv_list_sym
    return df_merged

def create_symlink_html(df_merged, processed_base_path):
    step3_path = os.path.join(processed_base_path, "modelcard_step3_merged.parquet")
    df_html = pd.read_parquet(step3_path, columns=['modelId', 'html_table_list_mapped'])
    df_merged = pd.merge(df_merged, df_html, on='modelId', how='left')
    output_html = os.path.join(processed_base_path, "sym_html_csvs")
    os.makedirs(output_html, exist_ok=True)
    df_merged['html_table_list_sym'] = [[] for _ in range(len(df_merged))]
    for i, row in tqdm(df_merged.iterrows(), total=len(df_merged), desc="Linking HTML tables"):
        model_id = row['modelId']
        model_id_sanitized = model_id.replace("/", "_")
        html_list = row.get("html_table_list_mapped", [])
        html_syms = []
        if isinstance(html_list, (list, tuple, np.ndarray)):
            for idx, real_path in enumerate(html_list, start=1):
                if not os.path.exists(real_path):
                    continue
                link_name = f"{model_id_sanitized}_html_table{idx}.csv"
                link_path = os.path.join(output_html, link_name)
                if os.path.lexists(link_path):
                    os.remove(link_path)
                try:
                    os.symlink(os.path.abspath(real_path), link_path)
                    html_syms.append(link_path)
                except Exception as e:
                    print(f"❌ Error linking HTML {real_path}: {e}")
        df_merged.at[i, "html_table_list_sym"] = [
            p[p.index("data/processed/"):] if "data/processed/" in p else p for p in html_syms
        ]
    return df_merged

def create_symlink_llm(df_merged, processed_base_path):
    step3_path = os.path.join(processed_base_path, "modelcard_step3_merged.parquet")
    df_llm = pd.read_parquet(step3_path, columns=['modelId', 'llm_table_list_mapped'])
    df_merged = pd.merge(df_merged, df_llm, on='modelId', how='left')
    output_llm = os.path.join(processed_base_path, "sym_llm_csvs")
    os.makedirs(output_llm, exist_ok=True)
    df_merged['llm_table_list_sym'] = [[] for _ in range(len(df_merged))]
    for i, row in tqdm(df_merged.iterrows(), total=len(df_merged), desc="Linking LLM tables"):
        model_id = row['modelId']
        model_id_sanitized = model_id.replace("/", "_")
        llm_list = row.get("llm_table_list_mapped", [])
        llm_syms = []
        if isinstance(llm_list, (list, tuple, np.ndarray)):
            for idx, real_path in enumerate(llm_list, start=1):
                if not os.path.exists(real_path):
                    continue
                link_name = f"{model_id_sanitized}_llm_table{idx}.csv"
                link_path = os.path.join(output_llm, link_name)
                if os.path.lexists(link_path):
                    os.remove(link_path)
                try:
                    os.symlink(os.path.abspath(real_path), link_path)
                    llm_syms.append(link_path)
                except Exception as e:
                    print(f"❌ Error linking LLM {real_path}: {e}")
        df_merged.at[i, "llm_table_list_sym"] = [
            p[p.index("data/processed/"):] if "data/processed/" in p else p for p in llm_syms
        ]
    return df_merged

if __name__ == "__main__":
    config = load_config('config.yaml')
    processed_base_path = os.path.join(config.get('base_path'), 'processed')
    data_type = 'modelcard'
    
    df_merged = pd.read_parquet(os.path.join(processed_base_path, f"{data_type}_step2.parquet"), columns=['modelId', 'readme_path', 'readme_hash'])
    print(f"Loaded DataFrame with {len(df_merged)} rows.")

    df_merged = create_symlink_github(df_merged, processed_base_path)
    df_merged = create_symlink_hugging(df_merged, processed_base_path)

    df_merged = create_symlink_html(df_merged, processed_base_path)  ########
    df_merged = create_symlink_llm(df_merged, processed_base_path)   ########

    df_merged.to_parquet(os.path.join(processed_base_path, f"{data_type}_step4.parquet"), index=False)
    print(f"✅ Symlinks recreated and saved to data/processed/modelcard_step4.parquet.")
    