"""
Author: Zhengyuan Dong
Created: 2025-03-11
Last Modified: 2025-04-06
Description: Create symlinks for various CSV resources (Hugging, GitHub, HTML, LLM)
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
import pickle

global_symlink_mapping = {}

def create_symlinks_generic(df, processed_base_path, input_col, output_subfolder, link_tag, output_col, use_abspath=False, enable_symlink=True):
    """
    Generic function to create symlinks for CSV files.
    
    Parameters:
      - df: DataFrame containing the original CSV paths.
      - processed_base_path: Base path for processed data.
      - input_col: Column name containing the list of original CSV paths.
      - output_subfolder: Subfolder name to store symlinks.
      - link_tag: Identifier used in symlink filenames.
      - output_col: Column name to store the list of symlink paths.
      - use_abspath: Whether to use absolute path for the symlink target (e.g., for HTML and LLM).
    """
    output_folder = os.path.join(processed_base_path, output_subfolder)
    os.makedirs(output_folder, exist_ok=True)
    df[output_col] = [[] for _ in range(len(df))]
    print(f"Creating symlinks in '{output_subfolder}' ...")
    for i, row in tqdm(df.iterrows(), total=len(df), desc=f"Linking {link_tag} CSVs"):
        model_id = row['modelId']
        model_id_sanitized = model_id.replace('/', '_')
        original_paths = row[input_col]
        symlink_list = []
        for idx, orig_path in enumerate(original_paths, start=1):
            #orig_path = os.path.normpath(orig_path)
            #if not os.path.isabs(orig_path):
            #    orig_path = os.path.join(processed_base_path, orig_path)
            if not os.path.exists(orig_path):
                print('Skipping non-existent CSV:', orig_path)
                continue
            link_filename = f"{model_id_sanitized}_{link_tag}{idx}.csv"
            link_path = os.path.join(output_folder, link_filename)
            if os.path.lexists(link_path):
                os.remove(link_path)
            target_path = os.path.abspath(orig_path) if use_abspath else orig_path
            if enable_symlink:
                try:
                    os.symlink(target_path, link_path)
                except Exception as e:
                    print(f"Error creating symlink for {orig_path}: {e}")
            relative_link = link_path[link_path.index("data/processed/"):] if "data/processed/" in link_path else link_path
            symlink_list.append(relative_link)
            global_symlink_mapping[relative_link] = target_path
        # Convert paths to be relative starting from "data/processed"
        #symlink_list = [p[p.index("data/processed/"):] if "data/processed/" in p else p for p in symlink_list]
        df.at[i, output_col] = symlink_list
    return df

def symlink_factory(input_col, output_subfolder, link_tag, output_col, use_abspath=False, enable_symlink=True):
    """
    Factory function to create a symlink creation function for a specific resource.
    """
    def create_symlink_func(df, processed_base_path):
        return create_symlinks_generic(
            df,
            processed_base_path,
            input_col=input_col,
            output_subfolder=output_subfolder,
            link_tag=link_tag,
            output_col=output_col,
            use_abspath=use_abspath,
            enable_symlink=enable_symlink
        )
    return create_symlink_func

# Generate symlink creation functions for different resources.
# For HTML and LLM, we assume the DataFrame already contains the columns:
# 'html_table_list_mapped' and 'llm_table_list_mapped' respectively.
enable_symlink = False
create_symlink_hugging = symlink_factory("hugging_table_list_dedup", "sym_hugging_csvs", "hugging_table", "hugging_table_list_sym", False, enable_symlink)
create_symlink_github = symlink_factory("github_table_list_dedup", "sym_github_csvs", "github_table", "github_table_list_sym", False, enable_symlink)
create_symlink_html = symlink_factory("html_table_list_mapped_dedup", "sym_html_csvs", "html_table", "html_table_list_sym", True, enable_symlink)
create_symlink_llm = symlink_factory("llm_table_list_mapped_dedup", "sym_llm_csvs", "llm_table", "llm_table_list_sym", True, enable_symlink)

# Main execution
if __name__ == "__main__":
    config = load_config('config.yaml')
    processed_base_path = os.path.join(config.get('base_path'), 'processed')
    data_type = 'modelcard'
    
    # load four keys directly from step3_merged.parquet
    df_merged = pd.read_parquet(os.path.join(processed_base_path, f"{data_type}_step3_dedup.parquet"),
                                columns=['modelId', 'html_table_list_mapped_dedup', 'llm_table_list_mapped_dedup', 'hugging_table_list_dedup', 'github_table_list_dedup'])
    print(f"Loaded DataFrame with {len(df_merged)} rows.")

    # Create symlinks for all resources
    df_merged = create_symlink_hugging(df_merged, processed_base_path)
    df_merged = create_symlink_github(df_merged, processed_base_path)
    df_merged = create_symlink_html(df_merged, processed_base_path)
    df_merged = create_symlink_llm(df_merged, processed_base_path)

    # Save the final DataFrame
    df_merged.to_parquet(os.path.join(processed_base_path, f"{data_type}_step4.parquet"), compression='zstd', engine='pyarrow', index=False)
    print(f"Symlinks recreated and saved to data/processed/{data_type}_step4.parquet.")

    mapping_path = os.path.join(processed_base_path, "symlink_mapping.pickle")
    with open(mapping_path, "wb") as f:
        pickle.dump(global_symlink_mapping, f)
    print(f"Symlink mapping saved to {mapping_path}")