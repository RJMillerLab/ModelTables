"""
Author: Zhengyuan Dong
Created: 2025-03-08
Last Modified: 2025-03-08
Description: Simulated download of READMEs from GitHub URLs without actual network requests. Actually save the parquet and cache files.
"""
import os
import re
import time
import pandas as pd
import numpy as np
import hashlib
from tqdm import tqdm
from src.utils import load_config
import urllib.parse

cache = {}

def clean_github_link(github_link):
    return github_link.split('{')[0].split('}')[0].split('[')[0].split(']')[0].split('(')[0].split(')')[0].split('<')[0].split('>')[0].split('*')[0].split('`')[0].split('"')[0].split("'")[0].split('!')[0]

def create_local_filename(base_output_dir, github_url):
    url_hash = hashlib.md5(github_url.encode('utf-8')).hexdigest()
    filename = f"{url_hash}.md"
    return os.path.join(base_output_dir, filename)

def main_download(df, base_output_dir, to_path="data/github_readmes_info.parquet"):
    assert 'github_link' in df.columns, "Missing 'github_link' column in DataFrame."
    assert 'modelId' in df.columns, "Missing 'modelId' column in DataFrame."
    
    start_time = time.time()
    all_raw_links = df[['modelId', 'github_link']].explode('github_link').dropna()
    total_links_before_dedup = len(all_raw_links)
    all_links = set(clean_github_link(str(link).strip()) for link in all_raw_links['github_link'] if link)
    
    print(f"Found {total_links_before_dedup} GitHub links before deduplication.")
    print(f"Found {len(all_links)} unique GitHub URLs after deduplication.")
    print(f"Speedup ratio: {total_links_before_dedup / len(all_links):.2f}x reduction in requests.")
    print(f"Step1 time cost: {time.time() - start_time:.2f} seconds.")
    
    # Step2: Simulated download
    start_time = time.time()
    simulate_download_github_urls(all_links, base_output_dir)
    print(f"Step2 time cost: {time.time() - start_time:.2f} seconds.")

    # step3: link downloaded files back ot the model data
    start_time = time.time()
    download_info = []
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Assembling Results"):
        model_id = row['modelId']
        raw_links = row['github_link']
        if raw_links is None:
            download_info.append({
                "modelId": model_id,
                "github_link": [],
                "readme_path": []
            })
            continue
        if isinstance(raw_links, str):
            raw_links = [raw_links]
        elif isinstance(raw_links, (list, tuple, np.ndarray)):
            raw_links = list(raw_links)
        readme_paths = []
        for g_link in raw_links:
            if not g_link:
                continue
            cleaned_link = clean_github_link(g_link.strip())
            local_path = cache.get(cleaned_link)
            if local_path:
                readme_paths.append(local_path)
        download_info.append({
            "modelId": model_id,
            "github_link": raw_links,
            "readme_path": readme_paths
        })
    download_info_df = pd.DataFrame(download_info)
    download_info_df.to_parquet(to_path, compression="zstd", engine="pyarrow", index=False)
    # save the cache as parquet
    cache_df = pd.DataFrame(list(cache.items()), columns=['raw_url', 'downloaded_path'])
    cache_df.to_parquet(os.path.join(config.get('base_path'), "processed", "github_readme_cache.parquet"), compression="zstd", engine="pyarrow", index=False)
    print(f"Downloaded {len([d for d in download_info if d['readme_path']])} READMEs.")
    print(f"Skipped {len([d for d in download_info if not d['readme_path']])} READMEs.")
    print(f"Step3 time cost: {time.time() - start_time:.2f} seconds.")
    return download_info_df

def simulate_download_github_urls(all_links, base_output_dir):
    """
    Simulate the GitHub README download process without actually downloading files.
    Instead, check if the corresponding SHA-based filename exists locally.
    """
    download_info = []
    for link in tqdm(all_links, desc="Simulating Downloads"):
        url_hash = hashlib.md5(link.encode('utf-8')).hexdigest()
        local_filename = os.path.join(base_output_dir, f"{url_hash}.md")
        if os.path.exists(local_filename):
            cache[link] = local_filename  # Simulate successful download
        else:
            cache[link] = None  # Simulate failed download

if __name__ == "__main__":
    config = load_config('config.yaml')
    processed_base_path = os.path.join(config.get('base_path'), "processed")
    base_output_dir = os.path.join(config.get('base_path'), "downloaded_github_readmes")
    data_type = "modelcard"
    os.makedirs(base_output_dir, exist_ok=True)
    df_split_temp = pd.read_parquet(os.path.join(processed_base_path, f'{data_type}_step1.parquet'), columns=['github_link', 'modelId'])
    print(df_split_temp.info())
    df_split_temp['github_link'] = df_split_temp['github_link'].apply(lambda x: eval(x) if isinstance(x, str) and x.startswith('[') and x.endswith(']') else x)
    download_info_df = main_download(df_split_temp, base_output_dir, to_path=os.path.join(processed_base_path, 'giturl_info.parquet'))
