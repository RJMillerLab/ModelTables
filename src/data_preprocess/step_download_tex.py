
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author: Zhengyuan Dong
Created: 2025-02-24
Last Modified: 2025-02-24
Description: Download .tar.gz files from arXiv links extracted from input CSV.
"""


import os
import pandas as pd
import requests
import tarfile
from tqdm import tqdm
from joblib import Parallel, delayed
import re
from src.utils import load_config

def extract_arxiv_id(url):
    if isinstance(url, str):
        match = re.search(r'arxiv\.org/abs/(\d+\.\d+)', url)
        return match.group(1) if match else None
    return None

def is_valid_tar_gz(file_path):
    """Check if the file is a valid .tar.gz archive."""
    try:
        with tarfile.open(file_path, "r:gz") as tar:
            tar.getmembers()  # Try to list archive contents
        return True
    except (tarfile.TarError, EOFError) as e:
        print(f"Error reading tar.gz file {file_path}: {e}")
        return False

def download_tex(arxiv_id, output_path):
    """Download arXiv source file (.tar.gz) if it does not exist."""
    if os.path.exists(output_path) and is_valid_tar_gz(output_path):
        return output_path  # Skip download if already valid
    
    try:
        url = f"https://arxiv.org/src/{arxiv_id}"
        response = requests.get(url, stream=True, timeout=10)
        if response.status_code == 200:
            with open(output_path, "wb") as f:
                f.write(response.content)
            # Validate downloaded file
            if is_valid_tar_gz(output_path):
                return output_path
            else:
                os.remove(output_path)
                return None
        else:
            return None
    except Exception as e:
        print('Error:', e)
        return None

def download_tex_row(row, base_output_dir):
    url = row['final_url']
    arxiv_id = extract_arxiv_id(url)
    if arxiv_id:
        model_id = row['modelId']
        model_dir = os.path.join(base_output_dir, model_id)
        os.makedirs(model_dir, exist_ok=True)
        file_path = os.path.join(model_dir, f"{arxiv_id}.tar.gz")
        downloaded_path = download_tex(arxiv_id, file_path)
        return {
            "modelId": model_id,
            "final_url": url,
            "local_path": downloaded_path if downloaded_path else None
        }
    return None

def main_download(df, base_output_dir="downloaded_tex_files", to_path="data/downloaded_tex_info.csv"):
    assert 'final_url' in df.columns, "Missing 'final_url' column in DataFrame."
    assert 'modelId' in df.columns, "Missing 'modelId' column in DataFrame."
    os.makedirs(base_output_dir, exist_ok=True)
    
    download_info = Parallel(n_jobs=-1)(
        delayed(download_tex_row)(row, base_output_dir) for _, row in tqdm(df.iterrows(), total=len(df))
    )
    download_info = [info for info in download_info if info is not None]
    download_info_df = pd.DataFrame(download_info)
    download_info_df.to_csv(to_path, index=False)
    print(f"Downloaded {len(download_info)} .tar.gz files.")
    print(f"Skipped {len(df) - len(download_info)} .tar.gz files.")
    return download_info_df

if __name__ == "__main__":
    config = load_config('config.yaml')
    base_path = config.get('base_path')
    df_processed = pd.read_csv(f"{base_path}/processed_final_urls.csv")
    to_path = f"{base_path}/downloaded_tex_info.csv"
    download_info_df = main_download(df_processed, to_path=to_path)
    print(download_info_df.head())
    print(f"Downloaded .tar.gz files saved to '{to_path}'.")
