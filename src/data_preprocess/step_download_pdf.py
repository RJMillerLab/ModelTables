"""
Author: Zhengyuan Dong
Created: 2025-02-24
Last Modified: 2025-02-24
Description: Download PDFs from the final URLs.
"""

import os
import pandas as pd
import requests
from tqdm import tqdm
from src.utils import load_config
import hashlib

cache = {}

def get_cache_key(url):
    return hashlib.md5(url.encode()).hexdigest()  # Generate a unique key based on the URL

def download_pdf(url, output_path):
    cache_key = get_cache_key(url)
    if cache_key in cache:
        return cache[cache_key]  # Return cached path if already downloaded
    if os.path.exists(output_path):
        cache[cache_key] = output_path  # Cache the existing file path
        return output_path
    try:
        response = requests.get(url, stream=True, timeout=10)
        if response.status_code == 200 and "application/pdf" in response.headers.get("Content-Type", ""):
            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024):
                    f.write(chunk)
            cache[cache_key] = output_path  # Cache the downloaded file path
            return output_path
        else:
            return None
    except Exception as e:
        return None

def main_download(df, base_output_dir, to_path="data/downloaded_pdfs_info.parquet"):
    assert 'final_url' in df.columns, "Missing 'final_url' column in DataFrame."
    assert 'modelId' in df.columns, "Missing 'modelId' column in DataFrame."
    download_info = []

    def download_row(row):
        url = row['final_url']
        if pd.notna(url):
            file_name = url.split('/')[-1]
            file_path = os.path.join(base_output_dir, file_name)
            downloaded_path = download_pdf(url, file_path)
            return {
                "modelId": row['modelId'],
                "final_url": url,
                "local_path": downloaded_path if downloaded_path else None
            }
        return None
    
    with ThreadPoolExecutor() as executor:
        download_info = list(tqdm(executor.map(download_row, [row for _, row in df.iterrows()]), total=len(df)))
    
    download_info = [info for info in download_info if info is not None]
    download_info_df = pd.DataFrame(download_info)
    download_info_df.to_parquet(to_path, index=False)
    cache_path = os.path.join(base_output_dir, "downloaded_pdf_cache.parquet")
    #pd.DataFrame(cache).to_parquet(cache_path)
    print(f"Downloaded {len([d for d in download_info if d['local_path']])} PDFs.")
    print(f"Skipped {len([d for d in download_info if not d['local_path']])} PDFs.")
    return download_info_df

if __name__ == "__main__":
    config = load_config('config.yaml')
    base_path = config.get('base_path')
    df_processed = pd.read_parquet(os.path.join(base_path, "processed", "processed_final_urls.parquet"), columns=['final_url', 'modelId'])
    to_path = os.path.join(base_path, "processed", "downloaded_pdfs_info.parquet")
    base_output_dir = os.path.join(base_path, "downloaded_pdfs")
    os.makedirs(base_output_dir, exist_ok=True)
    download_info_df = main_download(df_processed, base_output_dir=base_output_dir, to_path=to_path)
    print(download_info_df.head())
    print(f"Downloaded PDFs saved to '{to_path}'.")
