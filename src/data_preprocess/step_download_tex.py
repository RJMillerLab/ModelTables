
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
import subprocess
from tqdm import tqdm
from joblib import Parallel, delayed
import re
from src.utils import load_config
import feedparser

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"

def extract_arxiv_id(url):
    match = re.search(r'(\d{4}\.\d{5})', url)
    return match.group(1) if match else None

def is_valid_tar_gz(file_path):
    """Check if the file is a valid .tar.gz archive."""
    try:
        with tarfile.open(file_path, "r:gz") as tar:
            tar.getmembers()  # Try to list archive contents
        return True
    except (tarfile.TarError, EOFError) as e:
        print(f"Error reading tar.gz file {file_path}: {e}")
        return False

def fetch_arxiv_entries(query, max_results=5):
    """
    Search for arXiv entries using the arXiv API.
    """
    base_url = 'http://export.arxiv.org/api/query?'
    search_query = f'search_query={query}&start=0&max_results={max_results}'
    response = requests.get(base_url + search_query)
    response.raise_for_status()
    return feedparser.parse(response.content)

def fetch_arxiv_version(arxiv_id):
    api_url = f'http://export.arxiv.org/api/query?id_list={arxiv_id}'
    headers = {'User-Agent': USER_AGENT}
    response = requests.get(api_url, headers=headers)
    response.raise_for_status()
    feed = feedparser.parse(response.content)
    if not feed.entries:
        raise ValueError("No entry found for arXiv ID.")
    entry = feed.entries[0]
    if 'id' not in entry:
        raise ValueError("Invalid API response.")
    # Extract version from the id (e.g., http://arxiv.org/abs/1234.5678v2)
    version = entry.id.split('v')[-1]
    return version

def download_arxiv_source(arxiv_id, output_dir='downloads'):
    #src_url = f'https://arxiv.org/e-print/{arxiv_id}'
    #src_url = f'https://arxiv.org/src/{arxiv_id}' # this need v1, v2 to work, otherwise we only download the html
    api_url = f'http://export.arxiv.org/api/query?id_list={arxiv_id}'
    response = requests.get(api_url)
    response.raise_for_status()
    entry = feedparser.parse(response.content).entries[0]
    if 'id' in entry:
        pass
    else:
        return None
    version = entry.id.split('v')[-1]  # Extract version from the ID
    src_url = f'https://arxiv.org/src/{arxiv_id}v{version}'  # Construct the URL with version
    print(src_url)
    #response = requests.get(src_url, stream=True)
    #if response.status_code == 404:
    #    print(f"Warning: {src_url} not found.")
    #    return None  # Return None if the file is not found
    #response.raise_for_status()
    #arxiv_id = arxiv_id.replace('/', '_')
    file_path = os.path.join(output_dir, f'{arxiv_id}v{version}.tar.gz')
    #file_path = os.path.join(output_dir, f'{arxiv_id}.tar.gz')
    try:
        subprocess.run([
            'wget', 
            src_url, 
            '-O', file_path, 
            '--header', 'User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            '--header', 'Referer: https://arxiv.org/'
        ], check=True)
        print(f"Downloaded {file_path}")
        return file_path
    except subprocess.CalledProcessError as e:
        print(f"Failed to download {src_url}: {e}")
        return None
    #with open(file_path, 'wb') as file:
    #    for chunk in response.iter_content(chunk_size=8192):
    #        file.write(chunk)
    return file_path

def download_tex_source(arxiv_id, output_dir="downloads"):
    os.makedirs(output_dir, exist_ok=True)
    try:
        version = fetch_arxiv_version(arxiv_id)
    except Exception as e:
        print(f"Error retrieving version for {arxiv_id}: {e}")
        return None
    src_url = f'https://arxiv.org/src/{arxiv_id}v{version}'
    print(f"Downloading from: {src_url}")
    output_file = os.path.join(output_dir, f'{arxiv_id}v{version}.tar.gz')
    try:
        subprocess.run([
            'wget',
            src_url,
            '-O', output_file,
            '--header', f'User-Agent: {USER_AGENT}',
            '--header', 'Referer: https://arxiv.org/'
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Download failed for {src_url}: {e}")
        return None
    print(f"Downloaded file to: {output_file}")
    return output_file

def download_tex_row(row, base_output_dir):
    url = row['final_url']
    arxiv_id = extract_arxiv_id(url)
    if arxiv_id:
        model_id = row['modelId']
        downloaded_path = download_tex_source(arxiv_id, output_dir=base_output_dir)
        return {
            "modelId": model_id,
            "final_url": url,
            "local_path": downloaded_path if downloaded_path else None
        }
    return None

def main_download(df, base_output_dir="downloaded_tex_files", to_path="data/downloaded_tex_info.parquet"):
    assert 'final_url' in df.columns, "Missing 'final_url' column in DataFrame."
    assert 'modelId' in df.columns, "Missing 'modelId' column in DataFrame."    
    download_info = Parallel(n_jobs=-1)(
        delayed(download_tex_row)(row, base_output_dir) for _, row in tqdm(df.iterrows(), total=len(df))
    )
    download_info = [info for info in download_info if info is not None]
    download_info_df = pd.DataFrame(download_info)
    download_info_df.to_parquet(to_path, index=False)
    print(f"Downloaded {len(download_info)} .tar.gz files.")
    print(f"Skipped {len(df) - len(download_info)} .tar.gz files.")
    return download_info_df

if __name__ == "__main__":
    config = load_config('config.yaml')
    processed_base_path = os.path.join(config.get('base_path'), "processed")
    df_processed = pd.read_parquet(os.path.join(processed_base_path, "processed_final_urls.parquet"))
    df_processed = df_processed[df_processed['final_url'].notnull()].copy()
    to_path = os.path.join(processed_base_path, "downloaded_tex_info.parquet")
    base_output_dir = os.path.join(config.get('base_path'),"downloaded_tex_files")
    os.makedirs(base_output_dir, exist_ok=True)
    download_info_df = main_download(df_processed, base_output_dir=base_output_dir, to_path=to_path)
    print(download_info_df.head())
    print(f"Downloaded .tar.gz files saved to '{to_path}'.")
