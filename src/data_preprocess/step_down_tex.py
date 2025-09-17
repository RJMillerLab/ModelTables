
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author: Zhengyuan Dong
Created: 2025-02-24
Last Modified: 2025-02-24
Description: Download .tar.gz files from arXiv links extracted from input CSV.
"""

import os, re, arxiv
import pandas as pd
from tqdm import tqdm
import requests
import tarfile
import subprocess
from joblib import Parallel, delayed
from src.utils import load_config
import feedparser
from stem.control import Controller

TOR_PROXY = "socks5h://127.0.0.1:9050"
PROXIES = {"http": TOR_PROXY, "https": TOR_PROXY}
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"

def untar(fname, output_dir):
    if not os.path.exists(fname):
        print(f"❌ File not found: {fname}")
        return False
    try:
        os.makedirs(output_dir, exist_ok=True)
        with tarfile.open(fname, "r:gz") as tar:
            tar.extractall(path=output_dir)
        print(f"✅ Unzip successfully: {output_dir}")
        return True
    except tarfile.ReadError:
        print(f"❌ Not a valid tar.gz file: {fname}")
    except Exception as e:
        print(f"❌ Failed to extract {fname}: {e}")
    return False

def change_tor_ip():
    with Controller.from_port(port=9051) as controller:
        controller.authenticate()
        controller.signal(2)
        print("Change IP")
        time.sleep(5)

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

def download_tex_source(arxiv_id, output_dir="downloads"):
    os.makedirs(output_dir, exist_ok=True)
    try:
        version = fetch_arxiv_version(arxiv_id)
    except Exception as e:
        print(f"Error retrieving version for {arxiv_id}: {e}")
        return {"tex": None, "html": None}
    tex_url = f'https://arxiv.org/src/{arxiv_id}v{version}'
    html_url = f'https://arxiv.org/abs/{arxiv_id}v{version}'
    print(f"Downloading TEX from: {tex_url}")
    print(f"Downloading HTML from: {html_url}")
    tex_output_file = os.path.join(output_dir, f'{arxiv_id}v{version}_tex.tar.gz')
    html_output_file = os.path.join(output_dir, f'{arxiv_id}v{version}_abs.html')
    extract_path = os.path.join(output_dir, f"{arxiv_id}v{version}")
    #change_tor_ip()
    #time.sleep(random.uniform(5, 10))
    # Download TEX source file (.tar.gz)
    """try:
        subprocess.run([
            'wget',
            tex_url,
            '-O', tex_output_file,
            '--header', f'User-Agent: {USER_AGENT}',
            '--header', 'Referer: https://arxiv.org/'
        ], check=True)
        print(f"Downloaded TEX file to: {tex_output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Download failed for {tex_url}: {e}")
        tex_output_file = None
    # Download HTML abstract page
    try:
        subprocess.run([
            'wget',
            html_url,
            '-O', html_output_file,
            '--header', f'User-Agent: {USER_AGENT}',
            '--header', 'Referer: https://arxiv.org/'
        ], check=True)
        print(f"Downloaded HTML file to: {html_output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Download failed for {html_url}: {e}")
        html_output_file = None
    """
    if os.path.exists(tex_output_file): # cache mechanism
        if is_valid_tar_gz(tex_output_file):
            print(f"✅ Cache hit: Using cached TEX file {tex_output_file}")
            return {"tex": tex_output_file, "html": html_output_file}
        else:
            print(f"⚠️ Cache invalid: Removing corrupt TEX file {tex_output_file}")
            os.remove(tex_output_file)

    if download_file(tex_url, tex_output_file): # and is_valid_tar_gz(tex_output_file):
        print(f"✅ Successfully downloaded and untared: {tex_output_file}")
        if is_valid_tar_gz(tex_output_file):
            print(f"✅ File is a valid tar.gz: {tex_output_file}")
            if untar(tex_output_file, extract_path):
                print(f"🗑️ Deleting original .tar.gz: {tex_output_file}")
                os.remove(tex_output_file)
            else:
                print(f"⚠️ Failed to extract {tex_output_file}")
        else:
            print(f"❌ Invalid tar.gz file: {tex_output_file}")
            tex_output_file = None
        #if untar(tex_output_file, extract_path):
            #os.remove(tex_output_file)
            #print(f"🗑️ Deleting original .tar.gz: {tex_output_file}")
    else:
        tex_output_file = None
    if not download_file(html_url, html_output_file):
        html_output_file = None
    return {"tex": tex_output_file, "html": html_output_file}

def download_tex_row(row, base_output_dir):
    url = row['final_url']
    arxiv_id = extract_arxiv_id(url)
    if arxiv_id:
        model_id = row['modelId']
        try:
            downloads = download_tex_source(arxiv_id, output_dir=base_output_dir)
        except Exception as e:
            print(f"Error downloading TEX for {arxiv_id}: {e}")
            downloads = {"tex": None, "html": None}
        return {
            "modelId": model_id,
            "final_url": url,
            #"local_path": downloaded_path if downloaded_path else None,
            "local_tex_path": downloads["tex"],
            "local_html_path": downloads["html"]
        }
    return None

def download_file(url, output_path):
    headers = {"User-Agent": USER_AGENT, "Referer": "https://arxiv.org/"}
    try:
        #response = requests.get(url, headers=headers, proxies=PROXIES, stream=True)
        response = requests.get(url, headers=headers, stream=True)
        response.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"✅ Download successfully: {output_path}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"❌ Download failure {url}: {e}")
        return False

def main_download(df, base_output_dir="downloaded_tex_files", to_path="data/downloaded_tex_info.parquet"):
    assert 'final_url' in df.columns, "Missing 'final_url' column in DataFrame."
    assert 'modelId' in df.columns, "Missing 'modelId' column in DataFrame."    
    download_info = Parallel(n_jobs=-1)(
        delayed(download_tex_row)(row, base_output_dir) for _, row in tqdm(df.iterrows(), total=len(df))
    )
    download_info = [info for info in download_info if info is not None]
    download_info_df = pd.DataFrame(download_info)
    download_info_df.to_parquet(to_path, compression="zstd", engine="pyarrow", index=False)
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
