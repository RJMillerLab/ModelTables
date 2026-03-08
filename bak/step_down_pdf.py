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
from concurrent.futures import ThreadPoolExecutor
from src.utils import load_config, to_parquet
import hashlib
import logging
from bs4 import BeautifulSoup
import re
from urllib.parse import urljoin

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

cache = {}
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}

def get_cache_key(url):
    return hashlib.md5(url.encode()).hexdigest()

def extract_pdf_from_html(url):
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            pdf_links = [a['href'] for a in soup.find_all('a', href=True) if re.search(r'\.pdf$', a['href'], re.IGNORECASE)]
            if pdf_links:
                return urljoin(url, pdf_links[0])
    except Exception as e:
        logging.error(f"Error fetching or parsing HTML page {url}: {e}")
    return None

def convert_to_pdf_url(url):
    if "arxiv.org/abs/" in url:
        return url.replace("arxiv.org/abs/", "arxiv.org/pdf/") + ".pdf"
    elif "doi.org" in url:
        return extract_pdf_from_html(url)
    return url

def download_pdf(url, output_path):
    cache_key = get_cache_key(url)
    if cache_key in cache:
        return cache[cache_key]
    if os.path.exists(output_path):
        cache[cache_key] = output_path
        return output_path
    
    pdf_url = convert_to_pdf_url(url)
    try:
        response = requests.get(pdf_url, headers=HEADERS, stream=True, timeout=10, allow_redirects=True)
        content_type = response.headers.get("Content-Type", "")
        
        if response.status_code == 200:
            if "application/pdf" in content_type:
                with open(output_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=1024):
                        f.write(chunk)
                cache[cache_key] = output_path
                return output_path
            elif "text/html" in content_type:
                extracted_pdf_url = extract_pdf_from_html(pdf_url)
                if extracted_pdf_url:
                    return download_pdf(extracted_pdf_url, output_path)
        
        logging.warning(f"Skipping: {pdf_url} (Status Code: {response.status_code}, Content-Type: {content_type})")
        return None
    except Exception as e:
        logging.error(f"Failed to download {pdf_url}: {e}")
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
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        download_info = list(tqdm(executor.map(download_row, [row for _, row in df.iterrows()]), total=len(df)))
    
    download_info = [info for info in download_info if info is not None]
    download_info_df = pd.DataFrame(download_info)
    to_parquet(download_info_df, to_path)
    
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
