# -*- coding: utf-8 -*-
"""
Author: Zhengyuan Dong
Created: 2025-03-29
Description: Batch-download PDFs from a Parquet file containing "openaccessurl".
             Caches each downloaded PDF's file path in JSON to avoid re-downloading.
             Implements domain-based round-robin fetching to avoid rate limits.
             Uses URL hash for filename to ensure uniqueness and avoid duplicate downloads.
             Supports two download modes: 'wget' (default) and 'request'.
             If downloads fail, the failed URLs will be retried once.
"""

import os
import time
import json
import hashlib  # 用于生成 URL 哈希
import requests
import subprocess  # 用于 wget 模式
import pandas as pd
import pyarrow.parquet as pq
from urllib.parse import urlparse  # 用于域名提取
from tqdm import tqdm  # 添加进度条

DOWNLOAD_MODE = "wget"  # 默认下载模式 ("wget" 或 "request")

######## JSON cache load/save functions ########
def load_json_cache(file_path):
    if not os.path.isfile(file_path):
        print("⚠️  JSON cache file not found:", file_path)
        return {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print("✅  Loaded JSON cache from", file_path, "with", len(data), "entries.")
        return data
    except Exception as e:
        print("❌  Could not load JSON cache:", e)
        return {}

def save_json_cache(data, file_path):
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print("❌  Could not save JSON cache:", e)

######## Utility: extract domain ########
def extract_domain(url):
    try:
        parsed = urlparse(url)
        return parsed.netloc.lower()
    except Exception as e:
        print("❌  Failed to extract domain from", url, ":", e)
        return "unknown"

######## PDF download function ########
def download_pdf(url, output_folder, mode=DOWNLOAD_MODE, max_retries=3, sleep_time=3, timeout=15):
    """
    下载 URL 对应的 PDF，返回本地路径；若文件已存在，则直接返回。
    """
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    # 用 SHA256 生成唯一文件名
    url_hash = hashlib.sha256(url.encode('utf-8')).hexdigest()
    safe_filename = url_hash + ".pdf"
    local_path = os.path.join(output_folder, safe_filename)

    # 已存在则直接返回
    if os.path.isfile(local_path):
        print("📂  Retrieved local file for", url)
        return local_path

    attempts = 0
    while attempts < max_retries:
        if mode == "wget":
            try:
                # 使用 wget 下载
                subprocess.run(["wget", "-q", "-O", local_path, url], timeout=timeout)
                if os.path.isfile(local_path) and os.path.getsize(local_path) > 0:
                    print("✅  Downloaded (wget) for", url)
                    return local_path
                else:
                    print("❌  Error downloading", url)
            except Exception as e:
                print("❌  Error downloading", url, ":", e)
        else:  # request 模式
            try:
                headers = {
                    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.81 Safari/537.36"
                }
                resp = requests.get(url, timeout=timeout, headers=headers)
                if resp.status_code == 200 and resp.content:
                    with open(local_path, 'wb') as f:
                        f.write(resp.content)
                    print("✅  Downloaded (request) for", url)
                    return local_path
                else:
                    print("❌  Error downloading", url, "- status", resp.status_code)
            except Exception as e:
                print("❌  Error downloading", url, "on attempt", attempts+1, ":", e)

        attempts += 1
        if attempts < max_retries:
            time.sleep(sleep_time)
    print("❌  Failed to download after", max_retries, "attempts:", url)
    return None

######## Domain-based round-robin download function ########
def domain_round_robin_download(urls, output_folder, pdf_cache, cache_path):
    """
    按域名分组后采用轮转方式下载，更新缓存并用 tqdm 显示进度。
    返回一个 (downloaded_paths, failed_urls) 的元组。
    """
    # 分组
    domain_groups = {}
    for url in urls:
        domain = extract_domain(url)
        domain_groups.setdefault(domain, []).append(url)

    total = len(urls)
    pbar = tqdm(total=total, desc="Downloading PDFs", unit="url")
    downloaded_paths = {}
    failed_urls = []
    # 遍历各组
    while any(domain_groups.values()):
        for domain in list(domain_groups.keys()):
            if domain_groups[domain]:
                url = domain_groups[domain].pop(0)
                local_pdf_path = download_pdf(url, output_folder)
                pbar.update(1)
                if local_pdf_path is not None:
                    pdf_cache[url] = local_pdf_path
                    downloaded_paths[url] = local_pdf_path
                else:
                    failed_urls.append(url)
                save_json_cache(pdf_cache, cache_path)  # 每次更新缓存
            if not domain_groups.get(domain):
                del domain_groups[domain]
    pbar.close()
    return downloaded_paths, failed_urls

######## Main Script ########
def main():
    parquet_path = "extracted_annotations.parquet"
    if not os.path.isfile(parquet_path):
        print("❌  Parquet file not found:", parquet_path)
        return
    df_parquet = pd.read_parquet(parquet_path)
    if "extracted_openaccessurl" not in df_parquet.columns:
        print("❌  'extracted_openaccessurl' column not found in the parquet file.")
        return
    all_urls = set(df_parquet["extracted_openaccessurl"].dropna().unique())
    print("📄  Loaded", len(df_parquet), "rows from", parquet_path, "with", len(all_urls), "unique URLs.")

    pdf_cache_path = "pdf_download_cache.json"
    pdf_cache = load_json_cache(pdf_cache_path)
    # 检查缓存中存在且本地文件存在的 URL
    cached_urls = {url for url, path in pdf_cache.items() if path and os.path.isfile(path)}
    missing_urls = all_urls - cached_urls
    print("📊  Total URLs:", len(all_urls))
    print("📂  Already cached:", len(cached_urls))
    print("🆕  Missing (need fetch):", len(missing_urls))

    output_folder = "downloaded_pdfs"
    # 第一轮下载
    downloaded_paths, failed_urls = domain_round_robin_download(missing_urls, output_folder, pdf_cache, pdf_cache_path)
    
    # 如果有失败的链接，重试一次
    if failed_urls:
        print("🔄  Retrying failed downloads for", len(failed_urls), "URLs...")
        _, failed_urls = domain_round_robin_download(failed_urls, output_folder, pdf_cache, pdf_cache_path)
    
    if failed_urls:
        print("❌  Final failed URLs:")
        for url in failed_urls:
            print("   ", url)
    else:
        print("🎉  All downloads succeeded.")

    print("🎉  PDF download process complete. Cache now has", len(pdf_cache), "entries.")

if __name__ == "__main__":
    main()
