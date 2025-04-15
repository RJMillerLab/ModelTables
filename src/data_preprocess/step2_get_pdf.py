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
import hashlib
import requests
import subprocess
import pandas as pd
import pyarrow.parquet as pq
from urllib.parse import urlparse
from tqdm import tqdm

DOWNLOAD_MODE = "wget"

def is_valid_pdf(file_path):
    """
    Quickly check if a file is a valid PDF by reading its header.
    Returns True if valid, False otherwise.
    """
    try: 
        with open(file_path, "rb") as f:
            header = f.read(5)
            return header == b"%PDF-"
    except Exception as e:
        print(f"[ERROR] Checking PDF validity failed for {file_path}: {e}")
        return False 

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

def extract_domain(url):
    try:
        parsed = urlparse(url)
        return parsed.netloc.lower()
    except Exception as e:
        print("❌  Failed to extract domain from", url, ":", e)
        return "unknown"

def download_pdf(url, output_folder, mode=DOWNLOAD_MODE, max_retries=3, sleep_time=3, timeout=15):
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    url_hash = hashlib.sha256(url.encode('utf-8')).hexdigest()
    safe_filename = url_hash + ".pdf"
    local_path = os.path.join(output_folder, safe_filename)

    if os.path.isfile(local_path):
        print("📂  Retrieved local file for", url)
        return local_path

    attempts = 0
    while attempts < max_retries:
        if mode == "wget":
            try:
                subprocess.run(["wget", "-q", "-O", local_path, url], timeout=timeout)
                if os.path.isfile(local_path) and os.path.getsize(local_path) > 0:
                    print("✅  Downloaded (wget) for", url)
                    return local_path
                else:
                    print("❌  Error downloading", url)
            except Exception as e:
                print("❌  Error downloading", url, ":", e)
        else: 
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

def domain_round_robin_download(urls, output_folder, pdf_cache, cache_path):
    domain_groups = {}
    for url in urls:
        domain = extract_domain(url)
        domain_groups.setdefault(domain, []).append(url)

    total = len(urls)
    pbar = tqdm(total=total, desc="Downloading PDFs", unit="url")
    downloaded_paths = {}
    failed_urls = []
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
                save_json_cache(pdf_cache, cache_path)
            if not domain_groups.get(domain):
                del domain_groups[domain]
    pbar.close()
    return downloaded_paths, failed_urls

def main():
    parquet_path = "data/processed/extracted_annotations.parquet"
    if not os.path.isfile(parquet_path):
        print("❌  Parquet file not found:", parquet_path)
        return
    df_parquet = pd.read_parquet(parquet_path)
    if "extracted_openaccessurl" not in df_parquet.columns:
        print("❌  'extracted_openaccessurl' column not found in the parquet file.")
        return
    all_urls = set(df_parquet["extracted_openaccessurl"].dropna().unique())
    print("📄  Loaded", len(df_parquet), "rows from", parquet_path, "with", len(all_urls), "unique URLs.")

    pdf_cache_path = "data/processed/pdf_download_cache.json"
    pdf_cache = load_json_cache(pdf_cache_path)

    ######## Detect invalid PDFs even if cached ########
    valid_cache = {}
    invalid_urls = set()
    for url, pdf_path in pdf_cache.items():
        if pdf_path and os.path.isfile(pdf_path):
            if is_valid_pdf(pdf_path):
                valid_cache[url] = pdf_path
            else:
                print(f"[WARN] Invalid PDF detected, deleting: {pdf_path}")
                try:
                    os.remove(pdf_path)
                except Exception as e:
                    print(f"[ERROR] Failed to delete {pdf_path}: {e}")
                invalid_urls.add(url)
                valid_cache[url] = None
        else:
            invalid_urls.add(url)
            valid_cache[url] = None

    ######## Update cache with filtered results ########
    pdf_cache = valid_cache
    save_json_cache(pdf_cache, pdf_cache_path)  ######## <-- optional: persist cleaned cache

    ######## Compute missing URLs ########
    already_cached_valid = {url for url, path in pdf_cache.items() if path}
    missing = all_urls - already_cached_valid  ######## <-- only valid paths count
    print("📊  Total URLs:", len(all_urls))
    print(f"📂  Valid cached PDFs: {len(already_cached_valid)}")
    print(f"🧹  Invalid or corrupt PDFs: {len(invalid_urls)}")
    if invalid_urls:
        print(f"🗑️  Removed {len(invalid_urls)} invalid PDF files.")
    print(f"🆕  Missing (need fetch): {len(missing)}")
    output_folder = "downloaded_pdfs"
    downloaded_paths, failed_urls = domain_round_robin_download(missing, output_folder, pdf_cache, pdf_cache_path)
    if failed_urls:
        print("🔄  Retrying failed downloads for", len(failed_urls), "URLs...")
        _, failed_urls = domain_round_robin_download(failed_urls, output_folder, pdf_cache, pdf_cache_path)
    
    if failed_urls:
        print("❌  Final failed URLs:")
        for url in failed_urls:
            print("   ", url)
    else:
        print("🎉  All downloads succeeded.")
    ######## Final re-check: clean any newly corrupted PDFs ########  ########
    final_valid_cache = {}
    final_invalid_urls = set()
    for url, path in pdf_cache.items():
        if path and os.path.isfile(path):
            if is_valid_pdf(path):
                final_valid_cache[url] = path
            else:
                print(f"[FINAL WARN] Removing corrupted PDF: {path}")
                try:
                    os.remove(path)
                except Exception as e:
                    print(f"[FINAL ERROR] Failed to delete corrupted file {path}: {e}")
                final_invalid_urls.add(url)
                final_valid_cache[url] = None
        else:
            final_invalid_urls.add(url)
            final_valid_cache[url] = None
    pdf_cache = final_valid_cache
    save_json_cache(pdf_cache, pdf_cache_path)
    if final_invalid_urls:
        print(f"🧽  Final clean-up: removed {len(final_invalid_urls)} additional invalid PDFs.")
        print("📄  You can manually download and save the PDFs to the following paths:")
        for url in sorted(final_invalid_urls):
            pdf_name = hashlib.sha256(url.encode('utf-8')).hexdigest() + ".pdf"
            pdf_path = os.path.join("downloaded_pdfs", pdf_name)
            print(f"🔗 {url}  ->  📁 {pdf_path}")
    print("🎉  PDF download process complete. Cache now has", len(pdf_cache), "entries.")

if __name__ == "__main__":
    main()
