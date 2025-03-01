"""
Author: Zhengyuan Dong
Created: 2025-02-24
Last Modified: 2025-02-24
Description: Update URLs from the final URLs and save to a text file.
"""

import os
import re
import pandas as pd
import requests
from bs4 import BeautifulSoup
import hashlib
import json
import time
from tqdm import tqdm
from urllib.parse import urljoin

CACHE_FILE = "url_cache.json"

def load_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Warning: Cache file corrupted ({e}). Resetting cache.")
            return {}
    return {}

def save_cache(cache):
    temp_file = CACHE_FILE + ".tmp"
    with open(temp_file, "w") as f:
        json.dump(cache, f, default=str)
    os.replace(temp_file, CACHE_FILE)

cache = load_cache()

def get_cache_key(url):
    return hashlib.md5(url.encode()).hexdigest()

def get_open_access_pdf(doi):
    api_url = f"https://api.unpaywall.org/v2/{doi}?email=zydong122@gmail.com"
    try:
        response = requests.get(api_url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            pdf_url = response.json().get("best_oa_location", {}).get("url")
            if not pdf_url and "oa_locations" in data:
                for loc in data["oa_locations"]:
                    if loc.get("url") and loc["url"].endswith(".pdf"):
                        pdf_url = loc["url"]
                        break
            return pdf_url if pdf_url and pdf_url.endswith(".pdf") else None
    except Exception as e:
        print(f"Error fetching Open Access PDF: {e}")
    return None

def get_pdf_link(url):
    if url.endswith('.pdf'):
        return url
    
    cache_key = get_cache_key(url)
    if cache_key in cache:
        return cache[cache_key]
    
    result = None
    if "arxiv.org" in url:
        match = re.search(r'arxiv.org/(?:abs|pdf)/([\d\.]+)', url)
        result = f"https://arxiv.org/pdf/{match.group(1)}.pdf" if match else None
    elif "openreview.net" in url:
        match = re.search(r'openreview.net/forum\?id=([\w-]+)', url)
        result = f"https://openreview.net/pdf?id={match.group(1)}" if match else None
    elif "aclanthology.org" in url:
        match = re.search(r'aclanthology.org/(\S+)', url)
        result = f"https://aclanthology.org/{match.group(1)}.pdf" if match else None
    elif "doi.org" in url:
        doi_match = re.search(r'doi.org/([\w\d\.\/]+)', url)
        if doi_match:
            result = get_open_access_pdf(doi_match.group(1))
    elif url.startswith("http"):
        result = extract_pdf_from_html(url)
    """elif "github.io" in url:
        result = extract_github_html_table(url)
    elif "osf.io" in url:
        result = get_osf_pdf(url)
    elif "huggingface.co" in url:
        result = url
    elif "hal.science" in url:
        result = get_hal_pdf(url)"""
    
    cache[cache_key] = result if result else url
    save_cache(cache)
    return cache[cache_key]

def extract_pdf_from_html(url):
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            pdf_links = [a['href'] for a in soup.find_all('a', href=True) if re.search(r'\.pdf$', a['href'])]
            if pdf_links:
                return urljoin(url, pdf_links[0])
    except Exception as e:
        print(f"Error fetching or parsing HTML page {url}: {e}")
    return url 

def extract_github_html_table(url):
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            tables = soup.find_all('table')
            return str(tables[0]) if tables else None 
    except Exception as e:
        print(f"Error fetching GitHub.io HTML: {e}")
    return None

def get_osf_pdf(url):
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            pdf_link = soup.find('a', {'href': re.compile(r'.*\.pdf$')})
            if pdf_link and pdf_link.has_attr('href'):
                return urljoin(url, pdf_link['href'])
    except Exception as e:
        print(f"Error fetching OSF page: {e}")
    return None

def get_hal_pdf(url):
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            pdf_link = soup.find('a', {'href': re.compile(r'.*\.pdf$')})
            if pdf_link and pdf_link.has_attr('href'):
                return urljoin(url, pdf_link['href'])
    except Exception as e:
        print(f"Error fetching HAL page: {e}")
    return None

def save_urls_to_txt(input_parquet, output_txt):
    df = pd.read_parquet(input_parquet, columns=['final_url'])
    if 'final_url' not in df.columns:
        raise ValueError("Missing 'final_url' column in the input parquet file.")
    urls = df['final_url'].dropna().unique()
    processed_urls = [get_pdf_link(url) or url for url in tqdm(urls, desc="Processing URLs")]
    with open(output_txt, 'w') as f:
        for url in processed_urls:
            f.write(url + '\n')
    print(f"Saved {len(processed_urls)} processed URLs to {output_txt}")

if __name__ == "__main__":
    input_parquet = "data/processed/processed_final_urls.parquet"
    output_txt = "xx.txt"
    save_urls_to_txt(input_parquet, output_txt)
