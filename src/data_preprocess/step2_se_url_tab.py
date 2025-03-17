# -*- coding: utf-8 -*-
"""
Author: Zhengyuan Dong
Created: 2025-02-24
Last Modified: 2025-03-16
Description: Enhanced title extraction tool for pdf_links and github_links.
             1. Extract links from pdf_links and github_links columns, remove duplicates.
             2. For arxiv.org links, extract IDs, then batch query titles.
             3. For biorxiv.org / medrxiv.org links, try to extract ID and query via API, otherwise fallback to HTML.
             4. For GitHub links, do HTTP or API with special handling for large files, plus post-process the GitHub page title.
             5. Filter invalid links by extension, track domain counts.
             6. For other domains, do normal HTML-based approach for titles.
"""

import os, re, json, time, io, logging, requests
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from xml.etree import ElementTree as ET
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
from src.utils import load_config
from joblib import Parallel, delayed

tqdm.pandas()

logging.basicConfig(
    filename='title_extraction.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

HEADERS = {"User-Agent": "Mozilla/5.0"}
ARXIV_IGNORE_IDS = ["1910.09700"]
MAX_WORKERS = 8
MAX_RETRIES = 3
TIMEOUT = 15
GITHUB_SIZE_THRESHOLD = 5 * 1024 * 1024

# ------------- Block: Extensions to skip -------------
SKIP_EXTENSIONS = {
    '.png', '.jpg', '.jpeg', '.gif', '.zip', '.rar', '.7z', '.doc', '.docx',
    '.xls', '.xlsx', '.ppt', '.pptx', '.mp4', '.mp3', '.md', '.pth', '.pt',
    '.ckpt', '.safetensors'
}
# ------------- End Block -------------

# ------------- Block: Utility functions -------------
def clean_url(url):
    """
    Clean a URL by:
      1. Stripping leading/trailing whitespace.
      2. If the URL contains Markdown format indicators like "](", split and take the first part.
      3. Remove extraneous surrounding characters like {}、[]、() from both ends.
    """
    cleaned = url.strip()
    if "](" in cleaned:
        cleaned = cleaned.split("](")[0]
    cleaned = cleaned.strip("{}[]()")
    return cleaned
# ------------- End Block -------------

# ------------- Block: ID extraction functions -------------
def extract_arxiv_id(url):
    # If "bioarxiv" in url, skip to avoid confusion
    if "bioarxiv" in url.lower():
        return None
    pattern = r'arxiv\.org/(?:pdf|abs|src|ops)/([\w\.-]+)'
    m = re.search(pattern, url)
    if m:
        full_id = m.group(1)
        return full_id.split("v")[0]
    return None

def extract_domain(url):
    try:
        if not re.search(r'https?://[^/]+/', url):
            url = url.strip() + "/"
        m = re.search(r'https?://([^/]+)/', url)
        if m:
            return m.group(1).lower()
    except Exception as e:
        logging.error(f"Error extracting domain from {url}: {e}")
    return None

def extract_links_from_columns(df, cols):
    # Modified process_cell to clean each URL using clean_url function
    def process_cell(x):
        cleaned_links = []
        if isinstance(x, (list, tuple, np.ndarray)):
            for s in x:
                cs = clean_url(s)
                if cs:
                    cleaned_links.append(cs)
            return cleaned_links
        elif isinstance(x, str) and x.strip():
            for s in x.split(","):
                cs = clean_url(s)
                if cs:
                    cleaned_links.append(cs)
            return cleaned_links
        else:
            return []
    results = Parallel(n_jobs=-1)(
        delayed(lambda col: df[col].apply(process_cell).tolist())(col)
        for col in tqdm(cols, desc="Processing columns")
    )
    combined_links = [link for col in results for cell in col for link in cell]
    return list(set(combined_links))
# ------------- End Block -------------

# ------------- Block: Cache functions -------------
def load_cache(cache_path):
    """Load cache data from a JSON file; if not exists, return a defaultdict."""
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Failed to load cache {cache_path}: {str(e)}")
    return defaultdict(str)

def save_cache(cache_path, data):
    """Save cache data to a JSON file."""
    try:
        with open(cache_path, 'w') as f:
            json.dump(data, f)
    except Exception as e:
        logging.error(f"Failed to save cache {cache_path}: {str(e)}")
# ------------- End Block -------------

# ------------- Block: Fetch with retry and batch fetch -------------
def fetch_with_retry(func, item, delay=1):
    """
    Common retry mechanism to call 'func(item)' up to MAX_RETRIES times.
    If it still fails, return "".
    """
    for attempt in range(MAX_RETRIES):
        try:
            result = func(item)
            if result:
                return result
        except Exception as e:
            logging.warning(f"Attempt {attempt+1} failed for {item}: {str(e)}")
            time.sleep(delay * (2 ** attempt))
    return ""

def batch_fetch(items, fetch_func, cache, cache_path, desc):
    """
    Generic batch fetch interface.
    Only items not in cache or with placeholder values ("Just a moment...", "just waiting")
    will be re-fetched.
    """
    # ------------- Check cache: if value is placeholder, treat as not cached -------------
    placeholder_values = {"just a moment...", "just waiting"}
    new_items = [item for item in items if (item not in cache) or (cache[item].strip().lower() in placeholder_values)]
    if not new_items:
        print(f"No new items to fetch: {desc}")
        return cache

    print(f"Fetching {len(new_items)} new items: {desc}")
    progress_bar = tqdm(total=len(new_items), desc=desc)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(fetch_with_retry, fetch_func, item): item for item in new_items}
        for future in as_completed(futures):
            item = futures[future]
            try:
                result = future.result()
                # ------------- If result is placeholder, do not cache (store empty string) -------------
                if result and result.strip().lower() in placeholder_values:
                    cache[item] = ""
                else:
                    cache[item] = result
            except Exception as e:
                logging.error(f"Failed processing {item}: {str(e)}")
                cache[item] = ""
            progress_bar.update(1)
            save_cache(cache_path, cache)
    progress_bar.close()
    return cache
# ------------- End Block -------------

# ------------- Block: arXiv batch fetching -------------
def batch_fetch_arxiv_titles(arxiv_ids, chunk_size=29999, delay_between_chunks=3):
    """
    Batch fetch arXiv titles.
    Returns a dict {arxiv_id: title}
    """
    results = {}
    total = len(arxiv_ids)
    if total == 0:
        return results
    print(f"Batch fetching {total} arXiv IDs with chunk_size={chunk_size}")
    for i in range(0, total, chunk_size):
        chunk = arxiv_ids[i: i+chunk_size]
        id_list_str = ",".join(chunk)
        api_url = f'http://export.arxiv.org/api/query?id_list={id_list_str}'
        try:
            response = requests.get(api_url, headers=HEADERS, timeout=TIMEOUT)
            response.raise_for_status()
            root = ET.fromstring(response.text)
            for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
                title_element = entry.find('{http://www.w3.org/2005/Atom}title')
                id_element = entry.find('{http://www.w3.org/2005/Atom}id')
                if title_element is not None and id_element is not None:
                    full_id = id_element.text.strip()
                    if full_id.startswith("http://arxiv.org/abs/"):
                        bare_id = full_id[len("http://arxiv.org/abs/"):]
                        bare_id = bare_id.split("v")[0]
                    else:
                        bare_id = full_id
                    title_text = title_element.text.strip()
                    results[bare_id] = title_text
                    logging.info(f"Retrieved arXiv title for {bare_id}: {title_text}")
                else:
                    logging.warning("Missing title or id element in an arXiv entry.")
        except Exception as e:
            logging.error(f"Failed to fetch arXiv titles for chunk index={i}: {str(e)}")
        if i + chunk_size < total:
            time.sleep(delay_between_chunks)
    return results
# ------------- End Block -------------

# ------------- Block: Preprint ID extraction and API fetch -------------
def extract_biorxiv_id(url):
    """
    Extract the bioRxiv numeric ID from the URL.
    
    Primary extraction: Look for a pattern where after a slash (optionally preceded by "10.1101/")
    there is a date-like numeric combination in the format YYYY.MM.DD.xxxxxx (exactly 6 digits at the end),
    optionally followed by a version indicator (e.g., v1) or a file extension (e.g., .pdf).
    Only the numeric combination is returned.
    
    Fallback extraction: If the primary pattern is not found, match a simpler pattern where after a slash 
    (optionally preceded by "10.1101/"), there is a sequence of digits (optionally followed by a version indicator).
    
    Examples:
      https://www.biorxiv.org/content/early/2024/07/03/10.1101/2024.05.07.593067v1 
         → returns "2024.05.07.593067"
      https://www.biorxiv.org/content/10.1101/622803v4 
         → returns "622803"
    """
    import re
    ######## Primary pattern: match a date-like numeric combination with exactly 6 digits after the last dot ########
    pattern_primary = r'/(?:10\.1101/)?(\d{4}\.\d{2}\.\d{2}\.\d{6})(?:[vV]\d+)?(?:\.\w+)?'
    match = re.search(pattern_primary, url)
    if match:
        return match.group(1)
    ######## Fallback pattern: match a sequence of digits ########
    pattern_fallback = r'/(?:10\.1101/)?(\d+)(?:[vV]\d+)?'
    match2 = re.search(pattern_fallback, url)
    if match2:
        return match2.group(1)
    return None

def extract_medrxiv_id(url):
    """
    Extract the medRxiv numeric ID from the URL.
    
    Primary extraction: Look for a pattern where after a slash (optionally preceded by "10.1101/")
    there is a date-like numeric combination in the format YYYY.MM.DD.xxxxxxxx (exactly 8 digits at the end),
    optionally followed by a version indicator (e.g., v1) or a file extension (e.g., .pdf).
    Only the numeric combination is returned.
    
    Fallback extraction: If the primary pattern is not found, match a simpler pattern where after a slash 
    (optionally preceded by "10.1101/"), there is a sequence of digits (optionally followed by a version indicator).
    
    Examples:
      https://www.medrxiv.org/content/early/2024/07/03/10.1101/2023.01.01.12345678v3 
         → returns "2023.01.01.12345678"
      https://www.medrxiv.org/content/10.1101/759696v1 
         → returns "759696"
    """
    import re
    ######## Primary pattern: match a date-like numeric combination with exactly 8 digits after the last dot ########
    pattern_primary = r'/(?:10\.1101/)?(\d{4}\.\d{2}\.\d{2}\.\d{8})(?:[vV]\d+)?(?:\.\w+)?'
    match = re.search(pattern_primary, url)
    if match:
        return match.group(1)
    ######## Fallback pattern: match a sequence of digits ########
    pattern_fallback = r'/(?:10\.1101/)?(\d+)(?:[vV]\d+)?'
    match2 = re.search(pattern_fallback, url)
    if match2:
        return match2.group(1)
    return None

def fetch_biorxiv_title_via_api(url):
    """
    If we can extract a valid bioRxiv ID, call the bioRxiv API to get the title.
    If no ID is found or API fails, return "".
    """
    try:
        bio_id = extract_biorxiv_id(url)
        if not bio_id:
            print(f"No valid bioRxiv ID found in {url}. Fallback to HTML.")
            return ""
        #print(f"Extracted biorxiv ID: {bio_id}")
        api_url = f"https://api.biorxiv.org/details/biorxiv/{bio_id}"
        resp = requests.get(api_url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data.get("collection"):
            title = data["collection"][0].get("title", "")
            return title.strip()
        return ""
    except Exception as e:
        print(f"API error for {url}: {e}")
        return ""

def fetch_medrxiv_title_via_api(url):
    """
    Similar logic for medRxiv.
    """
    try:
        med_id = extract_medrxiv_id(url)
        if not med_id:
            print(f"No valid medRxiv ID found in {url}. Fallback to HTML.")
            return ""
        #print(f"Extracted medrxiv ID: {med_id}")
        api_url = f"https://api.biorxiv.org/details/medrxiv/{med_id}"
        resp = requests.get(api_url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data.get("collection"):
            title = data["collection"][0].get("title", "")
            return title.strip()
        return ""
    except Exception as e:
        print(f"API error for {url}: {e}")
        return ""
# ------------- End Block -------------

# ------------- Block: URL title fetching functions -------------
def fetch_url_title(url):
    """
    Fallback or universal function to get the page title from HTML or PDF metadata.
    For GitHub, skip large files. For PDF, read via PyPDF2. Otherwise parse HTML.
    """
    try:
        domain = extract_domain(url)
        if domain and "github.com" in domain:
            # HEAD request to check content length
            head_resp = requests.head(url, headers=HEADERS, timeout=TIMEOUT)
            cl = head_resp.headers.get("Content-Length")
            if cl and int(cl) > GITHUB_SIZE_THRESHOLD:
                logging.info(f"Skipping {url} due to large size: {cl} bytes")
                return ""
        if url.lower().endswith('.pdf'):
            resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
            with io.BytesIO(resp.content) as pdf_file:
                pdf_title = PdfReader(pdf_file).metadata.get('/Title', '')
                return pdf_title.strip()
        else:
            resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
            soup = BeautifulSoup(resp.text, 'html.parser')
            # Attempt a DC.Title meta
            meta_title = soup.find('meta', {'name': 'DC.Title'})
            if meta_title and meta_title.get('content'):
                return meta_title['content'].strip()
            if soup.title and soup.title.string:
                page_title = soup.title.string.strip()
                # Postprocess if it's GitHub
                if domain and "github.com" in domain:
                    return postprocess_github_title(page_title)
                return page_title
            return ''
    except Exception as e:
        logging.error(f"Failed to fetch HTML title for {url}: {e}")
        return ""

def is_invalid_extension(url):
    """
    Check if the URL ends with a certain file extension that we consider invalid.
    """
    return any(url.lower().endswith(ext) for ext in SKIP_EXTENSIONS)

def fetch_github_info(url):
    """
    Example toy function to use GitHub API to fetch repo info.
    Note: GitHub API does exist, but this simple implementation might be slow.
    We'll address GitHub performance later.
    """
    try:
        match = re.match(r'https?://github\.com/([^/]+)/([^/]+)', url)
        if not match:
            return {}
        owner, repo = match.group(1), match.group(2)
        api_url = f"https://api.github.com/repos/{owner}/{repo}"
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(api_url, headers=headers, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            return {"description": data.get("description", "").strip()}
        else:
            logging.warning(f"GitHub API error {resp.status_code} for {url}")
            return {}
    except Exception as e:
        logging.error(f"GitHub fetch error for {url}: {e}")
        return {}
# ------------- End Block -------------

# ------------- Block: Main function -------------
def main():
    config = load_config('config.yaml')
    processed_base_path = os.path.join(config.get('base_path'), 'processed')
    data_type = 'modelcard'

    print("⚠️ Step 1: Loading data from parquet (modelcard_step1)...")
    df = pd.read_parquet(
        os.path.join(processed_base_path, f"{data_type}_step1.parquet"),
        columns=['modelId', 'card_tags', 'github_link', 'pdf_link']
    )
    
    print("Step 2: Extracting links from columns (pdf_link, github_link)")
    all_links = extract_links_from_columns(df, cols=["pdf_link", "github_link"])
    print(f"Total unique raw links: {len(all_links)}")

    # ------------- Block: Classify links -------------
    print("Step 3: Classifying links...")
    records = []
    for link in tqdm(all_links, desc="Classify links"):
        link_lower = link.lower()
        domain = extract_domain(link)
        is_invalid = is_invalid_extension(link)
        if is_invalid:
            category = "invalid"
            handler = "none"
        else:
            if ("arxiv" in link_lower) and ("biorxiv" not in link_lower):
                category = "arxiv"
                handler = "arxiv_batch"
            elif ("biorxiv" in link_lower) or ("medrxiv" in link_lower):
                category = "biorxiv_or_medrxiv"
                handler = "rxiv_api_or_html"
            elif "github.com" in link_lower:
                category = "github"
                handler = "github_api_or_html"
            else:
                category = "other"
                handler = "html_parser"
        records.append({
            "link": link,
            "domain": domain,
            "category": category,
            "handler": handler,
            "invalid": is_invalid
        })
    # ------------- End Block -------------

    df_links = pd.DataFrame(records)
    df_links.to_csv("all_links_with_category.csv", index=False)
    print("✅ Saved 'all_links_with_category.csv'. Checking valid links...")

    # ------------- Block: Filter and partition valid links -------------
    valid_df = df_links[df_links["invalid"] == False].copy()
    print(f"Valid links: {len(valid_df)}")
    arxiv_df = valid_df[valid_df["category"] == "arxiv"]
    rxiv_df = valid_df[valid_df["category"] == "biorxiv_or_medrxiv"]
    github_df = valid_df[valid_df["category"] == "github"]
    other_df = valid_df[valid_df["category"] == "other"]
    # ------------- End Block -------------

    # ------------- Block: arXiv batch fetch -------------
    print("Step 4A: arXiv batch fetch")
    arxiv_ids = []
    for link in arxiv_df["link"]:
        if "biorxiv" in link.lower():
            continue
        aid = extract_arxiv_id(link)
        if aid:
            arxiv_ids.append(aid)
    arxiv_ids = list(set(arxiv_ids))
    print(f"Found {len(arxiv_ids)} arXiv IDs")
    if arxiv_ids:
        arxiv_titles = batch_fetch_arxiv_titles(arxiv_ids, chunk_size=29999, delay_between_chunks=3)
        for i, (k,v) in enumerate(arxiv_titles.items()):
            print(f"arXiv ID={k}, title={v}")
            if i>4: break
    # ------------- End Block -------------

    # ------------- Block: bioRxiv/medRxiv fetch -------------
    print("Step 4B: biorxiv/medrxiv single-API or fallback HTML")
    def fetch_rxiv_title(url):
        link_lower = url.lower()
        if "biorxiv" in link_lower:
            t = fetch_biorxiv_title_via_api(url)
            if not t:
                t = fetch_url_title(url)
            return t
        elif "medrxiv" in link_lower:
            t = fetch_medrxiv_title_via_api(url)
            if not t:
                t = fetch_url_title(url)
            return t
        else:
            return ""
    rxiv_title_map = {}
    for link in tqdm(rxiv_df["link"], desc="Fetching Rxiv Titles"):
        rxiv_title_map[link] = fetch_rxiv_title(link)
    print("Sample biorxiv/medrxiv titles:")
    for i, (lk, ttl) in enumerate(rxiv_title_map.items()):
        print(f"{lk}\n   -> {ttl}")
        if i>4: break
    # ------------- End Block -------------

    # ------------- Block: GitHub fetch -------------
    print("Step 4C: GitHub API fetch (toy example)")
    github_title_map = {}
    for link in tqdm(github_df["link"], desc="Fetching GitHub Info"):
        info = fetch_github_info(link)
        desc_title = info.get("description") if isinstance(info, dict) else ""
        if not desc_title:
            desc_title = fetch_url_title(link)
        github_title_map[link] = desc_title
    print("Sample GitHub titles:")
    for i, (lk, ttl) in enumerate(github_title_map.items()):
        print(f"{lk}\n   -> {ttl}")
        if i>4: break
    # ------------- End Block -------------

    # ------------- Block: Other domains fetch -------------
    print("Step 4D: other domains => HTML fetch")
    other_title_map = {}
    for link in tqdm(other_df["link"], desc="Fetching other HTML Titles"):
        other_title_map[link] = fetch_url_title(link)
    # ------------- End Block -------------

    print("✅ Done with fetching. You can now merge results back if desired, or save as JSON.")
    with open("rxiv_title_map.json", "w") as f:
        json.dump(rxiv_title_map, f, indent=2)
    with open("github_title_map.json", "w") as f:
        json.dump(github_title_map, f, indent=2)
    with open("other_title_map.json", "w") as f:
        json.dump(other_title_map, f, indent=2)

    # ------------- Block: Merge extracted titles back to DataFrame -------------
    def get_extracted_title(row):
        link = row["link"]
        cat = row["category"]
        if cat == "arxiv":
            aid = extract_arxiv_id(link)
            return arxiv_titles.get(aid, "") if aid else ""
        elif cat == "biorxiv_or_medrxiv":
            return rxiv_title_map.get(link, "")
        elif cat == "github":
            return github_title_map.get(link, "")
        elif cat == "other":
            return other_title_map.get(link, "")
        else:
            return ""
    df_links["extracted_title"] = df_links.apply(get_extracted_title, axis=1)
    
    url_title_dict = pd.Series(df_links.extracted_title.values, index=df_links.link).to_dict()
    def update_cell(cell):
        if isinstance(cell, (list, tuple, np.ndarray)):
            return {link: url_title_dict.get(link, "") for link in cell}
        elif isinstance(cell, str) and cell.strip():
            links = [s.strip() for s in cell.split(",")]
            return {link: url_title_dict.get(link, "") for link in links}
        else:
            return {}
    
    df["extracted_titles"] = df.apply(lambda row: {
        "pdf_link": update_cell(row["pdf_link"]),
        "github_link": update_cell(row["github_link"])
    }, axis=1)
    # ------------- End Block -------------

    # ------------- Block: Save updated DataFrame -------------
    output_path = os.path.join(processed_base_path, f"{data_type}_with_extracted_titles.parquet")
    df.to_parquet(output_path, index=False)
    print(f"✅ Updated DataFrame with extracted titles saved to {output_path}")
    # ------------- End Block -------------

    print("All done. Exiting main().")

if __name__ == "__main__":
    main()


"""

"""