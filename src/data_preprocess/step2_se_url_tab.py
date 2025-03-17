# -*- coding: utf-8 -*-
"""
Author: Zhengyuan Dong
Created: 2025-02-24
Last Modified: 2025-03-16
Description: Enhanced title extraction tool for pdf_links and github_links.
             1. Extract links from pdf_links and github_links columns, remove duplicates.
             2. For arxiv.org links, extract IDs, then batch query titles.
             3. For biorxiv.org / medrxiv.org links, try to extract ID and query via API, otherwise fallback to HTML.
             4. For GitHub links, use raw HTML extraction (avoid GitHub API) to extract title.
                For GitHub pages, if a BibTeX citation block is found, return it;
                otherwise extract the first Markdown header (i.e. a line starting with "# ");
                if not found, then use the <title> tag.
             5. Filter invalid links by extension, track domain counts.
             6. For other domains, use normal HTML-based approach for titles.
"""

import os, re, json, time, io, logging, requests
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from xml.etree import ElementTree as ET
from bs4 import BeautifulSoup
from joblib import Parallel, delayed
from src.utils import load_config
import html2text  ######## 引入html2text转换库

from PyPDF2 import PdfReader
class PdfReadError(Exception):
    pass

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

# PDF download folder
PDF_DOWNLOAD_FOLDER = "pdf_downloads"
if not os.path.exists(PDF_DOWNLOAD_FOLDER):
    os.makedirs(PDF_DOWNLOAD_FOLDER)

# Folder to save GitHub HTML pages
GITHUB_HTML_FOLDER = "github_html"
if not os.path.exists(GITHUB_HTML_FOLDER):
    os.makedirs(GITHUB_HTML_FOLDER)

# ------------- Block: Extensions to skip -------------
SKIP_EXTENSIONS = {
    'png', 'jpg', 'jpeg', 'gif', 'zip', 'rar', '7z', 'doc', 'docx', 'xls', 'xlsx', 'ppt', 'pptx', 'mp4', 'mp3', 'md', 'pth', 'pt',
    'ckpt', 'safetensors', 'json', 'py', 'ipynb', 'csv', 'tsv', "bin", "h5", "onnx", "pkl", "tar", "gz", "mov"
}
#{"json", "pth", "bin", "ckpt", "pt", "h5", "onnx", "py", "pkl", "tar", "gz", "gif", "png", "jpg", "jpeg", "csv", "ipynb", "mp4", "mov"}
# ------------- End Block -------------

# ------------- Block: Utility functions -------------
def clean_url(url):
    cleaned = url.strip()
    if "](" in cleaned:
        cleaned = cleaned.split("](")[0]
    cleaned = cleaned.strip("{}[]()")
    return cleaned

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

def is_invalid_extension(url):
    return any(url.lower().endswith(ext) for ext in SKIP_EXTENSIONS)

def extract_links_from_columns(df, cols):
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

# ------------- Block: PDF partial + full parse -------------
def pdf_title_partial_fetch(url, max_bytes=2048):
    try:
        headers = {"Range": f"bytes=0-{max_bytes-1}"}
        resp = requests.get(url, headers=headers, timeout=TIMEOUT)
        if resp.status_code not in (200, 206):
            return pdf_title_full_fetch(url)
        content = resp.content
        try:
            pdf_reader = PdfReader(io.BytesIO(content))
            metadata = pdf_reader.metadata
            title = metadata.get('/Title') if metadata else None
            if title:
                return title.strip()
            else:
                return pdf_title_full_fetch(url)
        except (PdfReadError, Exception):
            return pdf_title_full_fetch(url)
    except Exception:
        return pdf_title_full_fetch(url)

def pdf_title_full_fetch(url):
    try:
        resp = requests.get(url, timeout=TIMEOUT)
        resp.raise_for_status()
        file_basename = os.path.basename(url)
        local_pdf_path = os.path.join(PDF_DOWNLOAD_FOLDER, file_basename)
        with open(local_pdf_path, 'wb') as f:
            f.write(resp.content)
        pdf_reader = PdfReader(io.BytesIO(resp.content))
        metadata = pdf_reader.metadata
        title = metadata.get('/Title') if metadata else None
        return title.strip() if title else None
    except Exception as e:
        logging.error(f"Failed to fetch or parse PDF {url}: {e}")
        return None
# ------------- End Block -------------

# ------------- Block: ID extraction for arXiv, bioRxiv, medRxiv -------------
def extract_arxiv_id(url):
    if "biorxiv" in url.lower():
        return None
    pattern = r'arxiv\.org/(?:pdf|abs|src|ops)/([\w\.-]+)'
    m = re.search(pattern, url)
    if m:
        full_id = m.group(1)
        return full_id.split("v")[0]
    return None

def extract_biorxiv_id(url):
    pattern_primary = r'/(?:10\.1101/)?(\d{4}\.\d{2}\.\d{2}\.\d{6})(?:[vV]\d+)?(?:\.\w+)?'
    m = re.search(pattern_primary, url)
    if m:
        return m.group(1)
    pattern_fallback = r'/(?:10\.1101/)?(\d+)(?:[vV]\d+)?'
    m2 = re.search(pattern_fallback, url)
    if m2:
        return m2.group(1)
    return None

def extract_medrxiv_id(url):
    pattern_primary = r'/(?:10\.1101/)?(\d{4}\.\d{2}\.\d{2}\.\d{8})(?:[vV]\d+)?(?:\.\w+)?'
    m = re.search(pattern_primary, url)
    if m:
        return m.group(1)
    pattern_fallback = r'/(?:10\.1101/)?(\d+)(?:[vV]\d+)?'
    m2 = re.search(pattern_fallback, url)
    if m2:
        return m2.group(1)
    return None

def fetch_biorxiv_title_via_api(url):
    try:
        bid = extract_biorxiv_id(url)
        if not bid:
            return ""
        api_url = f"https://api.biorxiv.org/details/biorxiv/{bid}"
        resp = requests.get(api_url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data.get("collection"):
            return data["collection"][0].get("title", "").strip()
        return ""
    except Exception:
        return ""

def fetch_medrxiv_title_via_api(url):
    try:
        mid = extract_medrxiv_id(url)
        if not mid:
            return ""
        api_url = f"https://api.biorxiv.org/details/medrxiv/{mid}"
        resp = requests.get(api_url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data.get("collection"):
            return data["collection"][0].get("title", "").strip()
        return ""
    except Exception:
        return ""
# ------------- End Block -------------

# ------------- Block: arXiv batch fetch -------------
def batch_fetch_arxiv_titles(arxiv_ids, chunk_size=29999, delay_between_chunks=3):
    results = {}
    total = len(arxiv_ids)
    if total == 0:
        return results
    print(f"Batch fetching {total} arXiv IDs with chunk_size={chunk_size}")
    for i in range(0, total, chunk_size):
        chunk_ids = arxiv_ids[i:i+chunk_size]
        id_list_str = ",".join(chunk_ids)
        url = f'http://export.arxiv.org/api/query?id_list={id_list_str}'
        try:
            resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
            resp.raise_for_status()
            root = ET.fromstring(resp.text)
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
                    results[bare_id] = title_element.text.strip()
        except Exception as e:
            logging.error(f"Failed arXiv batch fetch chunk at i={i}: {e}")
        if i + chunk_size < total:
            time.sleep(delay_between_chunks)
    return results
# ------------- End Block -------------

# ------------- Block: GitHub fallback HTML fetching -------------
def postprocess_github_title(title):
    if not title:
        return title
    title = re.sub(r'^GitHub\s*-\s*[^:]+:\s*', '', title)
    title = re.sub(r'^GitHub\s*-\s*[^:]+', '', title)
    return title.strip()

def extract_github_bibtex(html_text):
    # 尝试提取完整的BibTeX引用块（贪婪匹配至第一个结尾的'}'加换行）
    m = re.search(r'@[\w]+\{[\s\S]+?\}\s', html_text)
    if m:
        return m.group(0).strip()
    return None

def url_to_filename(url):
    import hashlib
    h = hashlib.sha256(url.encode('utf-8')).hexdigest()
    return h + ".html"

def fetch_github_title_raw(url):
    EXCLUDED_TERMS = ['/issues', '/assets', '/sponsor', '/discussions', '/pull', '/tag', '/releases']
    EXCLUDED_SUFFIXES = ['LICENSE']
    for term in EXCLUDED_TERMS:
        if term in url:
            return "SkippedURL"
    for suffix in EXCLUDED_SUFFIXES:
        if url.endswith(suffix):
            return "SkippedURL"
    # 转换为 raw URL：对于 GitHub 链接（非gist）转换为 raw.githubusercontent.com 格式
    raw_url = url
    if 'github.com' in raw_url and 'gist.github.com' not in raw_url:
        raw_url = raw_url.replace('github.com', 'raw.githubusercontent.com')
        raw_url = raw_url.replace('blob/', '').replace('tree/', '')
        raw_url = raw_url.rstrip("/")
    local_file = os.path.join(GITHUB_HTML_FOLDER, url_to_filename(url))
    if os.path.exists(local_file):
        with open(local_file, 'r', encoding='utf-8') as f:
            html_text = f.read()
    else:
        try:
            resp = requests.get(raw_url, headers=HEADERS, timeout=TIMEOUT)
            html_text = resp.text
            with open(local_file, 'w', encoding='utf-8') as f:
                f.write(html_text)
            #time.sleep(2)  # 控制访问频率
        except Exception as e:
            logging.error(f"Error fetching GitHub URL {url}: {e}")
            return "SkippedURL"
    # 尝试提取BibTeX引用块
    bibtex = extract_github_bibtex(html_text)
    if bibtex:
        return bibtex
    # 将 HTML 转换为 Markdown
    md_text = html2text.html2text(html_text)
    # 尝试从 Markdown中提取第一个以 "# " 开头的标题
    m = re.search(r'^# (.+)$', md_text, re.MULTILINE)
    if m:
        return m.group(1).strip()
    # 如果没有，则退回直接提取 HTML中的<title>
    soup = BeautifulSoup(html_text, "html.parser")
    if soup.title and soup.title.string:
        return postprocess_github_title(soup.title.string.strip())
    return ""
    
def sequential_fetch_github_titles(links, delay=2):
    result = {}
    invalid_set = {"Access has been restricted", "SkippedURL", ""}
    for lk in tqdm(links, desc="Fetch GitHub Titles"):
        title = fetch_github_title_raw(lk)
        if title not in invalid_set:
            print(f"extracted: {lk} -> {title}")
        result[lk] = title
        time.sleep(delay)
    return result

# ------------- End Block -------------

# ------------- Block: Other domains fetch (Parallel) -------------
def process_other_title(lk):
    title = fetch_url_title(lk)
    print(f"extracted: {lk} -> {title}")
    return (lk, title)

def parallel_fetch_other_titles(links, n_jobs=4):
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_other_title)(lk) for lk in tqdm(links, desc="Fetch other Titles")
    )
    return {lk: title for lk, title in results}
# ------------- End Block -------------

def fetch_url_title(url):
    try:
        domain = extract_domain(url)
        if domain and "github.com" in domain:
            head = requests.head(url, headers=HEADERS, timeout=TIMEOUT)
            cl = head.headers.get("Content-Length")
            if cl and int(cl) > GITHUB_SIZE_THRESHOLD:
                logging.info(f"Skipping large GitHub file: {url}, size={cl} bytes")
                return ""
        if url.lower().endswith(".pdf"):
            return pdf_title_partial_fetch(url) or ""
        else:
            resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
            soup = BeautifulSoup(resp.text, "html.parser")
            meta_title = soup.find('meta', {'name': 'DC.Title'})
            if meta_title and meta_title.get('content'):
                return meta_title['content'].strip()
            if soup.title and soup.title.string:
                return postprocess_github_title(soup.title.string.strip())
            return ""
    except Exception as e:
        logging.error(f"Error in fetch_url_title({url}): {e}")
        return ""
# ------------- End Block -------------

def main():
    config = load_config('config.yaml')
    processed_base_path = os.path.join(config.get('base_path'), 'processed')
    data_type = 'modelcard'

    print("⚠️ Step 1: Loading data from parquet (modelcard_step1)...")
    df = pd.read_parquet(
        os.path.join(processed_base_path, f"{data_type}_step1.parquet"),
        columns=['modelId', 'card_tags', 'github_link', 'pdf_link']
    )

    print("⚠️ Step 2: Extracting links from columns (pdf_link, github_link)")
    all_links = extract_links_from_columns(df, ["pdf_link", "github_link"])
    print(f"Total unique links: {len(all_links)}")

    print("⚠️ Step 3: Classifying links (arxiv/biorxiv/medrxiv/github/other/invalid)...")
    records = []
    for lk in tqdm(all_links, desc="Classify links"):
        lk_lower = lk.lower()
        dm = extract_domain(lk)
        invalid = is_invalid_extension(lk)
        if invalid:
            cat, handler = "invalid", "none"
        else:
            if "arxiv" in lk_lower and "biorxiv" not in lk_lower:
                cat, handler = "arxiv", "arxiv_batch"
            elif ("biorxiv" in lk_lower) or ("medrxiv" in lk_lower):
                cat, handler = "biorxiv_or_medrxiv", "rxiv_api_or_html"
            elif "github.com" in lk_lower:
                cat, handler = "github", "github_html"
            else:
                cat, handler = "other", "html_parser"
        records.append(dict(
            link=lk, domain=dm, category=cat, handler=handler, invalid=invalid
        ))
    df_links = pd.DataFrame(records)
    df_links.to_csv("all_links_with_category.csv", index=False)
    print("✅ Wrote all_links_with_category.csv")

    valid_df = df_links[df_links["invalid"] == False].copy()
    print(f"Valid links: {len(valid_df)}")

    arxiv_df    = valid_df[valid_df["category"] == "arxiv"]
    rxiv_df     = valid_df[valid_df["category"] == "biorxiv_or_medrxiv"]
    github_df   = valid_df[valid_df["category"] == "github"]
    other_df    = valid_df[valid_df["category"] == "other"]

    print("⚠️ Step 4A: arXiv batch fetch")
    arxiv_ids = []
    for lk in arxiv_df["link"]:
        aid = extract_arxiv_id(lk)
        if aid:
            arxiv_ids.append(aid)
    arxiv_ids = list(set(arxiv_ids))
    print(f"Found {len(arxiv_ids)} unique arXiv IDs")
    arxiv_titles = {}
    if arxiv_ids:
        arxiv_titles = batch_fetch_arxiv_titles(arxiv_ids)
        for i, (a_id, a_tt) in enumerate(arxiv_titles.items()):
            print(f"arXiv ID={a_id}, title={a_tt}")
            if i > 3:
                break

    print("⚠️ Step 4B: biorxiv/medrxiv fetch")
    rxiv_title_map = {}
    for lk in tqdm(rxiv_df["link"], desc="Fetch biorxiv/medrxiv"):
        if "biorxiv" in lk.lower():
            t = fetch_biorxiv_title_via_api(lk)
            if not t:
                t = fetch_url_title(lk)
            rxiv_title_map[lk] = t
        else:
            t = fetch_medrxiv_title_via_api(lk)
            if not t:
                t = fetch_url_title(lk)
            rxiv_title_map[lk] = t

    print("⚠️ Step 4C: GitHub fetch => fallback to HTML (Sequential with delay)")
    github_links = github_df["link"].tolist()
    def sequential_fetch_github_titles(links, delay=2):
        result = {}
        for lk in tqdm(links, desc="Fetch GitHub Titles"):
            title = fetch_github_title_raw(lk)
            print(f"extracted: {lk} -> {title}")
            result[lk] = title
            time.sleep(delay)
        return result
    github_title_map = sequential_fetch_github_titles(github_links, delay=2)

    print("⚠️ Step 4D: other => HTML or PDF partial fetch (Parallel)")
    other_links = other_df["link"].tolist()
    def process_other_title(lk):
        title = fetch_url_title(lk)
        print(f"extracted: {lk} -> {title}")
        return (lk, title)
    other_results = Parallel(n_jobs=4)(
        delayed(process_other_title)(lk) for lk in tqdm(other_links, desc="Fetch other Titles")
    )
    other_title_map = {lk: title for lk, title in other_results}

    print("Samples from rxiv mapping:")
    for i, (lk, t) in enumerate(rxiv_title_map.items()):
        print(f"{lk} => {t}")
        if i > 2:
            break

    def get_extracted_title(row):
        lk = row["link"]
        cat = row["category"]
        if cat == "arxiv":
            a_id = extract_arxiv_id(lk)
            return arxiv_titles.get(a_id, "") if a_id else ""
        elif cat == "biorxiv_or_medrxiv":
            return rxiv_title_map.get(lk, "")
        elif cat == "github":
            return github_title_map.get(lk, "")
        elif cat == "other":
            return other_title_map.get(lk, "")
        else:
            return ""
    df_links["extracted_title"] = df_links.apply(get_extracted_title, axis=1)

    url_title_dict = pd.Series(df_links["extracted_title"].values, index=df_links["link"]).to_dict()

    def update_cell(cell):
        if isinstance(cell, (list, tuple, np.ndarray)):
            return {u: url_title_dict.get(u, "") for u in cell}
        elif isinstance(cell, str) and cell.strip():
            splitted = [s.strip() for s in cell.split(",")]
            return {u: url_title_dict.get(u, "") for u in splitted}
        else:
            return {}
    df["extracted_titles"] = df.apply(
        lambda row: {
            "pdf_link": update_cell(row["pdf_link"]),
            "github_link": update_cell(row["github_link"])
        },
        axis=1
    )

    out_path = os.path.join(processed_base_path, f"{data_type}_with_extracted_titles.parquet")
    df.to_parquet(out_path, index=False)
    print(f"✅ Done. Updated DataFrame with extracted titles saved to {out_path}")

if __name__ == "__main__":
    main()
