# -*- coding: utf-8 -*-
"""
Author: Zhengyuan Dong
Created: 2025-02-24
Last Modified: 2025-03-16 (Updated GitHub URL handling, parallel GitHub processing, debug prints)
Description: Enhanced title extraction tool for pdf_links and github_links.
             1. Extract links from pdf_links and github_links columns, remove duplicates.
             2. For arxiv.org links, extract IDs, then batch query titles.
             3. For biorxiv.org / medrxiv.org links, try to extract ID and query via API, otherwise fallback to HTML.
             4. For GitHub links, use enhanced logic:
                - Clean the URL.
                - Check local cache by file hash.
                - If missing, download into a separate folder (avoid conflict with first script).
                - Extract (a) BibTeX block, (b) first Markdown heading, (c) HTML <title>.
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
import html2text
import hashlib
from src.data_ingestion.readme_parser import BibTeXExtractor
from src.data_ingestion.bibtex_parser import BibTeXFactory
from src.data_preprocess.step1 import process_bibtex_tuple, parse_bibtex_entries
from joblib import parallel_backend 
from PyPDF2 import PdfReader
import pyarrow as pa
import pyarrow.parquet as pq
import ast

class PdfReadError(Exception):
    pass

ARXIV_CACHE_PATH = "data/processed/arxiv_titles_cache.json"
RXIV_CACHE_PATH = "data/processed/rxiv_titles_cache.json"


def extract_titles(bibtex_list):
    if not isinstance(bibtex_list, (list, tuple, np.ndarray)):
        return []
    return [
        d.get("title", "")
         .replace("{", "")
         .replace("}", "")
         .lower()
         .strip()
        for d in bibtex_list
        if isinstance(d, dict) and d.get("title")
    ]

def load_cache(file_path):
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def save_cache(data, file_path, mode="overwrite"):
    if mode == "overwrite":
        with open(file_path, 'w') as f:
            json.dump(data, f)
    elif mode == "update":
        existing = load_cache(file_path)
        existing.update(data)
        with open(file_path, 'w') as f:
            json.dump(existing, f)

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

PDF_DOWNLOAD_FOLDER = "pdf_downloads"
if not os.path.exists(PDF_DOWNLOAD_FOLDER):
    os.makedirs(PDF_DOWNLOAD_FOLDER)

GITHUB_README_FOLDER = "data/downloaded_github_readmes_processed"
assert os.path.exists(GITHUB_README_FOLDER)

# Separate folder for this script's GitHub downloads
GITHUB_README_FOLDER_2 = "github_readme_output_2"  
if not os.path.exists(GITHUB_README_FOLDER_2):
    os.makedirs(GITHUB_README_FOLDER_2)  

SKIP_EXTENSIONS = {
    'png', 'jpg', 'jpeg', 'gif', 'zip', 'rar', '7z', 'doc', 'docx', 'xls', 'xlsx',
    'ppt', 'pptx', 'mp4', 'mp3', 'md', 'pth', 'pt', 'ckpt', 'safetensors',
    'json', 'py', 'ipynb', 'csv', 'tsv', "bin", "h5", "onnx", "pkl", "tar", "gz", "mov"
}

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
        print(f"Error extracting domain from {url}: {e}")
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
    with parallel_backend('loky', n_jobs=4, temp_folder="./joblib_tmp"): 
        results = Parallel(n_jobs=4)(
            delayed(lambda col: df[col].apply(process_cell).tolist())(col)
            for col in tqdm(cols, desc="Processing columns")
        )
    combined_links = [link for col in results for cell in col for link in cell]
    return list(set(combined_links))

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
                if not isinstance(title, str): 
                    title = str(title) 
                return title.strip()
            else:
                return pdf_title_full_fetch(url)
        except (PdfReadError, Exception):
            return pdf_title_full_fetch(url)
    except Exception:
        return pdf_title_full_fetch(url)

def pdf_title_full_fetch(url):
    try:
        file_basename = os.path.basename(url)
        local_pdf_path = os.path.join(PDF_DOWNLOAD_FOLDER, file_basename)
        if os.path.isfile(local_pdf_path) and os.path.getsize(local_pdf_path) > 0:
            with open(local_pdf_path, 'rb') as f: 
                pdf_bytes = f.read() 
        else:
            resp = requests.get(url, timeout=TIMEOUT)
            resp.raise_for_status()
            with open(local_pdf_path, 'wb') as f:
                f.write(resp.content)
            pdf_bytes = resp.content
        pdf_reader = PdfReader(io.BytesIO(pdf_bytes))
        metadata = pdf_reader.metadata
        title = metadata.get('/Title') if metadata else None
        if title:
            if isinstance(title, str):
                return title.strip()
            else:
                return str(title).strip()
        else:
            return None
        #return title.strip() if title else None
    except Exception as e:
        print(f"Failed to fetch or parse PDF {url}: {e}")
        return None

def extract_arxiv_id(url):
    if "biorxiv" in url.lower():
        return None
    if "arxiv" not in url.lower():
        return None
    m = re.search(r'(\d{4}\.\d{5})', url)
    if m:
        return m.group(1)
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
    # Added debug print 
    #print(f"[biorxiv/medrxiv debug] Trying biorxiv API for: {url}") 
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
    except Exception as e:
        print(f"[biorxiv/medrxiv debug] biorxiv API exception: {e}") 
        return ""

def fetch_medrxiv_title_via_api(url):
    # Added debug print 
    #print(f"[biorxiv/medrxiv debug] Trying medrxiv API for: {url}") 
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
    except Exception as e:
        print(f"[biorxiv/medrxiv debug] medrxiv API exception: {e}") 
        return ""

def batch_fetch_arxiv_titles(arxiv_ids, chunk_size=20, delay_between_chunks=1):
    results = {}
    total = len(arxiv_ids)
    if total == 0:
        print("[DEBUG] No arXiv IDs provided, returning empty results")  
        return results  
    print(f"[DEBUG] Batch fetching {total} arXiv IDs with chunk_size={chunk_size}")  
    for i in range(0, total, chunk_size):
        chunk_ids = arxiv_ids[i:i+chunk_size]
        id_list_str = ",".join(chunk_ids)
        url = f'http://export.arxiv.org/api/query?id_list={id_list_str}&max_results={chunk_size}'
        print(f"[DEBUG] Processing chunk starting at index {i}")  
        try:
            resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
            resp.raise_for_status()
            root = ET.fromstring(resp.text)
            for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
                title_element = entry.find('{http://www.w3.org/2005/Atom}title')
                id_element = entry.find('{http://www.w3.org/2005/Atom}id')
                if title_element is not None and id_element is not None:
                    full_id = id_element
                    title = title_element.text.strip().replace('\n', '').replace('  ', ' ')
                    raw_id = full_id.text.strip().split('/')[-1].rstrip('v').split('v')[0]
                    results[raw_id] = title
                    print(f"[DEBUG] Extracted ID: {raw_id} with Title: {title}")  
                else:
                    #pass
                    print("[DEBUG] Missing title or id element in an entry")  
        except Exception as e:
            print(f"[ERROR] Failed arXiv batch fetch chunk at index {i}: {e}")  
        if i + chunk_size < total:
            print(f"[DEBUG] Sleeping for {delay_between_chunks} seconds before processing next chunk")  
            time.sleep(delay_between_chunks)
    print("[DEBUG] Finished processing all chunks")  
    return results

def update_downloaded_path(df):
    df["downloaded_path"] = df["downloaded_path"].apply(
        lambda x: x.replace("downloaded_github_readmes", "downloaded_github_readmes_processed") if isinstance(x, str) else x
    )
    return df

# GitHub utilities from first script (in English comments now)
def clean_github_link(github_link):
    return github_link.split('{')[0].split('}')[0].split('[')[0].split(']')[0] \
                      .split('(')[0].split(')')[0].split('<')[0].split('>')[0] \
                      .split('*')[0].split('`')[0].split('"')[0].split("'")[0] \
                      .split('!')[0]

def is_text_file(url):
    try:
        response = requests.head(url, allow_redirects=True, timeout=5)
        content_type = response.headers.get("Content-Type", "")
        mime_type = content_type.split(";")[0].strip().lower()
        return mime_type.startswith("text/")
    except:
        return False

def extract_title_from_readme(content): # important func!
    """
    Extract title from a markdown file using the following rules:
    1. Find the marker "repository files navigation" (case-insensitive).
    2. Starting from the line after the marker, skip blank lines and lines that start with "["
       until encountering the first line that starts with "#" (a header).
    3. Collect that header line (after stripping the leading "#" and spaces) and subsequent 
       non-blank lines until an empty line is encountered.
    4. Join the collected lines with a space and return as the title.
    5. If no such marker or header block is found, fallback to the first line starting with "# ".
    """
    lines = content.splitlines()
    marker = "repository files navigation"
    start_index = None
    # Find the marker (ignore case)
    for i, line in enumerate(lines):
        if marker in line.lower():
            start_index = i + 1
            break
    if start_index is not None:
        header_index = None
        # Skip blank lines and lines starting with "[" until a header is found
        for j in range(start_index, len(lines)):
            stripped = lines[j].strip()
            if stripped == "" or stripped.startswith("["):
                continue
            if stripped.startswith("#"):
                header_index = j
                break
        if header_index is not None:
            title_lines = []
            # Collect contiguous non-blank lines starting from header_index
            for k in range(header_index, len(lines)):
                stripped = lines[k].strip()
                if stripped == "":
                    break
                # Remove leading "#" and extra spaces
                title_lines.append(stripped.lstrip("#").strip())
            if title_lines:
                return "".join(title_lines)
    # Fallback: return the first header line starting with "# "
    for line in lines:
        if line.startswith("# "):
            return line[2:].strip()
    return None

def parse_html_title(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    if soup.title and soup.title.string:
        return soup.title.string.strip()
    return None

def process_html_title(title):
    """
    Process an HTML title string.
    If it matches a pattern like "GitHub - username/reponame: actual title", 
    remove the prefix ("GitHub - username/reponame:") and return the rest.
    Otherwise, if the title contains a colon, remove everything before the first colon.
    If no colon is present, return the title unchanged.
    """
    if not title:
        return title
    if title.startswith('GitHub'):
        m = re.match(r"GitHub\s*-\s*[^:]+:\s*(.*)", title)
        if m:
            return m.group(1).strip()
        if ":" in title:
            return title.split(":", 1)[1].strip()
    return title

def download_github_readme_2(github_url, output_path):
    EXCLUDED_TERMS = ['/issues', '/assets', '/sponsor', '/discussions', '/pull', '/tag', '/releases']
    EXCLUDED_SUFFIXES = ['LICENSE']
    skip_extensions = {"json","pth","bin","ckpt","pt","h5","onnx","py","pkl","tar","gz",
                       "gif","png","jpg","jpeg","csv","ipynb","mp4","mov"}

    if any(term in github_url for term in EXCLUDED_TERMS) or any(github_url.endswith(suffix) for suffix in EXCLUDED_SUFFIXES):
        return None
    raw_url = github_url
    if raw_url.endswith('.git'):
        raw_url = raw_url[:-4]
    if raw_url.endswith(':'):
        raw_url = raw_url[:-1]

    try:
        last_segment = raw_url.split('/')[-1]
        last_segment = re.split(r'[#\?]', last_segment)[0]
        if '.' in last_segment:
            ext = last_segment.rsplit('.', 1)[-1].lower()
            if ext in skip_extensions:
                return None
            accept_extensions = {"txt", "rst", "md", "markdown"}
            if ext not in accept_extensions:
                if not is_text_file(raw_url):
                    return None

        if 'gist.github.com' not in raw_url and 'github.com' in raw_url:
            raw_url = raw_url.replace('github.com', 'raw.githubusercontent.com') \
                             .replace('blob/', '').replace('tree/', '') \
                             .rstrip("/")
        if raw_url.endswith(('.md','.rst','.txt','.markdown')) or 'gist.github.com' in raw_url:
            response = requests.get(raw_url, timeout=10)
            if response.status_code == 200:
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(response.text)
                return output_path
            else:
                return None

        readme_variants = [
            'README.md','README.rst','README.txt','README.markdown','README',
            'Readme.md','readme.md','Readme.rst','readme.rst','Readme.txt','readme.txt',
            'Readme.markdown','readme.markdown','Readme','readme',
        ]
        readme_urls = [github_url]
        for variant in readme_variants:
            readme_urls.append(f"{raw_url}/{variant}")
            readme_urls.append(f"{raw_url}/master/{variant}")
            readme_urls.append(f"{raw_url}/main/{variant}")

        for rurl in readme_urls:
            try:
                resp = requests.get(rurl, timeout=10)
                if resp.status_code == 200:
                    with open(output_path, "w", encoding="utf-8") as f:
                        f.write(resp.text)
                    return output_path
            except:
                pass

        html_response = requests.get(github_url, timeout=10)
        if html_response.status_code == 200:
            md_text = html2text.html2text(html_response.text)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(md_text)
            return output_path
        return None
    except:
        return None

readme_cache = {}

def make_upper(content): 
    return content.upper() if content is not None else "" 

def process_github_url_debug(github_url, GITHUB_PATH_CACHE): 
    """Process a single GitHub URL with debug prints."""
    cleaned_url = clean_github_link(github_url)
    if not cleaned_url.strip():
        print(f"[GitHub debug] Empty or invalid cleaned URL from: {github_url}") 
        return (github_url, {"bibtex": None, "readme_title": None, "html_title": None})
    url_hash = hashlib.md5(cleaned_url.encode('utf-8')).hexdigest()
    url_name = f"{url_hash}.md"
    local_path = None
    local_path_cached = GITHUB_PATH_CACHE.get(github_url)
    local_path_2 = os.path.join(GITHUB_README_FOLDER_2, url_name)
    if local_path_cached and os.path.exists(local_path_cached):
        with open(local_path_cached, 'r', encoding='utf-8') as f:
            local_content = f.read()
        local_path = local_path_cached
    elif os.path.exists(local_path_2):
        with open(local_path_2, 'r', encoding='utf-8') as f:
            local_content = f.read()
        local_path = local_path_2
    else:
        local_content = None
        local_path = None
    #content_upper = cached_content.upper() if cached_content else ""
    #if "<!DOCTYPE" in make_upper(local_content):
    #    #local_content = html2text.html2text(local_content)
    #    #with open(local_path, 'w', encoding='utf-8') as f:
    #    #    f.write(cached_content)
    if not local_content:
        outpath = download_github_readme_2(cleaned_url, local_path_2)
        if outpath and os.path.exists(outpath):
            with open(outpath, 'r', encoding='utf-8') as f:
                local_content = f.read()
            if "<!DOCTYPE" in make_upper(local_content):
                # print(f"[GitHub debug] Detected raw HTML in cached content for: {github_url}. Reprocessing with html2text.")
                local_content = html2text.html2text(local_content)
                with open(outpath, 'w', encoding='utf-8') as f:
                    f.write(local_content)
            GITHUB_PATH_CACHE[cleaned_url] = outpath
            local_path = outpath
        else:
            print(f"[GitHub debug] No content found for: {github_url}") 
            local_path = None
            return (github_url, {"bibtex": None, "readme_title": None, "html_title": None})
    bibtex_block = BibTeXExtractor().extract(local_content)
    html_title = None
    readme_title = None
    if "<!DOCTYPE" not in make_upper(local_content):
        readme_title = extract_title_from_readme(local_content)
    else:
        #if re.search(r'<(html|head|title)', cached_content, re.IGNORECASE): # not readme_title and 
        try:
            html_title = parse_html_title(local_content)
            html_title = process_html_title(html_title)
        except Exception as e:
            print('Error in process_github_url_debug: ', e)
    # Debug prints for extracted info 
    print(f"[GitHub debug] URL={github_url}") 
    print(f"[GitHub debug]  - Local MD path: {local_path if os.path.exists(local_path) else 'None'}") 
    print(f"[GitHub debug]  - BibTeX: {bibtex_block if bibtex_block else 'N/A'}") 
    print(f"[GitHub debug]  - README title: {readme_title if readme_title else 'N/A'}") 
    print(f"[GitHub debug]  - HTML title: {html_title if html_title else 'N/A'}") 
    return (
        github_url,
        {
            "bibtex": bibtex_block,
            "readme_title": readme_title,
            "html_title": html_title
        }
    )

def parallel_fetch_github_info(links, GITHUB_PATH_CACHE, n_jobs=4): 
    """Fetch GitHub info in parallel using joblib."""
    with parallel_backend('loky', n_jobs=n_jobs, temp_folder="./joblib_tmp"): 
        results = Parallel(n_jobs=n_jobs)(
            delayed(process_github_url_debug)(lk, GITHUB_PATH_CACHE) for lk in tqdm(links, desc="Parallel GitHub Info")
        )
    return dict(results)

def postprocess_github_title(title):
    if not title:
        return title
    title = re.sub(r'^GitHub\s*-\s*[^:]+:\s*', '', title)
    title = re.sub(r'^GitHub\s*-\s*[^:]+', '', title)
    return title.strip()

def url_to_filename(url):
    h = hashlib.sha256(url.encode('utf-8')).hexdigest()
    return h + ".html"

def process_other_title(lk):
    try:
        title = fetch_url_title(lk)
    except Exception as e:
        print(f"Error processing {lk}: {e}")
        title = ""
    print(f"extracted: {lk} -> {title}")
    return (lk, title)

def parallel_fetch_other_titles(links, n_jobs=4):
    with parallel_backend('loky', n_jobs=n_jobs, temp_folder="./joblib_tmp"): 
        results = Parallel(n_jobs=n_jobs)(
            delayed(process_other_title)(lk) for lk in tqdm(links, desc="Fetch other Titles")
        )
    return {lk: title for lk, title in results}

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
            print(f"Fetching PDF title: {url}, title={pdf_title_partial_fetch(url)}")
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
        print(f"Error in fetch_url_title({url}): {e}")
        return ""

def load_github_cache(config):
    mapping_path = os.path.join(config.get('base_path'), "processed", "github_readme_cache.parquet")
    updated_mapping_path = os.path.join(config.get('base_path'), "processed", "github_readme_cache_update.parquet")
    if os.path.exists(updated_mapping_path):
        mapping_path = updated_mapping_path
    else:
        pass
    assert os.path.exists(mapping_path)
    mapping_df = pd.read_parquet(mapping_path)
    mapping_df = update_downloaded_path(mapping_df) # fix path for new folder
    url_to_hash = {
        str(k): str(v)
        for k, v in zip(mapping_df.get('raw_url', []), mapping_df.get('downloaded_path', []))
        if pd.notnull(k) and pd.notnull(v)
    }
    print(f"Loaded {len(url_to_hash)} valid URL mappings")
    return url_to_hash

def main():
    config = load_config('config.yaml')
    processed_base_path = os.path.join(config.get('base_path'), 'processed')
    data_type = 'modelcard'
    GITHUB_PATH_CACHE = load_github_cache(config)
    print(f"Loaded {len(GITHUB_PATH_CACHE)} GitHub cache entries.")

    print("Step 1: Loading data from parquet (modelcard_step1)...")
    df = pd.read_parquet(os.path.join(processed_base_path, f"{data_type}_step1.parquet"), columns=['modelId', 'github_link', 'pdf_link'])

    print("Step 2: Extracting links from columns (pdf_link, github_link)")
    all_links = extract_links_from_columns(df, ["pdf_link", "github_link"])
    print(f"Total unique links: {len(all_links)}")

    print("Step 3: Classifying links (arxiv/biorxiv/medrxiv/github/other/invalid)...")
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
                cat, handler = "github", "github_updated"
            else:
                cat, handler = "other", "html_parser"
        records.append(dict(
            link=lk, domain=dm, category=cat, handler=handler, invalid=invalid
        ))
    df_links = pd.DataFrame(records)
    df_links.to_csv("data/processed/all_links_with_category.csv", index=False)
    print("Wrote data/processed/all_links_with_category.csv")

    valid_df = df_links[df_links["invalid"] == False].copy()
    print(f"Valid links: {len(valid_df)}")

    arxiv_df    = valid_df[valid_df["category"] == "arxiv"]
    rxiv_df     = valid_df[valid_df["category"] == "biorxiv_or_medrxiv"]
    github_df   = valid_df[valid_df["category"] == "github"]
    other_df    = valid_df[valid_df["category"] == "other"]

    print("Step 4A: arXiv batch fetch")
    # extract arxiv ids
    arxiv_cache = load_cache(ARXIV_CACHE_PATH)
    new_arxiv_links = [
        lk for lk in arxiv_df["link"] 
        if (lk not in arxiv_cache) or 
        (arxiv_cache.get(lk) is None) or 
        (isinstance(arxiv_cache.get(lk), str) and arxiv_cache.get(lk).strip().lower() == "just a moment...") or 
        (isinstance(arxiv_cache.get(lk), str) and arxiv_cache.get(lk).strip() == "")
    ]
    if new_arxiv_links:
        print('===============new_arxiv_links:', len(new_arxiv_links))
        new_arxiv_ids = list(set(extract_arxiv_id(lk) for lk in new_arxiv_links if extract_arxiv_id(lk)))
        new_titles = batch_fetch_arxiv_titles(new_arxiv_ids)
        non_empty_count = sum(1 for v in new_titles.values() if v not in [None, "", [], {}, ()])
        print('get non empty arxiv id batch query:', non_empty_count)
        # save new_titles
        #save_cache(new_titles, "arxivid_cache.json")
        new_cache = {lk: new_titles.get(extract_arxiv_id(lk), "") for lk in new_arxiv_links}
        for link, the_title in new_cache.items():
            print(f"[DEBUG] arxiv link => {link}, cached title => {the_title}")  
        save_cache(new_cache, ARXIV_CACHE_PATH, mode="update")
        arxiv_cache.update(new_cache)
    arxiv_titles = arxiv_cache

    print("Step 4B: biorxiv/medrxiv fetch")
    rxiv_cache = load_cache(RXIV_CACHE_PATH)
    new_rxiv_links = [
        lk for lk in rxiv_df["link"]
        if (lk not in rxiv_cache) or 
        (rxiv_cache.get(lk) is None) or 
        (isinstance(rxiv_cache.get(lk), str) and rxiv_cache.get(lk).strip().lower() == "just a moment...") or 
        (isinstance(rxiv_cache.get(lk), str) and rxiv_cache.get(lk).strip() == "")
    ]
    if new_rxiv_links:
        new_rxiv_titles = {}
        for lk in new_rxiv_links:
            if "biorxiv" in lk.lower():
                title = fetch_biorxiv_title_via_api(lk)
                if not title:
                    title = fetch_url_title(lk)
                new_rxiv_titles[lk] = title
            else:
                title = fetch_medrxiv_title_via_api(lk)
                if not title:
                    title = fetch_url_title(lk)
                new_rxiv_titles[lk] = title
        save_cache(new_rxiv_titles, RXIV_CACHE_PATH, mode="update")
        rxiv_cache.update(new_rxiv_titles)
    rxiv_title_map = rxiv_cache

    print("Step 4C: GitHub fetch => updated logic with README extraction and BibTeX")
    github_links_raw = github_df["link"].tolist()
    unique_github_links = list({clean_github_link(x) for x in github_links_raw})

    GITHUB_EXTRA_CACHE_PATH = os.path.join(config.get('base_path'), "processed", "github_extraction_cache.json")  #
    github_extraction_cache = load_cache(GITHUB_EXTRA_CACHE_PATH)  #
    urls_to_process = []
    for url in unique_github_links:
        if url not in github_extraction_cache:
            urls_to_process.append(url)
    print(f"GitHub extraction cache has {len(github_extraction_cache)} entries; processing {len(urls_to_process)} new URLs")
    
    new_github_results = {}
    if urls_to_process:
        new_github_results = parallel_fetch_github_info(urls_to_process, GITHUB_PATH_CACHE, n_jobs=4)  #
        github_extraction_cache.update(new_github_results)  #
        with open(GITHUB_EXTRA_CACHE_PATH, 'w', encoding='utf-8') as f:  #
            json.dump(github_extraction_cache, f)  #

    # Parallel fetch for GitHub 
    #github_results = parallel_fetch_github_info(unique_github_links, GITHUB_PATH_CACHE, n_jobs=4) 
    github_results = {url: github_extraction_cache.get(url) for url in unique_github_links}  #

    new_mapping_df = pd.DataFrame({
        'raw_url': list(GITHUB_PATH_CACHE.keys()),
        'downloaded_path': list(GITHUB_PATH_CACHE.values())
    })
    new_mapping_path = os.path.join(config.get('base_path'), "processed", "github_readme_cache_update.parquet")
    new_mapping_df.to_parquet(new_mapping_path, compression="zstd", engine="pyarrow", index=False)
    print(f"Saved updated GitHub cache to {new_mapping_path}")

    print("Step 4D: other => HTML or PDF partial fetch (Parallel)")
    other_title_map = {}
    """other_links = other_df["link"].tolist()
    other_results = parallel_fetch_other_titles(other_links, n_jobs=4)
    other_title_map = {lk: title for lk, title in other_results.items()}"""

    link2title = {}
    for lk in arxiv_df["link"]:
        link2title[lk] = arxiv_cache.get(lk, "")
    for lk in rxiv_df["link"]:
        link2title[lk] = rxiv_cache.get(lk, "")
    #for lk in github_df["link"]:
    #    link2title[lk] = github_results.get(lk, {"bibex": [], "readme_title": None, "html_title": None})
    for lk in other_df["link"]:
        link2title[lk] = other_title_map.get(lk, "")
    for lk in github_df["link"]:
        c_lk = clean_github_link(lk)
        link2title[c_lk] = github_results.get(c_lk, {"bibtex": [], "readme_title": None, "html_title": None})
    # save link2title
    #save_cache(link2title, "link2title_cache.json")
    #print("!Saved link2title_cache.json")

    def convert_title(val):
        if isinstance(val, dict):
            return val.get("readme_title") or val.get("html_title") or ""
        return val

    def map_links_to_new_title_columns(row):
        links = []
        pdf_link_val = row.get("pdf_link")
        if isinstance(pdf_link_val, str) and pdf_link_val.strip():
            links.extend([x.strip() for x in pdf_link_val.split(",") if x.strip()])
        elif isinstance(pdf_link_val, (list, tuple, np.ndarray)) and len(pdf_link_val) > 0:
            links.extend(pdf_link_val)
            
        github_link_val = row.get("github_link")
        if isinstance(github_link_val, str) and github_link_val.strip():
            links.extend([x.strip() for x in github_link_val.split(",") if x.strip()])
        elif isinstance(github_link_val, (list, tuple, np.ndarray)) and len(github_link_val) > 0:
            links.extend(github_link_val)
            
        title_arxiv = []
        title_rxiv = []
        title_github_readme = []
        title_github_html = []
        title_github_bibtex = []
        title_pdf = []
        title_other = []
        
        for link in links:
            link_lower = link.lower()
            if is_invalid_extension(link):
                continue
            if "arxiv" in link_lower and "biorxiv" not in link_lower:
                title = arxiv_cache.get(link, "")
                if title and title.strip():
                    title_arxiv.append(title.strip())
            elif "biorxiv" in link_lower or "medrxiv" in link_lower:
                title = rxiv_cache.get(link, "")
                if title and title.strip():
                    title_rxiv.append(title.strip())
            elif "github.com" in link_lower:
                cleaned = clean_github_link(link)
                github_info = github_results.get(cleaned, {"bibtex": None, "readme_title": None, "html_title": None})
                if github_info.get("readme_title") and str(github_info.get("readme_title")).strip():
                    title_github_readme.append(str(github_info.get("readme_title")).strip())
                if github_info.get("html_title") and str(github_info.get("html_title")).strip():
                    title_github_html.append(str(github_info.get("html_title")).strip())
                if github_info.get("bibtex") and str(github_info.get("bibtex")).strip():
                    bibtex_str = str(github_info.get("bibtex")).strip()
                    if bibtex_str.startswith('[') and bibtex_str.endswith(']'):
                        try:
                            bibtex_list = ast.literal_eval(bibtex_str)
                        except Exception as e:
                            print("Error evaluating bibtex string:", e)
                            bibtex_list = [bibtex_str]
                    else:
                        bibtex_list = [bibtex_str]
                    title_github_bibtex.extend(bibtex_list)
            elif link_lower.endswith(".pdf"):
                pass
            else:
                title = other_title_map.get(link, "")
                if title and title.strip():
                    title_other.append(title.strip())
        
        return pd.Series({
            "title_arxiv": title_arxiv if title_arxiv else None,
            "title_rxiv": title_rxiv if title_rxiv else None,
            "title_github_readme": title_github_readme if title_github_readme else None,
            "title_github_html": title_github_html if title_github_html else None,
            "title_github_bibtex": title_github_bibtex if title_github_bibtex else None,
            "title_pdf": title_pdf if title_pdf else None,
            "title_other": title_other if title_other else None,
        })
    
    def merge_bibtex_titles(row):
        list1 = row["parsed_bibtex_tuple_list"]
        list2 = row["parsed_bibtex_tuple_list_github"]
        titles_1 = extract_titles(list1)
        titles_2 = extract_titles(list2)
        merged = list(set(list(titles_1) + list(titles_2)))
        return merged
    
    def merge_all_titles(row):
        bibtex_titles = row['all_bibtex_titles']
        additional_titles = []
        for col, val in row.items():
            if isinstance(col, str) and col.startswith("title_"):
                if isinstance(val, str):
                    if val.strip():
                        additional_titles.append(
                            val.replace("{", "").replace("}", "").lower().strip()
                        )
                elif isinstance(val, (list, np.ndarray, tuple)):
                    for item in val:
                        if isinstance(item, str) and item.strip():
                            additional_titles.append(
                                item.replace("{", "").replace("}", "").lower().strip()
                            )
        all_titles = list(set(bibtex_titles + additional_titles))
        return all_titles

    df_new_titles = df.apply(map_links_to_new_title_columns, axis=1)
    df = pd.concat([df, df_new_titles], axis=1)
    #df["github_bibtex_tuple"] = df["github_titles"].apply(extract_bibtex_from_github_titles)
    parse_bibtex_entries(df, key="title_github_bibtex", output_key="parsed_bibtex_tuple_list_github", count_key = "successful_parse_count_github")
    print("\n-- Final df --")
    df_step1 = pd.read_parquet(os.path.join(processed_base_path, f"{data_type}_step1.parquet"), columns=['modelId', 'parsed_bibtex_tuple_list'])
    df_final = pd.merge(df_step1, df, on="modelId", how="left")
    df_final["all_bibtex_titles"] = df_final.apply(merge_bibtex_titles, axis=1)
    df_final["all_title_list"] = df_final.apply(merge_all_titles, axis=1)
    #df_final.to_parquet(os.path.join(processed_base_path, f"{data_type}_all_title_list.parquet"), compression='zstd', engine='pyarrow')
    df_final.drop(columns=['card_tags', 'downloads'], inplace=True)
    pq.write_table(pa.Table.from_pandas(df_final), os.path.join(processed_base_path, f"{data_type}_all_title_list.parquet"))
    print("✅ Merged BibTeX columns into 'all_bibtex_titles'")


if __name__ == "__main__":
    main()
