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
from joblib import Parallel, delayed ########
from src.utils import load_config
import html2text
import hashlib

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

PDF_DOWNLOAD_FOLDER = "pdf_downloads"
if not os.path.exists(PDF_DOWNLOAD_FOLDER):
    os.makedirs(PDF_DOWNLOAD_FOLDER)

# Old logic folder for GitHub HTML
GITHUB_HTML_FOLDER = "github_html"
if not os.path.exists(GITHUB_HTML_FOLDER):
    os.makedirs(GITHUB_HTML_FOLDER)

# Separate folder for this script's GitHub downloads
GITHUB_README_FOLDER_2 = "github_readme_output_2"  ########
if not os.path.exists(GITHUB_README_FOLDER_2):
    os.makedirs(GITHUB_README_FOLDER_2)  ########

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
    # Added debug print ########
    #print(f"[biorxiv/medrxiv debug] Trying biorxiv API for: {url}") ########
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
        print(f"[biorxiv/medrxiv debug] biorxiv API exception: {e}") ########
        return ""

def fetch_medrxiv_title_via_api(url):
    # Added debug print ########
    #print(f"[biorxiv/medrxiv debug] Trying medrxiv API for: {url}") ########
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
        print(f"[biorxiv/medrxiv debug] medrxiv API exception: {e}") ########
        return ""

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

def create_local_filename_2(base_output_dir, github_url):
    url_hash = hashlib.md5(github_url.encode('utf-8')).hexdigest()
    filename = f"{url_hash}.md"
    return os.path.join(base_output_dir, filename)

"""def extract_bibtex_from_html(html_text):
    m = re.search(r'@[\w]+\{[\s\S]+?\}\s', html_text)
    if m:
        return m.group(0).strip()
    return None"""

def extract_bibtex_from_html(html_text):
    # Robust BibTeX extraction: process line by line to capture complete BibTeX entries
    bibtex_entries = []
    bibtex_pattern = r"@(\w+)\{"
    current_entry = ""
    open_braces = 0
    inside_entry = False
    for line in html_text.splitlines():
        line = line.strip()
        if not line:
            continue
        if not inside_entry and re.match(bibtex_pattern, line):
            inside_entry = True
            current_entry = line
            open_braces = line.count("{") - line.count("}")
        elif inside_entry:
            current_entry += " " + line
            open_braces += line.count("{") - line.count("}")
        if inside_entry and open_braces == 0:
            bibtex_entries.append(current_entry.strip())
            inside_entry = False
            current_entry = ""
    if bibtex_entries:
        return bibtex_entries[0]  # Return the first valid BibTeX entry
    return None

def extract_title_from_readme(readme_content):
    if not readme_content:
        return None
    lines = readme_content.splitlines()
    for line in lines:
        if line.startswith("# "):
            return line[2:].strip()
    return None

def parse_html_title(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    if soup.title and soup.title.string:
        return soup.title.string.strip()
    return None

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

def process_github_url_debug(github_url, GITHUB_CONTENT_CACHE): ########
    """Process a single GitHub URL with debug prints."""
    cleaned_url = clean_github_link(github_url)
    if not cleaned_url.strip():
        print(f"[GitHub debug] Empty or invalid cleaned URL from: {github_url}") ########
        return (github_url, {"bibtex": None, "readme_title": None, "html_title": None})

    cached_content = GITHUB_CONTENT_CACHE.get(cleaned_url)
    local_path = create_local_filename_2(GITHUB_README_FOLDER_2, cleaned_url)
    if not cached_content and os.path.exists(local_path):
        with open(local_path, 'r', encoding='utf-8') as f:
            cached_content = f.read()
        GITHUB_CONTENT_CACHE[cleaned_url] = cached_content

    if not cached_content:
        outpath = download_github_readme_2(cleaned_url, local_path)
        if outpath and os.path.exists(outpath):
            with open(outpath, 'r', encoding='utf-8') as f:
                cached_content = f.read()
            GITHUB_CONTENT_CACHE[cleaned_url] = cached_content

    if not cached_content:
        print(f"[GitHub debug] No content found for: {github_url}") ########
        return (github_url, {"bibtex": None, "readme_title": None, "html_title": None})

    bibtex_block = extract_bibtex_from_html(cached_content)
    readme_title = extract_title_from_readme(cached_content)
    html_title = None
    if not readme_title and re.search(r'<(html|head|title)', cached_content, re.IGNORECASE):
        html_title = parse_html_title(cached_content)

    # Debug prints for extracted info ########
    print(f"[GitHub debug] URL={github_url}") ########
    print(f"[GitHub debug]  - Local MD path: {local_path if os.path.exists(local_path) else 'None'}") ########
    print(f"[GitHub debug]  - BibTeX: {bibtex_block if bibtex_block else 'N/A'}") ########
    print(f"[GitHub debug]  - README title: {readme_title if readme_title else 'N/A'}") ########
    #print(f"[GitHub debug]  - HTML title: {html_title if html_title else 'N/A'}") ########

    return (
        github_url,
        {
            "bibtex": bibtex_block,
            "readme_title": readme_title,
            "html_title": html_title
        }
    )

def parallel_fetch_github_info(links, GITHUB_CONTENT_CACHE, n_jobs=4): ########
    """Fetch GitHub info in parallel using joblib."""
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_github_url_debug)(lk, GITHUB_CONTENT_CACHE) for lk in tqdm(links, desc="Parallel GitHub Info") ########
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

def fetch_github_title_raw(url):
    EXCLUDED_TERMS = ['/issues', '/assets', '/sponsor', '/discussions', '/pull', '/tag', '/releases']
    EXCLUDED_SUFFIXES = ['LICENSE']
    for term in EXCLUDED_TERMS:
        if term in url:
            return "SkippedURL"
    for suffix in EXCLUDED_SUFFIXES:
        if url.endswith(suffix):
            return "SkippedURL"
    if url.endswith('.git'):
        url = url[:-4]
    if url.endswith(':'):
        url = url[:-1]
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
        except Exception as e:
            logging.error(f"Error fetching GitHub URL {url}: {e}")
            return "SkippedURL"
    bibtex = extract_bibtex_from_html(html_text)
    if bibtex:
        return bibtex
    md_text = html2text.html2text(html_text)
    m = re.search(r'^# (.+)$', md_text, re.MULTILINE)
    if m:
        return m.group(1).strip()
    soup = BeautifulSoup(html_text, "html.parser")
    if soup.title and soup.title.string:
        return postprocess_github_title(soup.title.string.strip())
    return ""

def process_other_title(lk):
    title = fetch_url_title(lk)
    print(f"extracted: {lk} -> {title}")
    return (lk, title)

def parallel_fetch_other_titles(links, n_jobs=4):
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

def load_github_cache(config):
    cache = {}
    mapping_path = os.path.join(config.get('base_path'), "processed", "github_readme_cache.parquet")
    url_to_hash = {}
    if os.path.exists(mapping_path):
        try:
            mapping_df = pd.read_parquet(mapping_path)
            url_to_hash = {
                str(k): str(v)
                for k, v in zip(mapping_df.get('raw_url', []), mapping_df.get('downloaded_path', []))
                if pd.notnull(k) and pd.notnull(v)
            }
            print(f"Loaded {len(url_to_hash)} valid URL mappings")
        except Exception as e:
            print(f"Mapping file load error: {e}")

    readme_folder = os.path.join(config.get('base_path'), "github_readme_output")
    if not os.path.exists(readme_folder):
        return cache

    for fname in os.listdir(readme_folder):
        if not fname.endswith('.md'):
            continue
        try:
            file_hash, _ = os.path.splitext(fname)
            matched_urls = []
            if url_to_hash:
                matched_urls = [
                    k for k, v in url_to_hash.items()
                    if isinstance(v, str) and v.endswith(fname)
                ]
            if not matched_urls:
                for possible_url in url_to_hash.keys():
                    if hashlib.md5(possible_url.encode()).hexdigest() == file_hash:
                        matched_urls.append(possible_url)
                        break
            with open(os.path.join(readme_folder, fname), 'r', encoding='utf-8') as f:
                content = f.read()
            if matched_urls:
                cache[matched_urls[0]] = content
            else:
                cache[file_hash] = content
        except Exception as e:
            print(f"Error processing {fname}: {str(e)[:50]}...")
    print(f"Total cached entries: {len(cache)}")
    return cache

def main():
    config = load_config('config.yaml')
    processed_base_path = os.path.join(config.get('base_path'), 'processed')
    data_type = 'modelcard'
    GITHUB_CONTENT_CACHE = load_github_cache(config)
    print(f"Loaded {len(GITHUB_CONTENT_CACHE)} GitHub cache entries.")

    print("Step 1: Loading data from parquet (modelcard_step1)...")
    df = pd.read_parquet(
        os.path.join(processed_base_path, f"{data_type}_step1.parquet"),
        columns=['modelId', 'card_tags', 'github_link', 'pdf_link']
    )

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
    df_links.to_csv("all_links_with_category.csv", index=False)
    print("Wrote all_links_with_category.csv")

    valid_df = df_links[df_links["invalid"] == False].copy()
    print(f"Valid links: {len(valid_df)}")

    arxiv_df    = valid_df[valid_df["category"] == "arxiv"]
    rxiv_df     = valid_df[valid_df["category"] == "biorxiv_or_medrxiv"]
    github_df   = valid_df[valid_df["category"] == "github"]
    other_df    = valid_df[valid_df["category"] == "other"]

    print("Step 4A: arXiv batch fetch")
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

    print("Step 4B: biorxiv/medrxiv fetch")
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
    
    print("Samples from biorxiv/medrxiv cache (first 5 entries):")
    for i, (lk, title) in enumerate(rxiv_title_map.items()):
        print(f"{lk} => {title}")
        if i >= 4:
            break

    print("Step 4C: GitHub fetch => updated logic with README extraction and BibTeX")
    github_links_raw = github_df["link"].tolist()
    unique_github_links = list({clean_github_link(x) for x in github_links_raw})
    # Parallel fetch for GitHub ########
    github_results = parallel_fetch_github_info(unique_github_links, GITHUB_CONTENT_CACHE, n_jobs=4) ########

    print("Step 4D: other => HTML or PDF partial fetch (Parallel)")
    other_links = other_df["link"].tolist()
    other_results = parallel_fetch_other_titles(other_links, n_jobs=4)
    other_title_map = {lk: title for lk, title in other_results.items()}

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
            # The dictionary is stored under the raw URL key
            return github_results.get(lk, {})
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
    print(f"Done. Updated DataFrame with extracted titles saved to {out_path}")

if __name__ == "__main__":
    main()

