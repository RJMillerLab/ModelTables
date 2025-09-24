"""
Author: Zhengyuan Dong
Created: 2025-02-24
Last Modified: 2025-03-08
Description: Download README & HTML as .md files from the GitHub URLs.

TODO: solve the memory leakage issue
"""
import os, re, time
import pandas as pd
import numpy as np
import requests
from tqdm import tqdm
from src.utils import load_config, to_parquet
import urllib.parse
import html2text
import hashlib
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib
import aiohttp
import asyncio

cache = {}

def clean_github_link(github_link):
    return github_link.split('{')[0].split('}')[0].split('[')[0].split(']')[0].split('(')[0].split(')')[0].split('<')[0].split('>')[0].split('*')[0].split('`')[0].split('"')[0].split("'")[0].split('!')[0]

def create_local_filename(base_output_dir, github_url):
    url_hash = hashlib.md5(github_url.encode('utf-8')).hexdigest()
    filename = f"{url_hash}.md"
    return os.path.join(base_output_dir, filename)

def parse_github_link(github_link):
    if not isinstance(github_link, str):
        return None, None
    parsed_url = urllib.parse.urlparse(github_link)
    if not parsed_url.scheme or not parsed_url.netloc:
        print(f"Invalid GitHub link: {github_link}")
        return None, None
    path_parts = parsed_url.path.strip('/').split('/')
    if len(path_parts) >= 2:
        user = path_parts[0]
        repo = path_parts[1]
        return user, repo
    return None, None

def is_text_file(url):
    try:
        response = requests.head(url, allow_redirects=True, timeout=5)
        content_type = response.headers.get("Content-Type", "")
        mime_type = content_type.split(";")[0].strip().lower()
        return mime_type.startswith("text/")
    except:
        return False

def main_download(df, base_output_dir, to_path="data/github_readmes_info.parquet"):
    assert 'github_link' in df.columns, "Missing 'github_link' column in DataFrame."
    assert 'modelId' in df.columns, "Missing 'modelId' column in DataFrame."

    # step1: get all links
    start_time = time.time()
    #df['github_link'] = df['github_link'].apply(
    #    lambda x: eval(x) if isinstance(x, str) and x.startswith('[') and x.endswith(']') else x
    #)
    all_raw_links = df[['modelId', 'github_link']].explode('github_link').dropna()
    total_links_before_dedup = len(all_raw_links)
    all_links = set(clean_github_link(str(link).strip()) for link in all_raw_links['github_link'] if link)

    print(f"Found {total_links_before_dedup} GitHub links before deduplication.")
    print(f"Found {len(all_links)} unique GitHub URLs after deduplication.")
    print(f"Speedup ratio: {total_links_before_dedup / len(all_links):.2f}x reduction in requests.")
    print(f"Step1 time cost: {time.time() - start_time:.2f} seconds.")

    # step2: download all
    start_time = time.time()
    bulk_download_github_urls(all_links, base_output_dir)
    print(f"Step2 time cost: {time.time() - start_time:.2f} seconds.")
    
    # step3: link downloaded files back ot the model data
    start_time = time.time()
    download_info = []
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Assembling Results"):
        model_id = row['modelId']
        raw_links = row['github_link']
        if raw_links is None:
            download_info.append({
                "modelId": model_id,
                "github_link": [],
                "readme_path": []
            })
            continue

        if isinstance(raw_links, str):
            raw_links = [raw_links]
        elif isinstance(raw_links, (list, tuple, np.ndarray)):
            raw_links = list(raw_links)

        readme_paths = []
        for g_link in raw_links:
            if not g_link:
                continue
            cleaned_link = clean_github_link(g_link.strip())
            local_path = cache.get(cleaned_link)
            if local_path:
                readme_paths.append(local_path)
        download_info.append({
            "modelId": model_id,
            "github_link": raw_links,
            "readme_path": readme_paths
        })
    download_info_df = pd.DataFrame(download_info)
    to_parquet(download_info_df, to_path)
    # save the cache as parquet
    cache_df = pd.DataFrame(list(cache.items()), columns=['raw_url', 'downloaded_path'])
    to_parquet(cache_df, os.path.join(config.get('base_path'), "processed", "github_readme_cache.parquet"))
    skipped_df = pd.DataFrame(skipped_links, columns=['raw_url', 'reason'])
    to_parquet(skipped_df, os.path.join(config.get('base_path'), "processed", "github_skipped_urls.parquet"))
    print(f"Downloaded {len([d for d in download_info if d['readme_path']])} READMEs.")
    print(f"Skipped {len([d for d in download_info if not d['readme_path']])} READMEs.")
    print(f"Step3 time cost: {time.time() - start_time:.2f} seconds.")
    return download_info_df

async def fetch_readme(session, url, local_filename):
    if os.path.exists(local_filename):
        cache[url] = local_filename
        return url, local_filename
    try:
        async with session.get(url, timeout=10) as response:
            if response.status == 200:
                content = await response.text()
                with open(local_filename, "w", encoding="utf-8") as f:
                    f.write(content)
                cache[url] = local_filename
                return url, local_filename
            else:
                print(f"Error: {url} - Status {response.status}")
    except Exception as e:
        print(f"Error fetching {url}: {e}")
    cache[url] = None
    return url, None

async def async_download_github_urls(all_links, base_output_dir, num_workers=20):
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=50)) as session:
        tasks = [fetch_readme(session, link, create_local_filename(base_output_dir, link)) for link in all_links]
        results = await asyncio.gather(*tasks)
    for url, downloaded_path in results:
        cache[url] = downloaded_path

def bulk_download_github_urls(all_links, base_output_dir, num_workers=8):
    #asyncio.run(async_download_github_urls(all_links, base_output_dir, num_workers))
    def process_link(link):
        """ single URL downloading """
        if link in cache:
            return link, cache[link]
        local_filename = create_local_filename(base_output_dir, link)
        if os.path.exists(local_filename):
            cache[link] = local_filename
            return link, local_filename
        downloaded_path = download_readme(link, local_filename)
        downloaded_path = downloaded_path.split("CitationLake/", 1)[-1] if isinstance(downloaded_path, str) and "CitationLake/" in downloaded_path else downloaded_path
        cache[link] = downloaded_path
        return link, downloaded_path
    #results = Parallel(n_jobs=num_workers)(
    #    delayed(process_link)(link) for link in tqdm(all_links, desc="Bulk Download")
    #)
    with tqdm_joblib(tqdm(desc="Bulk Download", total=len(all_links))) as progress_bar:
        results = Parallel(n_jobs=num_workers)(
            delayed(process_link)(link) for link in all_links
        )
    for link, downloaded_path in results:
        cache[link] = downloaded_path

skipped_links = []

def download_readme(github_url, output_path):
    # special domain
    EXCLUDED_TERMS = ['/issues', '/assets', '/sponsor', '/discussions', '/pull', '/tag', '/releases'] # github official tags
    EXCLUDED_SUFFIXES = ['LICENSE']
    if any(term in github_url for term in EXCLUDED_TERMS) or any(github_url.endswith(suffix) for suffix in EXCLUDED_SUFFIXES):
        print(f"Skipping link (excluded pattern): {github_url}")
        skipped_links.append((github_url, "excluded pattern"))
        return None
    #raw_url = clean_github_link(github_url)# clean the url
    raw_url = github_url
    if raw_url.endswith('.git'):
        raw_url = raw_url[:-4]
    if raw_url.endswith(':'):
        raw_url = raw_url[:-1]
    try:
        """if '](' in github_url:
            github_url = github_url.split('](')[0] # only take the first url
        if '"' in github_url and not github_url.startswith('"'):
            github_url = github_url.split('"')[0] # only take the first url"""
        skip_extensions = {"json", "pth", "bin", "ckpt", "pt", "h5", "onnx", "py", "pkl", "tar", "gz", "gif", "png", "jpg", "jpeg", "csv", "ipynb", "mp4", "mov"}
        accept_extensions = {"txt", "rst", "md", "markdown"}
        last_segment = raw_url.split('/')[-1]
        last_segment = re.split(r'[#\?]', last_segment)[0]

        if '.' in last_segment:
            ext = last_segment.rsplit('.', 1)[-1].lower()
            if ext in skip_extensions:
                print(f"Skipping link (extension in skip list): {raw_url}")
                skipped_links.append((raw_url, f"extension .{ext} in skip list"))
                return None
            if ext not in accept_extensions:
                if not is_text_file(raw_url):
                    print(f"Skipping link (not text file): {raw_url}")
                    skipped_links.append((raw_url, "not text file"))
                    return None

        raw_url = '/'.join(raw_url.split('/')[:-1] + [last_segment])
        # get raw url for downloading resources
        if 'gist.github.com' not in raw_url and 'github.com' in raw_url:
            raw_url = raw_url.replace('github.com', 'raw.githubusercontent.com').replace('blob/', '').replace('tree/', '').rstrip("/")
        if raw_url.endswith(('.md', '.rst', '.txt', '.markdown')) or 'gist.github.com' in raw_url:
            response = requests.get(raw_url, timeout=10)
            if response.status_code == 200:
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(response.text)
                return output_path
            else:
                print(f"Error: Failed to download {raw_url} - Status code: {response.status_code}")
            return None
        #if re.search(r'/[^/]+\.(?!txt|md|rst|markdown)[^/]+$', raw_url):
        #    return None
        
        # Define possible README filename variants in priority order
        readme_variants = [
            'README.md', 'README.rst','README.txt', 'README.markdown', 'README', 'Readme.md', 'readme.md','Readme.rst',
            'readme.rst', 'Readme.txt','readme.txt', 'Readme.markdown', 'readme.markdown','Readme','readme',
        ]
        # Determine the base URL based on branch
        if any(sub in raw_url for sub in ['master', 'main']):
            base_url = raw_url.rstrip('/')
        else:
            #base_url = f"{raw_url.rstrip('/')}/master"
            base_url = raw_url
            pass
        # Generate all possible README URLs
        #readme_urls = [f"{base_url}/{variant}" for variant in readme_variants]
        readme_urls = [github_url] + [f"{base_url}/{variant}" for variant in readme_variants] + [f"{base_url}/master/{variant}" for variant in readme_variants] + [f"{base_url}/main/{variant}" for variant in readme_variants]
        # Attempt to download one variant from readme
        for readme_url in readme_urls:
            try:
                response = requests.get(readme_url, timeout=10)
                if response.status_code == 200:
                    with open(output_path, "w", encoding="utf-8") as f:
                        f.write(response.text)
                    return output_path
            except Exception as e:
                #print(f"Warning: Failed to download {readme_url} - {e}")
                pass
                #print(f"Warning: Failed to download {readme_url} - {e}")
        #print(f"Error: All README attempts failed for {raw_url}")
        #return None
        html_response = requests.get(raw_url, timeout=10)
        if html_response.status_code == 200:
            md_text = html2text.html2text(html_response.text)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(md_text)
            return output_path
        else:
            print(f"Error: HTML fallback failed for github_url {github_url}, raw_url {raw_url} - Status code: {html_response.status_code}")
            skipped_links.append((github_url, "HTML fallback failed"))
        return None
    except Exception as e:
        print(f"Error: Exception occurred while downloading github_url {github_url}, raw_url {raw_url} - {e}")
        skipped_links.append((raw_url, "exception occurred"))
        return None

if __name__ == "__main__":
    config = load_config('config.yaml')
    processed_base_path = os.path.join(config.get('base_path'), "processed")
    base_output_dir = os.path.join(config.get('base_path'), "downloaded_github_readmes")
    data_type = "modelcard"
    os.makedirs(base_output_dir, exist_ok=True)
    df_split_temp = pd.read_parquet(os.path.join(processed_base_path, f'{data_type}_step1.parquet'), columns=['github_link', 'modelId'])
    print(df_split_temp.info())
    df_split_temp['github_link'] = df_split_temp['github_link'].apply(lambda x: eval(x) if isinstance(x, str) and x.startswith('[') and x.endswith(']') else x)
    #df_split_temp['github_link'] = df_split_temp['github_link'].apply(lambda x: clean_github_link(x) if isinstance(x, str) else x)
    download_info_df = main_download(df_split_temp, base_output_dir, to_path=os.path.join(processed_base_path, 'github_readmes_info.parquet'))
    #print(download_info_df.head())

"""
Exampled output:

Found 664011 GitHub links before deduplication.
Found 28293 unique GitHub URLs after deduplication.
Speedup ratio: 23.47x reduction in requests.
Step1 time cost: 3.16 seconds.
"""
