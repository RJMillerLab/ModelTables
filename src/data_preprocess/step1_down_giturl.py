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
from src.utils import load_config, to_parquet, is_list_like, to_list_safe
import urllib.parse
import html2text
import hashlib
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib
import aiohttp
import asyncio
import argparse

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

def main_download(df, base_output_dir, to_path="data/github_readmes_info.parquet", 
                  baseline_info_path=None, baseline_cache_path=None, versioning=False, base_path=None):
    """
    Download GitHub README files.
    
    Args:
        df: DataFrame with modelId and github_link columns
        base_output_dir: Directory to save downloaded files (new folder for this tag)
        to_path: Output path for download info parquet
        baseline_info_path: Path to baseline github_readmes_info.parquet (for versioning mode)
        baseline_cache_path: Path to baseline github_readme_cache.parquet (for versioning mode)
        versioning: If True, reuse existing downloads from baseline (version control mode)
        base_path: Base path of the project (for resolving relative paths)
    """
    assert 'github_link' in df.columns, "Missing 'github_link' column in DataFrame."
    assert 'modelId' in df.columns, "Missing 'modelId' column in DataFrame."
    
    if base_path is None:
        base_path = os.getcwd()

    # step1: get all links
    start_time = time.time()
    all_raw_links = df[['modelId', 'github_link']].explode('github_link').dropna()
    total_links_before_dedup = len(all_raw_links)
    all_links = set(clean_github_link(str(link).strip()) for link in all_raw_links['github_link'] if link)

    print(f"Found {total_links_before_dedup} GitHub links before deduplication.")
    print(f"Found {len(all_links)} unique GitHub URLs after deduplication.")
    
    # Auto-cache: Always check for existing files in base_output_dir
    # This is independent of versioning mode
    existing_in_dir = 0
    for link in all_links:
        local_filename = create_local_filename(base_output_dir, link)
        if os.path.exists(local_filename):
            rel_path = os.path.relpath(local_filename, base_path)
            cache[link] = rel_path
            existing_in_dir += 1
    
    if existing_in_dir > 0:
        print(f"📦 Auto-cache: Found {existing_in_dir:,} existing files in {base_output_dir}")
    
    # Versioning mode: load baseline info and reuse existing downloads from previous versions
    links_to_download = {link for link in all_links if link not in cache or cache.get(link) is None}
    baseline_info_dict = {}  # url -> list of paths from baseline
    if versioning and baseline_info_path:
        baseline_info_path = os.path.expanduser(baseline_info_path)
        if os.path.exists(baseline_info_path):
            print(f"\n🔄 Versioning mode: Loading baseline info from {baseline_info_path}")
            baseline_info_df = pd.read_parquet(baseline_info_path)
            
            # Build a mapping from URL to existing paths
            # Explode readme_path lists to get all URLs and their paths
            for _, row in baseline_info_df.iterrows():
                github_links = row.get('github_link', [])
                readme_paths = row.get('readme_path', [])
                
                # Handle different data types: list, array, single value, or None
                if is_list_like(github_links):
                    github_links = to_list_safe(github_links)
                elif pd.isna(github_links) or github_links is None:
                    github_links = []
                else:
                    github_links = [github_links]
                
                if is_list_like(readme_paths):
                    readme_paths = to_list_safe(readme_paths)
                elif pd.isna(readme_paths) or readme_paths is None:
                    readme_paths = []
                else:
                    readme_paths = [readme_paths]
                
                for link, path in zip(github_links, readme_paths):
                    if link and path:
                        cleaned_link = clean_github_link(str(link).strip())
                        if cleaned_link not in baseline_info_dict:
                            baseline_info_dict[cleaned_link] = []
                        baseline_info_dict[cleaned_link].append(path)
            
            # Check which files exist and populate cache
            existing_count = 0
            for url, paths in baseline_info_dict.items():
                for path in paths:
                    if path:
                        # Try to resolve the path
                        # Path might be relative to base_path or absolute
                        if os.path.isabs(path):
                            full_path = path
                        else:
                            # Try relative to base_path first
                            full_path = os.path.join(base_path, path)
                            if not os.path.exists(full_path):
                                # Try relative to current working directory
                                full_path = path if os.path.exists(path) else None
                        
                        if full_path and os.path.exists(full_path):
                            # Store the original path (preserving folder structure)
                            if url not in cache:
                                cache[url] = path  # Store relative path to preserve folder info
                            existing_count += 1
                            break  # Found existing file, no need to check other paths for this URL
            
            # Update links_to_download to exclude URLs that already have existing files from baseline
            links_to_download = {link for link in links_to_download 
                                if link not in cache or cache.get(link) is None}
            
            print(f"   Baseline info has {len(baseline_info_dict)} unique URLs")
            print(f"   Existing files found from baseline: {existing_count:,}")
            print(f"   Already available (auto-cache + baseline): {len(all_links) - len(links_to_download):,} URLs")
            print(f"   Need to download: {len(links_to_download):,} URLs")
            if len(all_links) > 0:
                print(f"   Total speedup: {(len(all_links) - len(links_to_download)) / len(all_links) * 100:.1f}% reuse")
        else:
            print(f"\n⚠️  Versioning mode enabled but baseline info not found: {baseline_info_path}")
            print("   Continuing with auto-cache only")
    
    print(f"Speedup ratio: {total_links_before_dedup / len(all_links):.2f}x reduction in requests.")
    print(f"Step1 time cost: {time.time() - start_time:.2f} seconds.")

    # step2: download only new/missing links
    start_time = time.time()
    if links_to_download:
        bulk_download_github_urls(links_to_download, base_output_dir, base_path=base_path)
    else:
        print("   All URLs already downloaded, skipping download step.")
    print(f"Step2 time cost: {time.time() - start_time:.2f} seconds.")
    
    # step3: link downloaded files back to the model data
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
        elif is_list_like(raw_links):
            raw_links = to_list_safe(raw_links)

        readme_paths = []
        for g_link in raw_links:
            if not g_link:
                continue
            cleaned_link = clean_github_link(g_link.strip())
            local_path = cache.get(cleaned_link)
            if local_path:
                # Ensure path is relative to base_path (preserving folder structure)
                if os.path.isabs(local_path):
                    # Convert absolute path to relative path
                    try:
                        local_path = os.path.relpath(local_path, base_path)
                    except ValueError:
                        # If paths are on different drives (Windows), keep absolute
                        pass
                readme_paths.append(local_path)
        download_info.append({
            "modelId": model_id,
            "github_link": raw_links,
            "readme_path": readme_paths
        })
    download_info_df = pd.DataFrame(download_info)
    to_parquet(download_info_df, to_path)
    
    # Save the cache as parquet (with relative paths)
    cache_items = []
    for url, path in cache.items():
        if path:
            # Ensure path is relative
            if os.path.isabs(path):
                try:
                    path = os.path.relpath(path, base_path)
                except ValueError:
                    pass
            cache_items.append({'raw_url': url, 'downloaded_path': path})
    if cache_items:
        cache_df = pd.DataFrame(cache_items)
        cache_output_path = os.path.join(os.path.dirname(to_path), "github_readme_cache.parquet")
        if os.path.basename(to_path).startswith('github_readmes_info_'):
            # Extract tag from to_path and add to cache filename
            tag = os.path.basename(to_path).replace('github_readmes_info_', '').replace('.parquet', '')
            cache_output_path = os.path.join(os.path.dirname(to_path), f"github_readme_cache_{tag}.parquet")
        to_parquet(cache_df, cache_output_path)
    
    # Save skipped URLs
    if skipped_links:
        skipped_df = pd.DataFrame(skipped_links, columns=['raw_url', 'reason'])
        skipped_output_path = os.path.join(os.path.dirname(to_path), "github_skipped_urls.parquet")
        if os.path.basename(to_path).startswith('github_readmes_info_'):
            tag = os.path.basename(to_path).replace('github_readmes_info_', '').replace('.parquet', '')
            skipped_output_path = os.path.join(os.path.dirname(to_path), f"github_skipped_urls_{tag}.parquet")
        to_parquet(skipped_df, skipped_output_path)
    
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

def bulk_download_github_urls(all_links, base_output_dir, num_workers=8, base_path=None):
    """Download GitHub URLs to base_output_dir, saving paths relative to base_path"""
    if base_path is None:
        base_path = os.getcwd()
    
    def process_link(link):
        """ single URL downloading """
        if link in cache and cache[link]:
            return link, cache[link]
        local_filename = create_local_filename(base_output_dir, link)
        if os.path.exists(local_filename):
            # Convert to relative path
            rel_path = os.path.relpath(local_filename, base_path)
            cache[link] = rel_path
            return link, rel_path
        downloaded_path = download_readme(link, local_filename)
        if downloaded_path:
            # Convert to relative path
            try:
                downloaded_path = os.path.relpath(downloaded_path, base_path)
            except ValueError:
                # If paths are on different drives (Windows), keep as is
                pass
        cache[link] = downloaded_path
        return link, downloaded_path
    
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

def parse_args():
    parser = argparse.ArgumentParser(description="Download GitHub README files from step1 data")
    parser.add_argument("--input-step1", dest="input_step1", default=None,
                        help="Path to step1 parquet file (default: auto-detect from tag)")
    parser.add_argument("--tag", dest="tag", default=None,
                        help="Tag suffix for versioning (e.g., 251117). Enables versioning mode.")
    parser.add_argument("--versioning", dest="versioning", action="store_true",
                        help="Enable versioning mode: reuse existing downloads from baseline (requires --tag)")
    parser.add_argument("--baseline-info", dest="baseline_info", default=None,
                        help="Path to baseline github_readmes_info.parquet (for versioning mode)")
    parser.add_argument("--baseline-cache", dest="baseline_cache", default=None,
                        help="Path to baseline github_readme_cache.parquet (for versioning mode)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    config = load_config('config.yaml')
    base_path = config.get('base_path')
    processed_base_path = os.path.join(base_path, "processed")
    
    # Determine tag (default to date if not provided)
    tag = args.tag
    
    # Determine output folder and file based on tag
    if tag:
        base_output_dir = os.path.join(base_path, f"downloaded_github_readmes_{tag}")
        output_file = os.path.join(processed_base_path, f'github_readmes_info_{tag}.parquet')
    else:
        base_output_dir = os.path.join(base_path, "downloaded_github_readmes")
        output_file = os.path.join(processed_base_path, 'github_readmes_info.parquet')
    
    data_type = "modelcard"
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Determine input file
    if args.input_step1:
        input_file = args.input_step1
    else:
        input_suffix = f"_{tag}" if tag else ""
        input_file = os.path.join(processed_base_path, f'{data_type}_step1{input_suffix}.parquet')
    
    # Versioning mode: enable if tag is provided or --versioning flag is set
    enable_versioning = args.versioning or (args.tag is not None)
    
    # Determine baseline paths for versioning mode
    baseline_info_path = args.baseline_info
    baseline_cache_path = args.baseline_cache
    baseline_info_path_original = os.path.join(processed_base_path, 'github_readmes_info.parquet')
    
    if enable_versioning:
        if baseline_info_path is None:
            # Default baseline: no tag version (original)
            baseline_info_path = baseline_info_path_original
        if baseline_cache_path is None:
            baseline_cache_path = os.path.join(processed_base_path, 'github_readme_cache.parquet')
        
        print(f"📌 Versioning mode enabled (tag: {tag if tag else 'default'})")
        print(f"   Baseline info: {baseline_info_path}")
        print(f"   Baseline cache: {baseline_cache_path}")
        print(f"   New files will be saved to: {base_output_dir}")
    else:
        print(f"📦 Auto-cache mode: Will reuse existing files in {base_output_dir}")
    
    print(f"📁 Loading input: {input_file}")
    df_split_temp = pd.read_parquet(input_file, columns=['github_link', 'modelId'])
    print(df_split_temp.info())
    df_split_temp['github_link'] = df_split_temp['github_link'].apply(lambda x: eval(x) if isinstance(x, str) and x.startswith('[') and x.endswith(']') else x)
    
    download_info_df = main_download(
        df_split_temp, 
        base_output_dir, 
        to_path=output_file,
        baseline_info_path=baseline_info_path if enable_versioning else None,
        baseline_cache_path=baseline_cache_path if enable_versioning else None,
        versioning=enable_versioning,
        base_path=base_path
    )
    print(f"✅ Results saved to: {output_file}")
    print(f"✅ Downloaded files saved to: {base_output_dir}")

"""
Exampled output:

Found 664011 GitHub links before deduplication.
Found 28293 unique GitHub URLs after deduplication.
Speedup ratio: 23.47x reduction in requests.
Step1 time cost: 3.16 seconds.
"""
