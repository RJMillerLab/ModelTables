"""
Author: Zhengyuan Dong
Created: 2025-02-24
Last Modified: 2025-02-24
Description: Download READMEs from the GitHub URLs.
"""
import os, re
import pandas as pd
import numpy as np
import requests
from tqdm import tqdm
from src.utils import load_config
import urllib.parse

cache = {}

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

def main_download(df, base_output_dir, to_path="data/github_readmes_info.parquet"):
    assert 'github_link' in df.columns, "Missing 'github_link' column in DataFrame."
    assert 'modelId' in df.columns, "Missing 'modelId' column in DataFrame."
    download_info = []
    for index, row in tqdm(df.iterrows(), total=len(df)):
        github_link = row['github_link']
        model_id = row['modelId']
        if github_link is None:
            continue
        if isinstance(github_link, (list, tuple, np.ndarray)):
            if len(github_link) >= 0:
                github_link_sample = github_link[0]
            else:
                #print(f"Skipping invalid github_link for modelId {model_id}: {github_link} because it is empty.")
                continue
        elif isinstance(github_link, str):
            github_link_sample = github_link
        else:
            print(f"Skipping invalid github_link for modelId {model_id}: {github_link}")
            continue

        if github_link_sample in cache:  # Check if the link is in the cache
            downloaded_path = cache[github_link_sample]
            #if downloaded_path is None:
                #print(f"Warning: Cached link for modelId {model_id} is None.")
        else:
            raw_url = clean_github_link(github_link_sample)
            try:
                user, repo = parse_github_link(raw_url)
                if user and repo:
                    readme_path = os.path.join(base_output_dir, f"{user}_{repo}_README.md")
                    #raw_url = clean_github_link(github_link_sample)
                    if raw_url in cache:
                        downloaded_path = cache[raw_url]
                    else:
                        if os.path.exists(readme_path):
                            downloaded_path = readme_path
                        else:
                            downloaded_path = download_readme(raw_url, readme_path)
                        cache[raw_url] = downloaded_path
            except:
                downloaded_path = None
                print(f"Error: Failed to parse {raw_url}")
                cache[raw_url] = None
        download_info.append({
            "modelId": model_id,
            "github_link": github_link_sample,
            "readme_path": downloaded_path if downloaded_path else None
        })
    download_info_df = pd.DataFrame(download_info)
    download_info_df.to_parquet(to_path, index=False)
    # save the cache as parquet
    cache_df = pd.DataFrame(list(cache.items()), columns=['raw_url', 'downloaded_path'])
    cache_df.to_parquet(os.path.join(config.get('base_path'), "github_readme_cache.parquet"), index=False)
    print(f"Downloaded {len([d for d in download_info if d['readme_path']])} READMEs.")
    print(f"Skipped {len([d for d in download_info if not d['readme_path']])} READMEs.")
    return download_info_df

def download_readme(github_url, output_path):
    # special domain
    if 'issues' in github_url or 'assets' in github_url or 'sponsor' in github_url or 'discussions' in github_url:
        return None
    #raw_url = clean_github_link(github_url)# clean the url
    raw_url = github_url
    if raw_url.endswith('LICENSE'):
        return None
    try:
        """if '](' in github_url:
            github_url = github_url.split('](')[0] # only take the first url
        if '"' in github_url and not github_url.startswith('"'):
            github_url = github_url.split('"')[0] # only take the first url"""
        if raw_url.endswith(':'):
            raw_url = raw_url[:-1]
        if 'gist.github.com' not in raw_url and 'github.com' in raw_url:
            raw_url = raw_url.replace('github.com', 'raw.githubusercontent.com').replace('blob/', '').replace('tree/', '').rstrip("/")
        if '.' in raw_url.split('/')[-1]:
            if raw_url.endswith('.md') or raw_url.endswith('.rst') or raw_url.endswith('.txt') or raw_url.endswith('.markdown'):
                pass
            else:
                return None
        #if re.search(r'/[^/]+\.(?!txt|md|rst|markdown)[^/]+$', raw_url):
        #    return None
        # Directly use the URL if it points to a README file
        if raw_url.endswith(('.md', '.rst', '.txt', '.markdown')) or 'gist.github.com' in raw_url:
            response = requests.get(raw_url, timeout=10)
            if response.status_code == 200:
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(response.text)
                return output_path
            print(f"Error: Failed to download {raw_url} - Status code: {response.status_code}")
            return None
        # Handle cases where URL is a directory
        if raw_url.endswith('.git'):
            raw_url = raw_url[:-4]
        if '#' in raw_url:
            raw_url = raw_url.split('#')[0]
        # Define possible README filename variants in priority order
        readme_variants = [
            'README.md', 'README.rst','README.txt', 'README.markdown', 'README', 'Readme.md', 'readme.md','Readme.rst',
            'readme.rst', 'Readme.txt','readme.txt', 'Readme.markdown', 'readme.markdown','Readme','readme',
        ]
        # Determine the base URL based on branch
        if any(sub in raw_url for sub in ['master', 'main']):
            base_url = raw_url.rstrip('/')
        else:
            base_url = f"{raw_url.rstrip('/')}/master"
        # Generate all possible README URLs
        readme_urls = [f"{base_url}/{variant}" for variant in readme_variants]
        # Attempt to download each variant
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
        print(f"Error: All README attempts failed for {raw_url}")
        return None
    except Exception as e:
        print(f"Error: Exception occurred while downloading {raw_url} - {e}")
        return None

"""def clean_github_link(github_link):
    #cleaned_link = re.sub(r'[\\\]\)</br>]', '', github_link)
    cleaned_link = github_link.replace('{', '').replace('}', '').replace('(','').replace(')','').replace('<','').replace('>','').replace('*','').replace('`','').replace('"','').replace("'",'').replace('!','')#.replace(":",'')#.replace(' ','').replace('\n','').replace('\t','').replace('\r','')
    return cleaned_link"""

"""def clean_github_link(github_link):
    return re.split(r'[{}()<>*`"!]', github_link)[0]"""

def clean_github_link(github_link):
    return github_link.split('{')[0].split('}')[0].split('[')[0].split(']')[0].split('(')[0].split(')')[0].split('<')[0].split('>')[0].split('*')[0].split('`')[0].split('"')[0].split("'")[0].split('!')[0]

if __name__ == "__main__":
    config = load_config('config.yaml')
    processed_base_path = os.path.join(config.get('base_path'), "processed")
    base_output_dir = os.path.join(config.get('base_path'), "downloaded_github_readmes")
    data_type = "modelcard"
    os.makedirs(base_output_dir, exist_ok=True)
    df_split_temp = pd.read_parquet(os.path.join(processed_base_path, f'{data_type}_step1.parquet'), columns=['github_link', 'modelId'])
    df_split_temp['github_link'] = df_split_temp['github_link'].apply(lambda x: eval(x) if isinstance(x, str) and x.startswith('[') and x.endswith(']') else x)
    #df_split_temp['github_link'] = df_split_temp['github_link'].apply(lambda x: clean_github_link(x) if isinstance(x, str) else x)
    download_info_df = main_download(df_split_temp, base_output_dir, to_path=os.path.join(processed_base_path, 'github_readmes_info.parquet'))
    #print(download_info_df.head())