"""
Author: Zhengyuan Dong
Created: 2025-02-24
Last Modified: 2025-02-24
Description: Download READMEs from the GitHub URLs.
"""
import os
import pandas as pd
import numpy as np
import requests
from tqdm import tqdm
from src.utils import load_config
import urllib.parse

def parse_github_link(github_link):
    parsed_url = urllib.parse.urlparse(github_link)
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
        if pd.notna(github_link_sample) and pd.notna(model_id):
            user, repo = parse_github_link(github_link_sample)
            if user and repo:
                readme_path = os.path.join(base_output_dir, f"{user}_{repo}_README.md")
                if os.path.exists(readme_path):
                    downloaded_path = readme_path
                else:
                    downloaded_path = download_readme(github_link_sample, readme_path)
                download_info.append({
                    "modelId": model_id,
                    "github_link": github_link_sample,
                    "readme_path": downloaded_path if downloaded_path else None
                })
    download_info_df = pd.DataFrame(download_info)
    download_info_df.to_parquet(to_path, index=False)
    print(f"Downloaded {len([d for d in download_info if d['readme_path']])} READMEs.")
    print(f"Skipped {len([d for d in download_info if not d['readme_path']])} READMEs.")
    return download_info_df

def download_readme(github_url, output_path):
    try:
        if isinstance(github_url, list):
            github_url = github_url[0]
        raw_url = github_url.replace("github.com", "raw.githubusercontent.com").rstrip("/") + "/main/README.md"
        response = requests.get(raw_url, timeout=10)
        if response.status_code == 200:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(response.text)
            return output_path
        else:
            print(f"Error: Unable to download README from {raw_url}. Status code: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error: Exception occurred while downloading {github_url} - {e}")
        return None

if __name__ == "__main__":
    config = load_config('config.yaml')
    processed_base_path = os.path.join(config.get('base_path'), "processed")

    base_output_dir = os.path.join(config.get('base_path'), "github_readmes")
    os.makedirs(base_output_dir, exist_ok=True)

    df_split_temp = pd.read_parquet(os.path.join(processed_base_path, 'tmp_df_split_temp.parquet'), columns=['github_link', 'modelId'])
    df_split_temp['github_link'] = df_split_temp['github_link'].apply(lambda x: list(x) if x else x)
    download_info_df = main_download(df_split_temp, base_output_dir, to_path=os.path.join(processed_base_path, 'github_readmes_info.parquet'))
    #print(download_info_df.head())