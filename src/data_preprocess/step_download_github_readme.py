"""
Author: Zhengyuan Dong
Created: 2025-02-24
Last Modified: 2025-02-24
Description: Download READMEs from the GitHub URLs.
"""
import os
import pandas as pd
import requests
from tqdm import tqdm

# Assuming df_split_temp is available and contains 'github_link' and 'modelId' columns
# Replace this with actual DataFrame loading if necessary
#df_split_temp = pd.read_parquet('data/modelcard_step3_markdown_gated.parquet')

def main_download(df, to_path="data/github_readmes_info.csv"):
    assert 'github_link' in df.columns, "Missing 'github_link' column in DataFrame."
    assert 'modelId' in df.columns, "Missing 'modelId' column in DataFrame."
    download_info = []
    for index, row in tqdm(df.iterrows(), total=len(df)):
        github_link = row['github_link']
        model_id = row['modelId']
        if isinstance(github_link, list) and len(github_link) > 0:
            github_link = github_link[0]
        elif not isinstance(github_link, str):
            print(f"Skipping invalid github_link for modelId {model_id}: {github_link}")
            continue
        if pd.notna(github_link) and pd.notna(model_id):
            model_dir = os.path.join(base_output_dir, model_id)
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
                print(f"Directory created: {model_dir}")
            else:
                print(f"Directory exists: {model_dir}")
            readme_path = os.path.join(model_dir, "README.md")
            if os.path.exists(readme_path):
                print(f"README already exists for modelId {model_id}, skipping download.")
                downloaded_path = readme_path
            else:
                downloaded_path = download_readme(github_link, readme_path)
            download_info.append({
                "modelId": model_id,
                "github_link": github_link,
                "readme_path": downloaded_path if downloaded_path else None
            })
    download_info_df = pd.DataFrame(download_info)
    download_info_df.to_csv(to_path, index=False)
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
    base_output_dir = "github_readmes"
    os.makedirs(base_output_dir, exist_ok=True)

    df_split_temp = pd.read_csv('data/tmp_df_split_temp.csv')

    to_path = "data/github_readmes_info.csv"
    download_info_df = main_download(df_split_temp, to_path=to_path)
    print(download_info_df.head())
    print(f"Downloaded READMEs saved to '{to_path}'.")