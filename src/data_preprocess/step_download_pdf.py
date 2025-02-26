"""
Author: Zhengyuan Dong
Created: 2025-02-24
Last Modified: 2025-02-24
Description: Download PDFs from the final URLs.
"""

import os
import pandas as pd
import requests
from tqdm import tqdm
from src.utils import load_config

def download_pdf(url, output_path):
    if os.path.exists(output_path):
        return output_path
    try:
        response = requests.get(url, stream=True, timeout=10)
        if response.status_code == 200 and "application/pdf" in response.headers.get("Content-Type", ""):
            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024):
                    f.write(chunk)
            return output_path
        else:
            return None
    except Exception as e:
        return None

def main_download(df, base_output_dir="downloaded_pdfs_by_model", to_path="data/downloaded_pdfs_info.csv"):
    assert 'final_url' in df.columns, "Missing 'final_url' column in DataFrame."
    assert 'modelId' in df.columns, "Missing 'modelId' column in DataFrame."
    os.makedirs(base_output_dir, exist_ok=True)
    download_info = []
    for index, row in tqdm(df.iterrows(), total=len(df)):
        url = row['final_url']
        model_id = row['modelId']
        
        if pd.notna(url) and pd.notna(model_id):
            model_dir = os.path.join(base_output_dir, model_id)
            os.makedirs(model_dir, exist_ok=True)
            
            file_name = os.path.basename(url.split('/')[-1])
            file_path = os.path.join(model_dir, file_name)
            
            downloaded_path = download_pdf(url, file_path)
            
            download_info.append({
                "modelId": model_id,
                "final_url": url,
                "local_path": downloaded_path if downloaded_path else None
            })
    download_info_df = pd.DataFrame(download_info)
    download_info_df.to_csv(to_path, index=False)
    print(f"Downloaded {len([d for d in download_info if d['local_path']])} PDFs.")
    print(f"Skipped {len([d for d in download_info if not d['local_path']])} PDFs.")
    return download_info_df

if __name__ == "__main__":
    config = load_config('config.yaml')
    base_path = config.get('base_path')
    df_processed = pd.read_csv(f"{base_path}/processed_final_urls.csv")
    to_path=f"{base_path}/downloaded_pdfs_info.csv"
    download_info_df = main_download(df_processed, to_path=to_path)
    print(download_info_df.head())
    print(f"Downloaded PDFs saved to '{to_path}'.")
