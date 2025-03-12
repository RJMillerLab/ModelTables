"""
Author: Zhengyuan Dong
Date: 2025-02-15
Description: This script downloads the OpenAlex snapshot files from the OpenAlex S3 bucket.
Issue: This downloading process do not download anything. While using aws cli is more efficient. However the whole data requires 300GB of space. So API might be a better choice.
"""

import os
import requests
from tqdm import tqdm

SNAPSHOT_BASE_URL = "https://openalex.s3.amazonaws.com/snapshot/"

SNAPSHOT_FILES = [
    "works",
    "authors",
    "institutions",
    "sources",
    "concepts",
    "publishers",
    "funders"
]

DOWNLOAD_DIR = "./openalex_snapshot"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

def download_snapshot_file(entity_type):
    """Download all parts of the specified entity type."""
    part_num = 1
    while True:
        file_url = f"{SNAPSHOT_BASE_URL}{entity_type}/part{part_num:02}.jsonl.gz"
        file_path = os.path.join(DOWNLOAD_DIR, f"{entity_type}_part{part_num:02}.jsonl.gz")
        response = requests.head(file_url)
        if response.status_code != 200:
            break  # No more parts to download
        print(f"Downloading {file_url}...")
        with requests.get(file_url, stream=True) as r:
            with open(file_path, "wb") as file:
                for chunk in r.iter_content(chunk_size=1024):
                    file.write(chunk)
        print(f"Downloaded {file_path}")
        part_num += 1
    print(f"All parts for {entity_type} downloaded.")

def main():
    for entity_type in SNAPSHOT_FILES:
        print(f"\nStarting download for {entity_type}...")
        download_snapshot_file(entity_type)

if __name__ == "__main__":
    main()

