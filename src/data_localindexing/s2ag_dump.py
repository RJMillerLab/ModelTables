"""
Author: Zhengyuan Dong
Date: 2025-02-15
Description: This script demonstrates how to download the latest Semantic Scholar datasets using the API.
Error: The application for Semantic Scholar API key has been prohibited due to the their capacity.
"""

import requests
import os

# Step 1: Get the latest release information
latest_release = requests.get("https://api.semanticscholar.org/datasets/v1/release/latest").json()
release_id = latest_release["release_id"]
print(f"Latest Release ID: {release_id}")

# Step 2: List all datasets in the release
datasets = latest_release["datasets"]
print("Available Datasets:")
for dataset in datasets:
    print(f"- {dataset['name']}: {dataset['description']}")

# Step 3: Choose a dataset to download (e.g., "abstracts")
dataset_name = "citations"  # Replace with the dataset you want
dataset_info = requests.get(f"https://api.semanticscholar.org/datasets/v1/release/{release_id}/dataset/{dataset_name}").json()
print("Full response:", dataset_info)
if "error" in dataset_info:
    print(f"Error: {dataset_info['error']}")

# Step 4: Download all files for the chosen dataset
download_folder = f"./s2ag_datasets/{dataset_name}"
os.makedirs(download_folder, exist_ok=True)

print(f"Downloading {dataset_name} dataset...")
print(dataset_info.keys())
for file_url in dataset_info["files"]:
    file_name = os.path.join(download_folder, file_url.split("/")[-1])
    response = requests.get(file_url, stream=True)
    with open(file_name, "wb") as file:
        for chunk in response.iter_content(chunk_size=1024):
            file.write(chunk)
    print(f"Downloaded: {file_name}")

print(f"All files for {dataset_name} have been downloaded to {download_folder}")

