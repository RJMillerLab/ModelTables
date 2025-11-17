"""
Download Hugging Face dataset: librarian-bots/model_cards_with_metadata
Save as parquet files to data/raw_251116

This script uses huggingface_hub to download the parquet files directly.
"""

import os
from pathlib import Path
import subprocess
import sys

# Configuration
DATASET_NAME = "librarian-bots/model_cards_with_metadata"
OUTPUT_DIR = Path("data/raw_251116")

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--break-system-packages", package])
    except subprocess.CalledProcessError:
        print(f"Warning: Could not install {package}, trying without --break-system-packages")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        except subprocess.CalledProcessError as e:
            print(f"Error installing {package}: {e}")
            return False
    return True

def main():
    # Try to import required packages
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError:
        print("Installing huggingface_hub...")
        if not install_package("huggingface_hub"):
            print("Please install huggingface_hub manually: pip install huggingface_hub")
            return
    
    try:
        import pyarrow.parquet as pq
    except ImportError:
        print("Installing pyarrow...")
        if not install_package("pyarrow"):
            print("Please install pyarrow manually: pip install pyarrow")
            return
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading dataset: {DATASET_NAME}")
    print(f"Output directory: {OUTPUT_DIR.absolute()}")
    
    # List files in the dataset repository
    files = list_repo_files(DATASET_NAME, repo_type="dataset")
    
    # Filter for parquet files
    parquet_files = [f for f in files if f.endswith('.parquet')]
    
    print(f"Found {len(parquet_files)} parquet file(s):")
    for f in parquet_files:
        print(f"  - {f}")
    
    # Download each parquet file
    downloaded_files = []
    for parquet_file in parquet_files:
        print(f"\nDownloading {parquet_file}...")
        local_path = hf_hub_download(
            repo_id=DATASET_NAME,
            filename=parquet_file,
            repo_type="dataset",
            local_dir=str(OUTPUT_DIR)
        )
        downloaded_files.append(local_path)
        print(f"✓ Downloaded to {local_path}")
    
    print(f"\n✓ All files downloaded successfully to {OUTPUT_DIR.absolute()}")
    
    # Show file sizes
    total_size = 0
    for file_path in downloaded_files:
        size = Path(file_path).stat().st_size
        total_size += size
        print(f"  {Path(file_path).name}: {size / (1024**3):.2f} GB")
    print(f"  Total size: {total_size / (1024**3):.2f} GB")

if __name__ == "__main__":
    main()

