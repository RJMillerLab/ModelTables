"""
Download Hugging Face dataset: librarian-bots/model_cards_with_metadata or dataset_cards_with_metadata
Save as parquet files to data/raw_<date>

This script uses huggingface_hub to download the parquet files directly.
Supports both model cards and dataset cards.
"""

import os
import argparse
import shutil
from pathlib import Path
import subprocess
import sys
from huggingface_hub import hf_hub_download, list_repo_files
import pyarrow.parquet as pq

# Configuration defaults
MODELCARD_DATASET = "librarian-bots/model_cards_with_metadata"
DATASETCARD_DATASET = "librarian-bots/dataset_cards_with_metadata"
DEFAULT_OUTPUT_DIR = Path("data/raw_251117")

def parse_args():
    parser = argparse.ArgumentParser(description="Download Hugging Face model_cards_with_metadata or dataset_cards_with_metadata snapshot")
    parser.add_argument("--date", type=str, default=None, help="Date tag (e.g., 251117). Output dir becomes data/raw_<date>.")
    parser.add_argument("--output-dir", type=str, default=None, help="Custom output directory (overrides --date).")
    parser.add_argument("--type", type=str, choices=["modelcard", "datasetcard"], default="modelcard", help="Type of cards to download: 'modelcard' or 'datasetcard' (default: modelcard).")
    parser.add_argument("--dataset-name", type=str, default=None, help="Hugging Face dataset ID (overrides --type).")
    return parser.parse_args()

def main():
    args = parse_args()

    # Determine dataset name
    if args.dataset_name:
        dataset_name = args.dataset_name
    elif args.type == "datasetcard":
        dataset_name = DATASETCARD_DATASET
    else:
        dataset_name = MODELCARD_DATASET
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif args.date:
        output_dir = Path("data") / f"raw_{args.date}"
    else:
        output_dir = DEFAULT_OUTPUT_DIR
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading dataset: {dataset_name}")
    print(f"Output directory: {output_dir.absolute()}")
    
    # List files in the dataset repository
    files = list_repo_files(dataset_name, repo_type="dataset")
    
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
            repo_id=dataset_name,
            filename=parquet_file,
            repo_type="dataset",
            local_dir=str(output_dir)
        )
        
        # Rename datasetcard files to avoid conflict with modelcard files
        # Modelcard files: train-*.parquet (keep as is)
        # Datasetcard files: datasetcard-train-*.parquet (add prefix)
        if args.type == "datasetcard":
            src_path = Path(local_path)
            # Add prefix to avoid conflict with modelcard files
            new_name = f"datasetcard-{src_path.name}"
            dst_path = src_path.parent / new_name
            if src_path != dst_path:
                shutil.move(str(src_path), str(dst_path))
                print(f"  ↻ Renamed to {new_name} to avoid conflict with modelcard files")
                local_path = str(dst_path)
        
        downloaded_files.append(local_path)
        print(f"✓ Downloaded to {local_path}")
    
    # flatten: parquets under subdirs (e.g. data/) → output_dir
    for p in output_dir.rglob("*.parquet"):
        if p.parent != output_dir:
            shutil.move(str(p), str(output_dir / p.name))
    for subdir in sorted(output_dir.iterdir(), key=lambda x: len(x.parts), reverse=True):
        if subdir.is_dir() and not any(subdir.iterdir()):
            subdir.rmdir()
    downloaded_files = [str(p) for p in output_dir.glob("*.parquet")]
    
    print(f"\n✓ All files downloaded successfully to {output_dir.absolute()}")
    
    # Show file sizes
    total_size = 0
    for file_path in downloaded_files:
        size = Path(file_path).stat().st_size
        total_size += size
        print(f"  {Path(file_path).name}: {size / (1024**3):.2f} GB")
    print(f"  Total size: {total_size / (1024**3):.2f} GB")

if __name__ == "__main__":
    main()

