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

# Configuration defaults
MODELCARD_DATASET = "librarian-bots/model_cards_with_metadata"
DATASETCARD_DATASET = "librarian-bots/dataset_cards_with_metadata"
DEFAULT_OUTPUT_DIR = Path("data/raw_251116")

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

def parse_args():
    parser = argparse.ArgumentParser(description="Download Hugging Face model_cards_with_metadata or dataset_cards_with_metadata snapshot")
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Date tag (e.g., 251117). Output dir becomes data/raw_<date>. If downloading to 1118, will move to 1117 after download."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Custom output directory (overrides --date)."
    )
    parser.add_argument(
        "--type",
        type=str,
        choices=["modelcard", "datasetcard"],
        default="modelcard",
        help="Type of cards to download: 'modelcard' or 'datasetcard' (default: modelcard)."
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Hugging Face dataset ID (overrides --type)."
    )
    parser.add_argument(
        "--move-to-date",
        type=str,
        default=None,
        help="After download, move files to this date directory (e.g., 251117). Useful when downloading on 1118 but want to use 1117 tag."
    )
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
    
    # Determine target date for moving files
    move_to_date = args.move_to_date or args.date
    
    # Try to import required packages
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError:
        print("Installing huggingface_hub...")
        if not install_package("huggingface_hub"):
            print("Please install huggingface_hub manually: pip install huggingface_hub")
            return
        from huggingface_hub import hf_hub_download, list_repo_files
    
    try:
        import pyarrow.parquet as pq
    except ImportError:
        print("Installing pyarrow...")
        if not install_package("pyarrow"):
            print("Please install pyarrow manually: pip install pyarrow")
            return
        import pyarrow.parquet as pq
    
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
    
    print(f"\n✓ All files downloaded successfully to {output_dir.absolute()}")
    
    # Show file sizes
    total_size = 0
    for file_path in downloaded_files:
        size = Path(file_path).stat().st_size
        total_size += size
        print(f"  {Path(file_path).name}: {size / (1024**3):.2f} GB")
    print(f"  Total size: {total_size / (1024**3):.2f} GB")
    
    # Move files to target date directory if specified
    if move_to_date and move_to_date != args.date:
        target_dir = Path("data") / f"raw_{move_to_date}"
        if target_dir != output_dir:
            print(f"\n📦 Moving files from {output_dir} to {target_dir}...")
            target_dir.mkdir(parents=True, exist_ok=True)
            
            # Move all parquet files
            for file_path in downloaded_files:
                src = Path(file_path)
                # Keep the renamed filename (with datasetcard- prefix if applicable)
                dst = target_dir / src.name
                if dst.exists():
                    print(f"  ⚠️  {dst.name} already exists, skipping...")
                else:
                    shutil.move(str(src), str(dst))
                    print(f"  ✓ Moved {src.name} to {target_dir}")
            
            # Remove source directory if empty
            try:
                if output_dir.exists() and not any(output_dir.iterdir()):
                    output_dir.rmdir()
                    print(f"  ✓ Removed empty directory {output_dir}")
            except:
                pass
            
            print(f"\n✓ All files moved to {target_dir.absolute()}")

if __name__ == "__main__":
    main()

