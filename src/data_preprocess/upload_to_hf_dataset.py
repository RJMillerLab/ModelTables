"""
Upload final results to Hugging Face dataset

This script:
1. Prepares the final parquet file (modelcard_step3_dedup_v2_<tag>.parquet)
2. Creates zip files for 4 resources (hugging, github, html, llm)
3. Uploads everything to Hugging Face dataset

Usage:
    python -m src.data_preprocess.upload_to_hf_dataset --tag 251117 --dataset-name your-username/your-dataset
    python -m src.data_preprocess.upload_to_hf_dataset --tag 251117 --dataset-name your-username/your-dataset --dry-run  # Preview commands only
"""

import os
import argparse
import zipfile
from pathlib import Path
import pandas as pd
from typing import Optional, List, Dict

def load_config(config_path='config.yaml'):
    """Load configuration from YAML file"""
    try:
        import yaml
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except:
        return {'base_path': 'data'}

def prepare_parquet_file(parquet_path: str, output_dir: str, tag: Optional[str] = None):
    """
    Copy parquet file to output directory for upload
    
    Args:
        parquet_path: Path to modelcard_step3_dedup_v2_<tag>.parquet
        output_dir: Directory to save file
        tag: Tag suffix for versioning
    
    Returns:
        Path to parquet file in output directory
    """
    print(f"\n{'='*60}")
    print("STEP 1: Preparing parquet file")
    print(f"{'='*60}")
    
    if not os.path.exists(parquet_path):
        print(f"❌ Error: Parquet file not found: {parquet_path}")
        return None
    
    # Get file info
    file_size = os.path.getsize(parquet_path) / (1024**2)  # MB
    print(f"Source file: {parquet_path}")
    print(f"File size: {file_size:.2f} MB")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Copy to output directory
    suffix = f"_{tag}" if tag else ""
    filename = f"modelcard_step3_dedup_v2{suffix}.parquet"
    output_path = os.path.join(output_dir, filename)
    
    print(f"Copying to: {output_path}")
    import shutil
    shutil.copy2(parquet_path, output_path)
    print(f"✓ Prepared: {output_path}")
    
    return output_path

def create_resource_zips(resource_dirs: Dict[str, str], output_dir: str, tag: Optional[str] = None):
    """
    Create zip files for each resource
    
    Args:
        resource_dirs: Dict mapping resource name to directory path
        output_dir: Directory to save zip files
        tag: Tag suffix for versioning
    
    Returns:
        List of created zip file paths
    """
    print(f"\n{'='*60}")
    print("STEP 2: Creating zip files for resources")
    print(f"{'='*60}")
    
    os.makedirs(output_dir, exist_ok=True)
    suffix = f"_{tag}" if tag else ""
    zip_files = []
    
    for resource_name, resource_dir in resource_dirs.items():
        if not os.path.exists(resource_dir):
            print(f"⚠️  Warning: {resource_dir} does not exist, skipping...")
            continue
        
        zip_path = os.path.join(output_dir, f"{resource_name}_tables{suffix}.zip")
        print(f"\nCreating zip: {zip_path}")
        print(f"  Source: {resource_dir}")
        
        # Count files
        file_count = sum(1 for _ in Path(resource_dir).rglob('*.csv'))
        print(f"  Files to zip: {file_count}")
        
        # Create zip file
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for csv_file in Path(resource_dir).rglob('*.csv'):
                # Store relative path within zip
                arcname = csv_file.relative_to(resource_dir)
                zipf.write(csv_file, arcname)
        
        # Get zip file size
        zip_size = os.path.getsize(zip_path) / (1024**3)  # GB
        print(f"✓ Created: {zip_path} ({zip_size:.2f} GB)")
        zip_files.append(zip_path)
    
    return zip_files

def upload_to_hf_dataset(
    dataset_name: str,
    parquet_file: str,
    zip_files: List[str],
    tag: Optional[str] = None,
    dry_run: bool = True
):
    """
    Upload files to Hugging Face dataset
    
    Args:
        dataset_name: Hugging Face dataset name (e.g., 'username/dataset-name')
        parquet_file: Path to modelcard_step3_dedup parquet file
        zip_files: List of zip files to upload
        tag: Tag suffix for versioning
        dry_run: If True, only print commands without executing
    """
    print(f"\n{'='*60}")
    print("STEP 3: Upload to Hugging Face Dataset")
    print(f"{'='*60}")
    
    if dry_run:
        print("🔍 DRY RUN MODE - Will preview commands only")
    else:
        print("🚀 EXECUTION MODE - Will upload files")
    
    print(f"\nDataset: {dataset_name}")
    
    # Check if huggingface_hub is installed
    try:
        from huggingface_hub import HfApi, login
        hf_available = True
    except ImportError:
        print("\n⚠️  huggingface_hub not installed. Install with: pip install huggingface_hub")
        hf_available = False
        return
    
    # Collect all files to upload
    all_files = [parquet_file] + zip_files
    all_files = [f for f in all_files if f and os.path.exists(f)]
    
    if not all_files:
        print("\n❌ Error: No files found to upload")
        return
    
    print(f"\nFiles to upload ({len(all_files)}):")
    total_size = 0
    for file_path in all_files:
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            total_size += file_size
            if file_size > 1024**3:  # GB
                size_str = f"{file_size / (1024**3):.2f} GB"
            else:  # MB
                size_str = f"{file_size / (1024**2):.2f} MB"
            print(f"  - {os.path.basename(file_path)} ({size_str})")
    
    if total_size > 1024**3:
        print(f"\nTotal size: {total_size / (1024**3):.2f} GB")
    else:
        print(f"\nTotal size: {total_size / (1024**2):.2f} MB")
    
    if dry_run:
        print("\n📝 Upload commands (DRY RUN - not executed):")
        print("\n# Step 1: Login (if not already)")
        print("huggingface-cli login")
        print("\n# Step 2: Create dataset (if not exists)")
        print(f"huggingface-cli repo create {dataset_name} --type dataset")
        print("\n# Step 3: Upload files using Python:")
        print("from huggingface_hub import HfApi")
        print("api = HfApi()")
        print(f'dataset_id = "{dataset_name}"')
        for file_path in all_files:
            filename = os.path.basename(file_path)
            print(f'api.upload_file(path_or_fileobj="{file_path}", path_in_repo="{filename}", repo_id=dataset_id, repo_type="dataset")')
        
        print("\n# Or use CLI:")
        for file_path in all_files:
            filename = os.path.basename(file_path)
            print(f'huggingface-cli upload {dataset_name} {file_path} {filename}')
        
        print("\n💡 To actually upload, run with --no-dry-run flag")
    else:
        # Actually upload
        print("\n🚀 Starting upload...")
        
        # Login check
        try:
            api = HfApi()
            # Try to get repo info to check if logged in
            try:
                api.repo_info(repo_id=dataset_name, repo_type="dataset")
                print(f"✓ Dataset exists: {dataset_name}")
            except Exception as e:
                error_str = str(e).lower()
                if "authentication" in error_str or "login" in error_str:
                    print("⚠️  Not logged in. Please login first:")
                    print("   huggingface-cli login")
                    login()
                elif "not found" in error_str or "404" in error_str:
                    # Dataset doesn't exist, create it
                    print(f"📝 Dataset not found. Creating: {dataset_name}")
                    try:
                        api.create_repo(
                            repo_id=dataset_name,
                            repo_type="dataset",
                            exist_ok=False
                        )
                        print(f"✓ Created dataset: {dataset_name}")
                    except Exception as create_error:
                        print(f"⚠️  Failed to create dataset: {create_error}")
                        print("   You may need to create it manually on Hugging Face website")
                        print(f"   Or run: huggingface-cli repo create {dataset_name} --type dataset")
                else:
                    print(f"⚠️  Error checking dataset: {e}")
        except Exception as e:
            print(f"⚠️  Login check failed: {e}")
            print("   Please login manually: huggingface-cli login")
            try:
                login()
            except:
                print("❌ Login failed. Please login manually and try again.")
                return
        
        # Upload files
        api = HfApi()
        for file_path in all_files:
            if not os.path.exists(file_path):
                print(f"⚠️  Skipping (not found): {file_path}")
                continue
            
            filename = os.path.basename(file_path)
            print(f"\n📤 Uploading {filename}...")
            try:
                api.upload_file(
                    path_or_fileobj=file_path,
                    path_in_repo=filename,
                    repo_id=dataset_name,
                    repo_type="dataset"
                )
                print(f"✓ Uploaded {filename}")
            except Exception as e:
                print(f"❌ Failed to upload {filename}: {e}")
        
        print(f"\n✅ Upload complete! Check: https://huggingface.co/datasets/{dataset_name}")

def main():
    parser = argparse.ArgumentParser(
        description="Upload final results to Hugging Face dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preview commands only (dry run)
  python -m src.data_preprocess.upload_to_hf_dataset --tag 251117 --dataset-name username/dataset-name
  
  # Actually upload (remove --dry-run)
  python -m src.data_preprocess.upload_to_hf_dataset --tag 251117 --dataset-name username/dataset-name --no-dry-run
        """
    )
    parser.add_argument('--tag', dest='tag', default=None,
                       help='Tag suffix for versioning (e.g., 251117). Required for most scripts.')
    parser.add_argument('--dataset-name', dest='dataset_name', required=True,
                       help='Hugging Face dataset name (e.g., username/dataset-name)')
    parser.add_argument('--parquet', dest='parquet', default=None,
                       help='Path to modelcard_step3_dedup parquet (default: auto-detect from tag)')
    parser.add_argument('--output-dir', dest='output_dir', default=None,
                       help='Directory to save extracted files and zips (default: data/upload_<tag>)')
    parser.add_argument('--dry-run', dest='dry_run', action='store_true', default=True,
                       help='Only generate commands, do not execute (default: True)')
    parser.add_argument('--no-dry-run', dest='dry_run', action='store_false',
                       help='Actually execute upload commands')
    
    args = parser.parse_args()
    
    if not args.tag:
        print("⚠️  Warning: --tag not provided. Using 'latest' as suffix.")
        tag = None
    else:
        tag = args.tag
    
    # Load config
    config = load_config('config.yaml')
    base_path = config.get('base_path', 'data')
    processed_base_path = os.path.join(base_path, 'processed')
    suffix = f"_{tag}" if tag else ""
    
    # Determine paths
    if args.parquet:
        parquet_path = args.parquet
    else:
        parquet_path = os.path.join(processed_base_path, f"modelcard_step3_dedup_v2{suffix}.parquet")
    
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(base_path, f"upload{suffix}")
    
    # Resource directories
    resource_dirs = {
        'hugging': os.path.join(processed_base_path, f"deduped_hugging_csvs_v2{suffix}"),
        'github': os.path.join(processed_base_path, f"deduped_github_csvs_v2{suffix}"),
        'html': os.path.join(processed_base_path, f"tables_output_v2{suffix}"),
        'llm': os.path.join(processed_base_path, f"llm_tables{suffix}")
    }
    
    print(f"\n{'='*60}")
    print("UPLOAD TO HUGGING FACE DATASET")
    print(f"{'='*60}")
    print(f"Tag: {tag or 'latest'}")
    print(f"Dataset: {args.dataset_name}")
    print(f"Parquet file: {parquet_path}")
    print(f"Output directory: {output_dir}")
    print(f"Mode: {'DRY RUN (preview only)' if args.dry_run else 'EXECUTE'}")
    
    # Check if parquet file exists
    if not os.path.exists(parquet_path):
        print(f"\n❌ Error: Parquet file not found: {parquet_path}")
        print("   Please run the pipeline first or specify --parquet")
        return
    
    # Step 1: Prepare parquet file (copy to output directory)
    parquet_file = prepare_parquet_file(parquet_path, output_dir, tag)
    
    if not parquet_file:
        print("\n❌ Error: Failed to prepare parquet file")
        return
    
    # Step 2: Create zip files
    zip_files = create_resource_zips(resource_dirs, output_dir, tag)
    
    # Step 3: Upload to Hugging Face
    upload_to_hf_dataset(
        args.dataset_name,
        parquet_file,
        zip_files,
        tag,
        dry_run=args.dry_run
    )
    
    print(f"\n✅ Done!")
    if args.dry_run:
        print(f"   Files prepared in: {output_dir}")
        print(f"   Run with --no-dry-run to actually upload")
    else:
        print(f"   Files uploaded to: {args.dataset_name}")

if __name__ == "__main__":
    main()

