#!/usr/bin/env python3
"""
Author: Zhengyuan Dong
Created: 2025-04-15
Last Modified: 2025-04-30
Script Description:
    This script scans several fixed source folders (relative to the repository root) for CSV files and 
    creates symbolic links in a target directory (also fixed relative to the repository root).
    It uses an incremental cache to only process new files and prints statistics for each folder.
    Parallel processing is implemented with joblib and a progress bar is shown using tqdm.

    The target directory is derived from the fixed base path:
        <repo_root>/starmie_internal/data/scilake_final{dir_suffix}/datalake
    And based on the provided mode, a suffix is appended to the filename:
        - base     -> dir_suffix="",         file_suffix=""
        - str      -> dir_suffix="_str",     file_suffix="_s"
        - tr       -> dir_suffix="_tr",      file_suffix="_t"
        - tr_str   -> dir_suffix="_tr_str",  file_suffix="_s_t"
    
    Mask file support:
        If --mask-file is provided (or auto-loaded when --tag is used), only files listed in the mask
        file will be linked. The mask file should contain full paths, and the script extracts basenames
        to match against source files. This allows filtering to only link valid/processed tables.

Usage:
    python -m src.data_symlink.ln_scilake --repo_root /u4/z6dong/Repo --mode base
    python -m src.data_symlink.ln_scilake --repo_root /u4/z6dong/Repo --mode str
    python -m src.data_symlink.ln_scilake --repo_root /u4/z6dong/Repo --mode all
    python -m src.data_symlink.ln_scilake --repo_root /u4/z6dong/Repo --mode base --tag 251117
    python -m src.data_symlink.ln_scilake --repo_root /u1/z6dong/Repo --mode base --tag 251117 --dir-name scilake_final_251117 --mask-file data/analysis/all_valid_title_valid_251117.txt
    # Note: When --tag is provided, mask file is auto-loaded from data/analysis/all_valid_title_valid_{tag}.txt if it exists
"""

import os
from joblib import Parallel, delayed
from tqdm import tqdm
import argparse

# Mapping from mode to (directory suffix, file suffix)
MODE_SUFFIX = {
    "base":   ("",         ""),     ######## define suffix mapping
    "str":    ("_str",     "_s"),    ########
    "tr":     ("_tr",      "_t"),    ########
    "tr_str": ("_tr_str",  "_s_t"),  ######## renamed from str_tr to tr_str
}


def create_symlink(src, target_dir, cache, file_suffix=""):
    """
    Create a symlink for src in target_dir, appending file_suffix if not already present.
    """
    basename = os.path.basename(src)
    name, ext = os.path.splitext(basename)
    if file_suffix and not basename.endswith(f"{file_suffix}{ext}"):
        target_name = f"{name}{file_suffix}{ext}"  ######## only append if missing
    else:
        target_name = basename
    target_path = os.path.join(target_dir, target_name)
    if basename in cache:
        return False
    try:
        if os.path.lexists(target_path):
            os.remove(target_path)
        os.symlink(src, target_path)
    except Exception as e:
        print(f"Error linking {src} -> {target_path}: {e}")
        return False
    return True


def load_mask_file(mask_file_path):
    """
    Load mask file and return a set of allowed base filenames (without file_suffix).
    The mask file contains full paths, we extract basenames and normalize them.
    Returns a set of base names (e.g., 'file.csv' for both 'file.csv' and 'file_s.csv').
    """
    if not mask_file_path or not os.path.exists(mask_file_path):
        return None
    
    allowed_base_names = set()
    with open(mask_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Extract basename from full path
            basename = os.path.basename(line)
            if basename.endswith('.csv'):
                name, ext = os.path.splitext(basename)
                # Remove file_suffix if present (e.g., 'file_s' -> 'file', 'file_t' -> 'file')
                for suffix in ['_s_t', '_t_s', '_s', '_t']:
                    if name.endswith(suffix):
                        name = name[:-len(suffix)]
                        break
                # Store the base name (without suffix)
                allowed_base_names.add(f"{name}{ext}")
    
    return allowed_base_names


def process_folder(source_folder, target_dir, cache, file_suffix, mask_set=None):
    csvs = [os.path.join(source_folder, f) for f in os.listdir(source_folder)
            if f.lower().endswith('.csv') and os.path.isfile(os.path.join(source_folder, f))]
    
    # Apply mask filter if provided
    if mask_set is not None:
        # Check if file's base name (without file_suffix) is in mask
        filtered_csvs = []
        for csv_path in csvs:
            basename = os.path.basename(csv_path)
            name, ext = os.path.splitext(basename)
            # Remove file_suffix to get base name
            base_name = name
            if file_suffix and name.endswith(file_suffix):
                base_name = name[:-len(file_suffix)]
            # Check if base name is in mask
            base_filename = f"{base_name}{ext}"
            if base_filename in mask_set:
                filtered_csvs.append(csv_path)
        csvs = filtered_csvs
    
    to_link = [p for p in csvs if os.path.basename(p) not in cache]   ########
    if not to_link:
        print(f"{source_folder}: no new files to link.")
        return

    print(f"{source_folder}: linking {len(to_link)} new CSVs...")
    results = Parallel(n_jobs=4, backend='threading')(        ########
        delayed(create_symlink)(path, target_dir, cache, file_suffix)
        for path in to_link
    )
    processed = sum(results)
    skipped = len(to_link) - processed
    print(f"{source_folder}: total_new={len(to_link)}, linked={processed}, skipped={skipped}")
    for path in to_link:
        cache.add(os.path.basename(path))


def main():
    parser = argparse.ArgumentParser(description="Incremental CSV symlinker with mode-based suffixes.")
    parser.add_argument("--repo_root", type=str, default="/u4/z6dong/Repo", help="Repository root path.")
    parser.add_argument("--mode", type=str, choices=list(MODE_SUFFIX.keys())+["all"], default="base",
                        help="Mode for folder and file suffix. Use 'all' to run every mode.")
    parser.add_argument("--dir-name", type=str, default=None,
                        help="Override target directory name (default: scilake_final{suffix}). E.g., scilake_final_v2")
    parser.add_argument("--tag", type=str, default=None,
                        help="Tag suffix for versioning (e.g., 251117). If provided, uses tagged folders like deduped_hugging_csvs_v2_<tag>")
    parser.add_argument("--mask-file", type=str, default=None,
                        help="Path to mask file (e.g., data/analysis/all_valid_title_valid_<tag>.txt). If provided, only links files listed in this file.")
    parser.add_argument("--no-mask", action="store_true", default=False,
                        help="Disable mask file filtering even if mask file exists. Links all files from source folders.")
    args = parser.parse_args()

    modes = [args.mode] if args.mode != "all" else list(MODE_SUFFIX.keys())

    # Determine tag suffix for source folders
    tag_suffix = f"_{args.tag}" if args.tag else ""
    
    # Load mask file if provided (unless --no-mask is specified)
    mask_set = None
    if args.no_mask:
        print("Mask file filtering disabled (--no-mask). Will link all files from source folders.")
    elif args.mask_file:
        mask_file_path = args.mask_file
        # If relative path, make it relative to CitationLake root
        if not os.path.isabs(mask_file_path):
            mask_file_path = os.path.join(args.repo_root, "CitationLake", mask_file_path)
        mask_set = load_mask_file(mask_file_path)
        if mask_set:
            print(f"Loaded mask file: {mask_file_path} ({len(mask_set)} allowed files)")
        else:
            print(f"Warning: Mask file {mask_file_path} is empty or invalid")
    elif args.tag:
        # Auto-detect mask file based on tag
        mask_file_path = os.path.join(args.repo_root, "CitationLake", "data", "analysis", f"all_valid_title_valid_{args.tag}.txt")
        if os.path.exists(mask_file_path):
            mask_set = load_mask_file(mask_file_path)
            if mask_set:
                print(f"Auto-loaded mask file: {mask_file_path} ({len(mask_set)} allowed files)")

    for mode in modes:
        dir_suffix, file_suffix = MODE_SUFFIX[mode]
        # Use v2 source directories for base data; augmented variants append suffixes to v2 dirs
        base_dirs = {
            "hugging": "deduped_hugging_csvs_v2",
            "github":  "deduped_github_csvs_v2",
            "html":    "tables_output_v2",
            "llm":     "llm_tables"
        }
        # Apply tag_suffix and dir_suffix to create augmented folder names
        # Note: llm_tables only gets tag_suffix if tag is provided, otherwise stays as "llm_tables"
        src_folders = [
            os.path.join(args.repo_root, "CitationLake", "data", "processed", f"{base_dirs['hugging']}{tag_suffix}{dir_suffix}"),
            os.path.join(args.repo_root, "CitationLake", "data", "processed", f"{base_dirs['github']}{tag_suffix}{dir_suffix}"),
            os.path.join(args.repo_root, "CitationLake", "data", "processed", f"{base_dirs['html']}{tag_suffix}{dir_suffix}"),
            os.path.join(args.repo_root, "CitationLake", "data", "processed", f"{base_dirs['llm']}{tag_suffix}{dir_suffix}" if args.tag else f"{base_dirs['llm']}{dir_suffix}")
        ]
        # Default dir_name: include tag if provided, otherwise just use suffix
        if args.dir_name:
            dir_name = args.dir_name
        elif args.tag:
            dir_name = f"scilake_final_{args.tag}{dir_suffix}"
        else:
            dir_name = f"scilake_final{dir_suffix}"
        target_dir = os.path.join(
            args.repo_root,
            "starmie_internal", "data",
            dir_name,
            "datalake"
        )  ######## build correct target directory

        print(f"\nMode={mode}, target_dir={target_dir}, file_suffix={file_suffix}")
        if args.tag:
            print(f"Using tag: {args.tag}")
        print(f"Source folders to scan:")
        for src in src_folders:
            exists = os.path.isdir(src)
            status = "✓ EXISTS" if exists else "✗ MISSING"
            print(f"  {status}: {src}")
        
        os.makedirs(target_dir, exist_ok=True)
        cache = {f for f in os.listdir(target_dir) if f.lower().endswith('.csv')}

        for src in src_folders:
            if not os.path.isdir(src):
                print(f"Skip missing source folder {src}")
                continue
            process_folder(src, target_dir, cache, file_suffix, mask_set)

        total = len([f for f in os.listdir(target_dir) if f.lower().endswith('.csv')])
        print(f"Done mode={mode}: total CSV in {target_dir} = {total}\n")

if __name__ == "__main__":
    main()
