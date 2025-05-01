#!/usr/bin/env python3

import os
import argparse
import random

def collect_files_from_dir(directory, limit, seed=None):
    files = []
    for fname in os.listdir(directory):
        full_path = os.path.join(directory, fname)
        if os.path.isfile(full_path):  # accept both symlinks and regular files
            files.append(fname)
    if seed is not None:
        random.seed(seed)
    return random.sample(files, min(limit, len(files)))

def main():
    parser = argparse.ArgumentParser(description="Randomly sample files from multiple subdirectories.")
    parser.add_argument('--root_dir', type=str, required=True, help='Root directory, e.g., /u4/z6dong/Repo')
    parser.add_argument('--output_file', type=str, default='file_list.txt', help='Output file path')
    parser.add_argument('--limit', type=int, default=1000, help='Max files per subdir')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')

    args = parser.parse_args()

    # Subdirectories to scan
    subdirs = [
        "CitationLake/data/processed/llm_tables",
        "CitationLake/data/processed/deduped_github_csvs",
        "CitationLake/data/processed/deduped_hugging_csvs",
        "CitationLake/data/processed/tables_output"
    ]

    #with open(args.output_file, 'w') as out_f:
    # Derive validation output filename 
    val_output_file = args.output_file.replace('.txt', '_val.txt') 
    with open(args.output_file, 'w') as train_f, open(val_output_file, 'w') as val_f: 
        for subdir in subdirs:
            abs_path = os.path.join(args.root_dir, subdir)
            if not os.path.exists(abs_path):
                print(f"Warning: {abs_path} does not exist. Skipping.")
                continue
            #sampled_files = collect_files_from_dir(abs_path, args.limit, args.seed)
            #for fname in sampled_files:
                #out_f.write(os.path.join(subdir, fname) + '\n')
                #out_f.write(os.path.join(args.root_dir, subdir, fname) + '\n')  
            #    out_f.write(fname + '\n')
            # Sample twice the limit and split into train/val 
            sampled_files = collect_files_from_dir(abs_path, args.limit * 2, args.seed)
            train_samples = sampled_files[:args.limit]
            val_samples = sampled_files[args.limit:]
            for fname in train_samples:
                train_f.write(fname + '\n')
            for fname in val_samples:
                val_f.write(fname + '\n')

    print(f"Generated {args.output_file}")

if __name__ == '__main__':
    main()

