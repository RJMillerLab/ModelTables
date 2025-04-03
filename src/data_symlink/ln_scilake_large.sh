#!/bin/bash

# Define repo root and target directory
repo_root="/u4/z6dong/Repo" ######################################
target_dir="$repo_root/starmie_internal/data/scilake_large/datalake"

# List of subfolders relative to the repo root
folders=(
    "CitationLake/data/processed/deduped_hugging_csvs"
    "CitationLake/data/processed/deduped_github_csvs"
    "CitationLake/tables_output"
    "CitationLake/llm_tables"
)

for rel_folder in "${folders[@]}"; do
    folder="$repo_root/$rel_folder"
    echo "Scanning: $folder"
    
    csv_count=$(find "$folder" -maxdepth 1 -iname '*.csv' | wc -l) ########
    echo "Folder: $folder, CSV count: $csv_count"

    find "$folder" -maxdepth 1 -iname '*.csv' | while read -r csv; do ########
        ln -sf "$csv" "$target_dir" ########
    done
done

total_count=$(find "$target_dir" -maxdepth 1 -iname '*.csv' | wc -l) ########
echo "Total CSV count in datalake folder: $total_count"
