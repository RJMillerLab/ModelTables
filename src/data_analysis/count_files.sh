#!/bin/bash

# This script counts the total number of files in each of the following directories:
# 1. data/processed/deduped_github_csvs
# 2. data/processed/deduped_hugging_csvs
# 3. data/processed/tables_output
# 4. data/processed/llm_tables
#
# Usage:
#   Save this script as count_specific_files.sh
#   Make it executable: chmod +x count_specific_files.sh
#   Run it: ./count_specific_files.sh

dirs=(
    "data/processed/deduped_github_csvs"
    "data/processed/deduped_hugging_csvs"
    "data/processed/tables_output"
    "data/processed/llm_tables"
)

for dir in "${dirs[@]}"; do
    if [ -d "$dir" ]; then
        # Count files recursively in the directory
        file_count=$(find "$dir" -type f | wc -l)
        echo "Total number of files in '$dir': $file_count"
    else
        echo "Directory '$dir' does not exist."
    fi
done
