#!/bin/bash

#repo_root="/u4/z6dong/Repo"
repo_root="/Users/doradong/Repo"
folders=(
    "CitationLake/data/processed/deduped_hugging_csvs"
    "CitationLake/data/processed/cleaned_markdown_csvs_github"
    "CitationLake/tables_output"
    #"CitationLake/llm_tables"
)

for rel_folder in "${folders[@]}"; do
    folder="$repo_root/$rel_folder"
    out_folder="${folder}_transpose"

    mkdir -p "$out_folder"

    echo "Processing folder: $folder -> $out_folder"

    find "$folder" -type f -name "*.csv" | while read -r csv; do
        filename="$(basename "$csv")"
        base_no_ext="${filename%.csv}"
        out_csv="${out_folder}/${base_no_ext}_t.csv"

        python -c '
import sys
import csv

csv_in = sys.argv[1]
csv_out = sys.argv[2]

with open(csv_in, newline="", encoding="utf-8") as f_in:
    reader = list(csv.reader(f_in))

if not reader:
    open(csv_out, "w").close()
    sys.exit(0)

# Move header into first row of data ########
header = reader[0] ########
data_rows = reader[1:] ########
data_with_header = [header] + data_rows ########

# Normalize lengths ########
max_len = max(len(row) for row in data_with_header)
normalized = [row + [""] * (max_len - len(row)) for row in data_with_header] ########

transpose = list(zip(*normalized)) ########

with open(csv_out, "w", newline="", encoding="utf-8") as f_out:
    writer = csv.writer(f_out)
    for row in transpose:
        writer.writerow(row)
' "$csv" "$out_csv"

        echo "Transposed: $csv -> $out_csv"
    done
done
