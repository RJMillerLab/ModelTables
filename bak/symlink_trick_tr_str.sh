#!/bin/bash

#repo_root="/u4/z6dong/Repo"
repo_root="/Users/doradong/Repo"
folders=(
    "CitationLake/data/processed/deduped_hugging_csvs"
    "CitationLake/data/processed/deduped_github_csvs"
    "CitationLake/data/processed/tables_output"
    "CitationLake/data/processed/llm_tables"
)

for rel_folder in "${folders[@]}"; do
    folder="$repo_root/$rel_folder"
    out_folder="${folder}_str_transpose"

    mkdir -p "$out_folder"

    echo "Processing folder: $folder -> $out_folder"

    find "$folder" -type f -name "*.csv" | while read -r csv; do
        filename="$(basename "$csv")"
        base_no_ext="${filename%.csv}"
        out_csv="${out_folder}/${base_no_ext}_s_t.csv"

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

header = reader[0]
data_rows = reader[1:]

# Pad all rows to header length ########
normalized_rows = [row + [""] * (len(header) - len(row)) for row in data_rows] ########

# Perform str conversion: colname-value ########
processed_data = []
for row in normalized_rows:
    new_row = []
    for col_idx, cell_value in enumerate(row):
        col_name = header[col_idx] if col_idx < len(header) else f"col{col_idx}"
        new_row.append(f"{col_name}-{cell_value}")
    processed_data.append(new_row)

# Transpose after str-processing ########
out_rows = []
for i in range(len(header)):
    new_row = [header[i]]
    for row in processed_data:
        new_row.append(row[i] if i < len(row) else "")
    out_rows.append(new_row)

with open(csv_out, "w", newline="", encoding="utf-8") as f_out:
    writer = csv.writer(f_out)
    writer.writerows(out_rows)
' "$csv" "$out_csv"

        echo "Str-processed (transpose): $csv -> $out_csv"
    done
done
