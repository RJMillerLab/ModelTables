#!/bin/bash
set -e

json_files=(
  data/baseline/table_neighbors_ori_tr_key_ori.json
  data/baseline/table_neighbors_ori_tr_key_tr.json
  data/baseline/table_neighbors_ori_str_key_ori.json
  data/baseline/table_neighbors_ori_str_key_str.json
  data/baseline/table_neighbors_mixed_key_ori.json
  data/baseline/table_neighbors_mixed_key_str.json
  data/baseline/table_neighbors_mixed_key_tr.json
)

for json in "${json_files[@]}"; do
  if [ -f "$json" ]; then
    echo "Processing $json"
    python3 src/baseline1/standardize_filenames.py --input "$json"
  else
    echo "Warning: $json not found, skip."
  fi
done 