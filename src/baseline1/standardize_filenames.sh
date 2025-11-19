#!/bin/bash
# Baseline1: Standardize filenames in search results
# Supports TAG environment variable for versioning (e.g., TAG=251117)

set -e

TAG="${TAG:-}"
SUFFIX="${TAG:+_${TAG}}"

json_files=(
  data/baseline/table_neighbors_ori_tr${SUFFIX}_key_ori.json
  data/baseline/table_neighbors_ori_tr${SUFFIX}_key_tr.json
  data/baseline/table_neighbors_ori_str${SUFFIX}_key_ori.json
  data/baseline/table_neighbors_ori_str${SUFFIX}_key_str.json
  data/baseline/table_neighbors_mixed${SUFFIX}_key_ori.json
  data/baseline/table_neighbors_mixed${SUFFIX}_key_str.json
  data/baseline/table_neighbors_mixed${SUFFIX}_key_tr.json
)

for json in "${json_files[@]}"; do
  if [ -f "$json" ]; then
    echo "Processing $json"
    python3 src/baseline1/standardize_filenames.py --input "$json"
  else
    echo "Warning: $json not found, skip."
  fi
done 