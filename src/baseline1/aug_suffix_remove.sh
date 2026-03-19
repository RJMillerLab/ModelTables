#!/bin/bash
# Baseline1: Standardize filenames in search results
# Supports TAG environment variable for versioning (e.g., TAG=251117)

set -e

TAG="${TAG:-}"
V2_MODE="${V2_MODE:-false}"

# Directory suffix: baseline1{TAG_SUFFIX}
if [[ -n "${TAG}" ]]; then
  TAG_SUFFIX="_${TAG}"
else
  TAG_SUFFIX=""
fi

# Filename suffix: {V2_SUFFIX}{TAG_SUFFIX}
if [[ "${V2_MODE}" == "true" ]]; then
  V2_SUFFIX="_v2"
else
  V2_SUFFIX=""
fi
FILE_SUFFIX="${V2_SUFFIX}${TAG_SUFFIX}"

json_files=(
  data/baseline1${TAG_SUFFIX}/table_neighbors_ori_tr${FILE_SUFFIX}_key_ori.json
  data/baseline1${TAG_SUFFIX}/table_neighbors_ori_tr${FILE_SUFFIX}_key_tr.json
  data/baseline1${TAG_SUFFIX}/table_neighbors_ori_str${FILE_SUFFIX}_key_ori.json
  data/baseline1${TAG_SUFFIX}/table_neighbors_ori_str${FILE_SUFFIX}_key_str.json
  data/baseline1${TAG_SUFFIX}/table_neighbors_mixed${FILE_SUFFIX}_key_ori.json
  data/baseline1${TAG_SUFFIX}/table_neighbors_mixed${FILE_SUFFIX}_key_str.json
  data/baseline1${TAG_SUFFIX}/table_neighbors_mixed${FILE_SUFFIX}_key_tr.json
)

for json in "${json_files[@]}"; do
  if [ -f "$json" ]; then
    echo "Processing $json"
    python3 -m src.baseline1.aug_suffix_remove --input "$json"
  else
    echo "Warning: $json not found, skip."
  fi
done 