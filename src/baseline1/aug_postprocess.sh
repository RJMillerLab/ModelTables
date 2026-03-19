#!/bin/bash
# Baseline1: Postprocess search results
# Supports TAG environment variable for versioning (e.g., TAG=251117)

V2_MODE="${V2_MODE:-false}"
TAG="${TAG:-}"

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

# ori+tr
python -m src.baseline1.aug_postprocess \
  --input data/baseline1${TAG_SUFFIX}/table_neighbors_ori_tr${FILE_SUFFIX}.json \
  --key_types "" \
  --value_types "" _t \
  --output data/baseline1${TAG_SUFFIX}/table_neighbors_ori_tr${FILE_SUFFIX}_key_ori.json

python -m src.baseline1.aug_postprocess \
  --input data/baseline1${TAG_SUFFIX}/table_neighbors_ori_tr${FILE_SUFFIX}.json \
  --key_types _t \
  --value_types "" _t \
  --output data/baseline1${TAG_SUFFIX}/table_neighbors_ori_tr${FILE_SUFFIX}_key_tr.json

#python analyze_table_neighbors.py --input data/baseline/table_neighbors_ori_tr${SUFFIX}_key_ori.json
# data/baseline/table_neighbors_ori_tr${SUFFIX}_key_tr.json

# ori+str
python -m src.baseline1.aug_postprocess \
  --input data/baseline1${TAG_SUFFIX}/table_neighbors_ori_str${FILE_SUFFIX}.json \
  --key_types "" \
  --value_types "" _s \
  --output data/baseline1${TAG_SUFFIX}/table_neighbors_ori_str${FILE_SUFFIX}_key_ori.json

python -m src.baseline1.aug_postprocess \
  --input data/baseline1${TAG_SUFFIX}/table_neighbors_ori_str${FILE_SUFFIX}.json \
  --key_types _s \
  --value_types "" _s \
  --output data/baseline1${TAG_SUFFIX}/table_neighbors_ori_str${FILE_SUFFIX}_key_str.json

# ori+tr+str
python -m src.baseline1.aug_postprocess \
  --input data/baseline1${TAG_SUFFIX}/table_neighbors_mixed${FILE_SUFFIX}.json \
  --key_types "" \
  --value_types "" _s _t \
  --output data/baseline1${TAG_SUFFIX}/table_neighbors_mixed${FILE_SUFFIX}_key_ori.json

python -m src.baseline1.aug_postprocess \
  --input data/baseline1${TAG_SUFFIX}/table_neighbors_mixed${FILE_SUFFIX}.json \
  --key_types _s \
  --value_types "" _s _t \
  --output data/baseline1${TAG_SUFFIX}/table_neighbors_mixed${FILE_SUFFIX}_key_str.json

python -m src.baseline1.aug_postprocess \
  --input data/baseline1${TAG_SUFFIX}/table_neighbors_mixed${FILE_SUFFIX}.json \
  --key_types _t \
  --value_types "" _s _t \
  --output data/baseline1${TAG_SUFFIX}/table_neighbors_mixed${FILE_SUFFIX}_key_tr.json
