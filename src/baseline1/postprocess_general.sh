#!/bin/bash
# Baseline1: Postprocess search results
# Supports TAG environment variable for versioning (e.g., TAG=251117)

TAG="${TAG:-}"
SUFFIX="${TAG:+_${TAG}}"

# ori+tr
python src/baseline1/postprocess_general.py \
  --input data/baseline/table_neighbors_ori_tr${SUFFIX}.json \
  --key_types "" \
  --value_types "" _t \
  --output data/baseline/table_neighbors_ori_tr${SUFFIX}_key_ori.json

python src/baseline1/postprocess_general.py \
  --input data/baseline/table_neighbors_ori_tr${SUFFIX}.json \
  --key_types _t \
  --value_types "" _t \
  --output data/baseline/table_neighbors_ori_tr${SUFFIX}_key_tr.json

#python analyze_table_neighbors.py --input data/baseline/table_neighbors_ori_tr${SUFFIX}_key_ori.json
# data/baseline/table_neighbors_ori_tr${SUFFIX}_key_tr.json

# ori+str
python src/baseline1/postprocess_general.py \
  --input data/baseline/table_neighbors_ori_str${SUFFIX}.json \
  --key_types "" \
  --value_types "" _s \
  --output data/baseline/table_neighbors_ori_str${SUFFIX}_key_ori.json

python src/baseline1/postprocess_general.py \
  --input data/baseline/table_neighbors_ori_str${SUFFIX}.json \
  --key_types _s \
  --value_types "" _s \
  --output data/baseline/table_neighbors_ori_str${SUFFIX}_key_str.json

# ori+tr+str
python src/baseline1/postprocess_general.py \
  --input data/baseline/table_neighbors_mixed${SUFFIX}.json \
  --key_types "" \
  --value_types "" _s _t \
  --output data/baseline/table_neighbors_mixed${SUFFIX}_key_ori.json

python src/baseline1/postprocess_general.py \
  --input data/baseline/table_neighbors_mixed${SUFFIX}.json \
  --key_types _s \
  --value_types "" _s _t \
  --output data/baseline/table_neighbors_mixed${SUFFIX}_key_str.json

python src/baseline1/postprocess_general.py \
  --input data/baseline/table_neighbors_mixed${SUFFIX}.json \
  --key_types _t \
  --value_types "" _s _t \
  --output data/baseline/table_neighbors_mixed${SUFFIX}_key_tr.json
