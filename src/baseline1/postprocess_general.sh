#!/bin/bash

# ori+tr
python src/baseline1/postprocess_general.py \
  --input data/baseline/table_neighbors_ori_tr.json \
  --key_types "" \
  --value_types "" _t \
  --output data/baseline/table_neighbors_ori_tr_key_ori.json

python src/baseline1/postprocess_general.py \
  --input data/baseline/table_neighbors_ori_tr.json \
  --key_types _t \
  --value_types "" _t \
  --output data/baseline/table_neighbors_ori_tr_key_tr.json

#python analyze_table_neighbors.py --input data/baseline/table_neighbors_ori_tr_key_ori.json
# data/baseline/table_neighbors_ori_tr_key_tr.json

# ori+str
python src/baseline1/postprocess_general.py \
  --input data/baseline/table_neighbors_ori_str.json \
  --key_types "" \
  --value_types "" _s \
  --output data/baseline/table_neighbors_ori_str_key_ori.json

python src/baseline1/postprocess_general.py \
  --input data/baseline/table_neighbors_ori_str.json \
  --key_types _s \
  --value_types "" _s \
  --output data/baseline/table_neighbors_ori_str_key_str.json

# ori+tr+str
python src/baseline1/postprocess_general.py \
  --input data/baseline/table_neighbors_mixed.json \
  --key_types "" \
  --value_types "" _s _t \
  --output data/baseline/table_neighbors_mixed_key_ori.json

python src/baseline1/postprocess_general.py \
  --input data/baseline/table_neighbors_mixed.json \
  --key_types _s \
  --value_types "" _s _t \
  --output data/baseline/table_neighbors_mixed_key_str.json

python src/baseline1/postprocess_general.py \
  --input data/baseline/table_neighbors_mixed.json \
  --key_types _t \
  --value_types "" _s _t \
  --output data/baseline/table_neighbors_mixed_key_tr.json
