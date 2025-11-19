#!/bin/bash
# Baseline2: Get metadata for sparse search
# Supports TAG environment variable for versioning (e.g., TAG=251117)
# Usage: TAG=251117 bash src/baseline2/get_metadata.sh

# 1. generate mapping from csv_path:readme_path
python src/baseline2/create_raw_csv_to_text_mapping.py
python src/baseline2/view_mapping.py
# 2. get incontext embedding for each csv_path, save to data/tmp/corpus/collection.jsonl
python src/baseline2/create_dedup_table_to_text_mapping.py
