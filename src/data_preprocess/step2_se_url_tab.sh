#!/bin/bash
python step2_se_url_tab.py \
--directory /u4/z6dong/shared_data/se_s2orc_250218 \
--db_path /u4/z6dong/shared_data/se_s2orc_250218/paper_index_mini.db \
--parquet_cache query_cache.parquet \
--output_parquet extracted_annotations.parquet \
--n_jobs -1

