### Non-main pipeline scripts (optional / deprecated)

This page collects scripts that are outside the main pipeline, including optional alternatives, patch/recovery utilities, and deprecated flows.  
For the streamlined main workflow, see `scripts.md`.

<details>
#### Option1:
# Query Semantic Scholar API for citation information (alternative to local database if no key, but may hit rate limits). Input: modelcard_dedup_titles_<tag>.json (from step2_s2orc_save) Output: s2orc_query_results_<tag>.parquet, s2orc_citations_cache_<tag>.parquet, s2orc_references_cache_<tag>.parquet, s2orc_titles2ids_<tag>.parquet
# Why API over local (build_mini_citation_es): (1) API provides fresher citations/references; (2) API's title fuzzy matching is more accurate (commercialized) than our local ES fuzzy match.

# copy from searched results, save some searched results, only search the missing titles
#cp -r data/processed/s2orc_titles2ids.parquet data/processed/s2orc_titles2ids_251117.parquet
#cp -r data/processed/s2orc_citations_cache.parquet data/processed/s2orc_citations_cache_251117.parquet
#cp -r data/processed/s2orc_references_cache.parquet data/processed/s2orc_references_cache_251117.parquet

python -m src.data_preprocess.s2or_title2ids_API --tag 251117 > logs/s2orc_title2ids_API_251117.log 2>&1
python -m src.data_preprocess.s2orc_refcit_API --tag 251117 > logs/s2orc_refcit_API_251117.log 2>&1
 # Optional: Local exact title:id batch (supplement API results, then manually concat). I: s2orc_titles2ids_<tag>.parquet O: s2orc_titles2ids_local_<tag>.parquet. Requires papers_index (build_mini_s2orc_es --mode build). Uses same ES setup as build_mini_citation_es.sh.
 #- bash src/data_localindexing/local_exact_title2id.sh 251117 > logs/local_exact_title2id_251117.log 2>&1

 # (Patches)
 #- PYTHONPATH=. python bak/s2orc_log_parser --tag 251117 --logdir logs # extract from s2orc_API_query*.log → s2orc_titles2ids_251117_5.parquet
 #- PYTHONPATH=. python bak/merge_s2orc_titles.py --file1 data/processed/s2orc_titles2ids_251117.parquet --file2 data/processed/s2orc_titles2ids_251117_2.parquet --output data/processed/s2orc_titles2ids_251117_3.parquet
 #- PYTHONPATH=. python bak/filter_s2orc_titles_by_dedup.py --tag 251117  # I: _3, dedup_titles | O: _4 (filter by dedup, success first)
 # mv data/processed/s2orc_titles2ids_251117_4.parquet data/processed/s2orc_titles2ids_251117.parquet
 # (Patches for 429 rate limit error)
 #- python -m src.data_preprocess.s2orc_log_429 --tag 251117 --logfile logs/s2orc_API_query_251117.log --error 429 > logs/s2orc_log_429_251117.log 2>&1 # if 429 errors, extract failed titles to modelcard_dedup_titles_251117_429.json
 #- python -m src.data_preprocess.s2orc_retry_missing --tag 251117 > logs/s2orc_retry_missing_251117.log 2>&1 # make up for the missing items (use after s2orc_log_429 if needed)
 python -m src.data_preprocess.s2orc_merge --tag 251117 > logs/s2orc_merge_251117.log 2>&1 # parse refs/cits | I: s2orc_*_251117.parquet, O: s2orc_rerun_251117.parquet. 
 #- bash src/data_localindexing/build_mini_citation_es.sh > logs/build_mini_citation_es.log 2>&1 # I: xx | O: batch_results
# Extract full records from batch query results. Input: batch_results + hit_ids_<tag>.txt, output: full_hits_<tag>.jsonl
python -m src.data_localindexing.s2orc_refcit_local --tag 251117 --src_dir /u501/z6dong/shared_data/se_citations_250218 > logs/extract_full_records.log 2>&1
# Merge extracted full records. Input: full_hits_<tag>.jsonl (or fallback full_hits.jsonl), Output: s2orc_*_<tag>.parquet
python -m src.data_localindexing.s2orc_refcit_local_post --tag 251117 > logs/s2orc_local_query_ref_cit_251117.log 2>&1
- python -m src.data_preprocess.s2orc_merge --tag 251117 > logs/s2orc_merge_251117.log 2>&1 # I: s2orc_*_251117.parquet, O: s2orc_rerun_251117.parquet Add --add-missing if you ran s2orc_retry_missing
 # (deprecate) - bash src/data_localindexing/build_mini_s2orc_es.sh # choose dump data to setup and batch query | I: paper_index_mini.db, modelcard_dedup_titles.json → O: Elasticsearch index (e.g., papers_index), query_cache.parquet
 - bash src/data_preprocess/step2_se_url_tab.sh # extract fulltext -> ref/cit info
# I: query_cache.parquet/s2orc_rerun.parquet, paper_index_mini.db, NDJSON files in /se_s2orc_250218 → O: extracted_annotations.parquet, tmp_merged_df.parquet, tmp_extracted_lines.parquet
### Option2: batch querying papers_index
python -m src.data_localindexing.build_mini_s2orc_es --mode batch_query --directory /u501/z6dong/shared_data/se_s2orc_250218 --index_name papers_index --titles_file data/processed/modelcard_dedup_titles_251117.json --cache_file data/processed/query_cache_251117.json # getting full tables
</details>



(Deprecated scripts: We previously downloaded PDFs and tried to parse them, but the Semantic Scholar dataset already covers this need.)
```bash
# deprecated as we don't use PDF for extraction at this time
#python -m bak.step2_get_pdf #TODO: wait se_url_tab and then test
python -m bak.step_down_pdf
python -m bak.step_add_pdftab # Issue: deterministic PDF2table is not accurate enough. Try LLM based image extraction (not implemented here)
python -m bak.step_down_tex # Issue: IP rate limit on accessing tex files, Possible solution: use arxiv bulk downloading
python -m bak.step_add_textab
python -m bak.step_add_gittab
python -m bak.tmp_extract_url # Update PDF url from extracted url (some don't have .pdf, need to extract from html or add)
python -m bak.tmp_extract_table # Extract table/figures caption and cited text from s2orc dumped data, but don't contain text and figure detailed content!
python -m bak.step4 # process groundtruth (work for API, not work for dump data)
python -m bak.step2_Citation_Info
# (Optional) python -m bak.step3_statistic_table # get statistic tables
python -m bak.step1_parsetags # Parse tags into columns with name start with `card_tag_xx`
bash bak/symlink_trick_str.sh # too slow
bash bak/symlink_trick_tr.sh # too slow
bash bak/symlink_trick_tr_str.sh # too slow
bash bak/symlink_ln_scilake_large.sh # too slow
bash bak/symlink_ln_scilake_final.sh
```
