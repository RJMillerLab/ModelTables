## Scripts

This section outlines the workflow for processing data, building the ground truth, and running evaluations.

---

## Data Processing Workflow

### 0\. Download Latest Hugging Face Snapshot

Use `download_hf_dataset.py` to pull the newest `librarian-bots/model_cards_with_metadata` or `librarian-bots/dataset_cards_with_metadata` parquet shards into a date-tagged folder (for example, `data/raw_251117`). The script automatically enumerates the parquet shards available on Hugging Face Hub and stores them locally so downstream steps can point to a specific snapshot when re-running the pipeline.

**Important**: 
- Model cards and dataset cards are stored in the same `data/raw_<date>` directory but with different filename prefixes to avoid conflicts:
  - **Modelcard files**: `train-*.parquet` (no prefix, original format)
  - **Datasetcard files**: `datasetcard-train-*.parquet` (with prefix to avoid conflict)
- Both can use the same tag (e.g., `251117`) to keep them synchronized
- The old `data/raw` directory (without date tag) maintains the original format for backward compatibility

```bash
mkdir logs
# Download modelcards
python -m src.data_preprocess.download_hf_dataset --date 251117 --type modelcard/datasetcard > logs/download_hf_dataset_251117.log 2>&1
```

### 1\. Parse Initial Elements

This step extracts key metadata from model cards and associated links.
```bash
# Split readme and tags, parse URLs, parse BibTeX entries.
# Output: ['modelId', 'author', 'last_modified', 'downloads', 'likes', 'library_name', 'tags', 'pipeline_tag', 'createdAt', 'card', 'card_tags', 'card_readme', 'pdf_link', 'github_link', 'all_links', 'extracted_bibtex', 'extracted_bibtex_tuple', 'parsed_bibtex_tuple_list', 'successful_parse_count']
# (Optional)
#python -m src.data_preprocess.load_raw_to_db sqlite/duckdb > logs/load_raw_to_db.log 2>&1 # save raw to DuckDB, but will explode the memory
python -m src.data_preprocess.step1_parse --raw-date 251117 --versioning --baseline-step1 data/processed/modelcard_step1.parquet > logs/step1_parse_251117.log 2>&1
# or 
python -m src.data_preprocess.step1_parse --raw-date 251117 > logs/step1_parse_251117.log 2>&1
python -m src.data_preprocess.step1_down_giturl --tag 251117 --versioning --baseline-cache data/processed/github_readme_cache.parquet > logs/step1_down_giturl_251117.log 2>&1 # Download GitHub READMEs and HTMLs from extracted URLs; Input: modelcard_step1.parquet, Output: giturl_info.parquet, downloaded_github_readmes/
#python -m src.data_preprocess.step1_down_giturl_fake > logs/step1_down_giturl_fake.log 2>&1 # if program has leakage but finished downloading, then re-run this code to save final parquet and cache files.
find data/downloaded_github_readmes -type f -exec stat -f "%z %N" {} + | sort -nr | head -n 50 | awk '{printf "%.2f MB %s\n", $1/1024/1024, $2}' > logs/find_large_readmes.log 2>&1 # some readme files are too large
# Query specific GitHub URL content (example). Input: local path to a downloaded README, Output: URL content
python -m src.data_analysis.query_giturl load --query "data/downloaded_github_readmes/0a0c3d247213c087eb2472c3fe387292.md" > logs/query_giturl.log 2>&1 # sql. # (Optional)
```

### 2\. Download and Build Database for Faster Querying

This step sets up local databases for efficient querying of Semantic Scholar data.

I don't update this section anymore, as the semantic scholar dataset is too large to maintain.

<details>
<summary>Click to expand database setup commands</summary>

```bash
# TODO: add command from privatecommonscript to here, for downloading the semantic scholar here
# Requirement: cd to the path of downloaded dataset, e.g.: cd ~/shared_data/se_s2orc_250218
python -m src.data_localindexing.build_mini_s2orc build --directory /u4/z6dong/shared_data/se_s2orc_250218/ # After downloading semantic scholar dataset, build database based on it.
python -m src.data_localindexing.build_mini_s2orc query --title "BioMANIA: Simplifying bioinformatics data analysis through conversation" --directory /u4/z6dong/shared_data/se_s2orc_250218/ # After building up database, query title based on db file.
python -m src.data_localindexing.build_mini_s2orc query_cid --corpusid 248779963 --directory /u4/z6dong/shared_data/se_s2orc_250218

# issue: citation edge is hard to store, it is too much ... Solution: I think we better using the API to query citation relationship? Or use cypher to query over graph condensely
# python -m src.data_localindexing.build_complete_citation build --directory ./ # build db for citation dataset
# python -m src.data_localindexing.build_complete_citation query --citationid 169 --directory ./
# (Optional) if you don't have key, use public API for querying citations instead
# python -m src.data_preprocess.step1_citationAPI # get citations through bibtex only by API. TODO: Update for bibtex + url, not bibtex only. TODO: Update for all bibtex, not the first bibtex

# Optional solution: we use kuzu database to store node and edge
#python -m src.data_localindexing.build_mini_citation_kuzu --mode build --directory /u4/z6dong/shared_data/se_citations_250218/
#python -m src.data_localindexing.test_node_edge_db # test how many nodes and edges are in built database
# issue: slow for our 300G ndjson files, not suitable for this stage

# Optional solution: we use neo4j database to store and query
#python src.data_localindexing.build_mini_citation_neo4j --mode build --directory ./ --fields minimal
#python src.data_localindexing.build_mini_citation_neo4j --mode query --citationid 248811336
# for slurm run this script to keep neo4j open in another terminal
# sbatch src.data_localindexing.neo4j_slurm

# fuzzy matching: elastic search for s2orc
python -m src.data_localindexing.build_mini_s2orc_es --mode build --directory /u4/z6dong/shared_data/se_s2orc_250218 --index_name papers_index --db_file /u4/z6dong/shared_data/se_s2orc_250218/paper_index_mini.db
python -m src.data_localindexing.build_mini_s2orc_es --mode query --directory /u4/z6dong/shared_data/se_s2orc_250218 --index_name papers_index --query "BioMANIA: Simplifying bioinformatics data analysis through conversation"
python -m src.data_localindexing.build_mini_s2orc_es --mode test --directory /u4/z6dong/shared_data/se_s2orc_250218 --index_name papers_index --db_file /u4/z6dong/shared_data/se_s2orc_250218/paper_index_mini.db
# batch querying papers_index
python build_mini_s2orc_es.py --mode batch_query --directory /u4/z6dong/shared_data/se_s2orc_250218 --index_name papers_index --titles_file data/processed/modelcard_dedup_titles.json --cache_file data/processed/query_cache.json
# getting full tables
```
</details>

### 3\. Extract Tables to Local Folder

This step extracts tabular data from various sources and processes it.
```bash
# Extract tables from Hugging Face Model Cards and GitHub READMEs. Saves CSVs to local folder.
# Versioning mode (with tag):
# Input: data/processed/modelcard_step1_<tag>.parquet, github_readmes_info_<tag>.parquet, downloaded_github_readmes_<tag>/
# Output: data/processed/modelcard_step2_v2_<tag>.parquet, data/processed/deduped_hugging_csvs_v2_<tag>/, data/processed/hugging_deduped_mapping_v2_<tag>.json, data/processed/deduped_github_csvs_v2_<tag>/, data/processed/deduped_github_csvs_v2_<tag>/md_to_csv_mapping.json
python -m src.data_preprocess.step2_hugging_github_extract --tag 251117 > logs/step2_hugging_github_extract_251117.log 2>&1

# Process downloaded GitHub HTML files to Markdown.
# Input: data/downloaded_github_readmes_<tag>/
# Output: data/downloaded_github_readmes_<tag>_processed/, data/processed/md_parsing_results_v2_<tag>.parquet
python -m src.data_preprocess.step2_git_md2text --tag 251117 > logs/step2_git_md2text_251117.log 2>&1
#python -m src.data_preprocess.step2_git_md2text_v2 --n_jobs 8 --output_dir data/processed/md_processed_v2 --save_mode csv/duckdb/sqlite > logs/step2_git_md2text_v2.log 2>&1
# Extract titles from arXiv and GitHub URLs (not S2ORC). For BibTeX entries and PDF URLs.
# Input: modelcard_step1_<tag>.parquet, github_readme_cache_<tag>.parquet, downloaded_github_readmes_<tag>_processed/, PDF/GitHub URLs
# Output: modelcard_all_title_list_<tag>.parquet, github_readme_cache_update_<tag>.parquet, github_extraction_cache_<tag>.json, all_links_with_category_<tag>.csv
python -m src.data_preprocess.step2_arxiv_github_title --tag 251117 > logs/step2_arxiv_github_title_251117.log 2>&1

# Save deduplicated titles for querying Semantic Scholar (S2ORC).
# Input: modelcard_all_title_list_<tag>.parquet
# Output: modelcard_dedup_titles_<tag>.json, modelcard_title_query_results_<tag>.json, modelcard_all_title_list_mapped_<tag>.parquet
python -m src.data_preprocess.step2_s2orc_save --tag 251117 > logs/step2_s2orc_save_251117.log 2>&1

<details>
<summary>Optional: LLM/S2ORC pipelines (currently skipped)</summary>
#### Option1:
# Query Semantic Scholar API for citation information (alternative to local database if no key, but may hit rate limits).
# Input: modelcard_dedup_titles.json
# Output: s2orc_query_results.parquet, s2orc_citations_cache.parquet, s2orc_references_cache.parquet, s2orc_titles2ids.parquet
python -m src.data_preprocess.s2orc_API_query > logs/s2orc_API_query_v3_429.log 2>&1
 - python -m src.data_preprocess.s2orc_log_429 > logs/s2orc_log_429.log 2>&1 # incase some title meet 429 error (API rate error)
 - python -m src.data_preprocess.s2orc_retry_missing > logs/s2orc_retry_missing.log 2>&1 # make up for the missing items
 - python -m src.data_preprocess.s2orc_merge > logs/s2orc_merge.log 2>&1 # parse the references and citations from retrieved results | I: s2orc*.parquet, O: s2orc_rerun.parquet
 - bash src/data_localindexing/build_mini_citation_es.sh > logs/build_mini_citation_es.log 2>&1 # I: xx | O: batch_results
# (Deprecate: Old method) bash src.data_localindexing/build_mini_citation_es.sh
# Extract full records from batch query results.
# Input: batch_results + hit_ids.txt
# Output: full_hits.jsonl
python -m src.data_localindexing.extract_full_records > logs/extract_full_records.log 2>&1
# Merge extracted full records.
# Input: full_hits.jsonl
# Output: s2orc_citations_cache, s2orc_references_cache, s2orc_query_results
python -m src.data_localindexing.extract_full_records_to_merge > logs/extract_full_records_to_merge.log 2>&1
- python -m src.data_preprocess.s2orc_merge > logs/s2orc_merge.log 2>&1 # I: s2orc*.parquet, O: s2orc_rerun.parquet
 # (deprecate) - bash src/data_localindexing/build_mini_s2orc_es.sh # choose dump data to setup and batch query |
  # I: paper_index_mini.db, modelcard_dedup_titles.json → O: Elasticsearch index (e.g., papers_index), query_cache.parquet
 - bash src/data_preprocess/step2_se_url_tab.sh # extract fulltext -> ref/cit info
# I: query_cache.parquet/s2orc_rerun.parquet, paper_index_mini.db, NDJSON files in /se_s2orc_250218 → O: extracted_annotations.parquet, tmp_merged_df.parquet, tmp_extracted_lines.parquet
</details>

# Download arXiv HTML content for table extraction.
# Input: extracted_annotations_<tag>.parquet, arxiv_titles_cache_<tag>.json
# Output: title2arxiv_new_cache_<tag>.json, arxiv_html_cache_<tag>.json, missing_titles_tmp_<tag>.txt, arxiv_fulltext_html_<tag>/*.html
# cp -r /Users/doradong/Repo/CitationLake/data/processed/extracted_annotations.parquet /Users/doradong/Repo/CitationLake/data/processed/extracted_annotations_251117.parquet
python -m src.data_preprocess.step2_arxiv_get_html --tag 251117 > logs/step2_arxiv_get_html_251117.log 2>&1

# Extract tables from arXiv HTML files.
# Input: arxiv_html_cache.json, arxiv_fulltext_html/*.html, html_table.parquet (optional)
# Output: html_table.parquet, tables_output/*.csv
python -m src.data_preprocess.step2_arxiv_parse --tag 251117 > logs/step2_arxiv_parse_251117.log 2>&1
python -m src.data_preprocess.step2_arxiv_parse_v2 --n_jobs 16 --output_dir data/processed/tables_output_v2_251117 --tag 251117 --save_mode csv > logs/step2_arxiv_parse_v2_251117.log 2>&1  #/duckdb/sqlite 

# Integrate all processed table data (arXiv HTML + S2ORC extracted annotations) and process with LLM.
# Input: title2arxiv_new_cache_<tag>.json, html_table_<tag>.parquet/html_parsing_results_v2_<tag>.parquet, extracted_annotations_<tag>.parquet, pdf_download_cache_<tag>.json
# Output: batch_input_<tag>.jsonl, batch_output_<tag>.jsonl, llm_markdown_table_results_<tag>.parquet
python -m src.data_preprocess.step2_integration_s2orc_llm --tag 251117 > logs/step2_integration_s2orc_llm_251117.log 2>&1
# (Optional) Check OpenAI batch job status (if using LLM for table processing)
bash src/data_preprocess/openai_batchjob_status.sh > logs/openai_batchjob_status.log 2>&1

# (Optional) If the sequence is wrong, reproduce from the log...
#python -m src.data_preprocess.quick_repro
#cp -r llm_outputs/llm_markdown_table_results_aligned.parquet llm_outputs/llm_markdown_table_results.parquet

# Save LLM-processed tables into local CSVs.
# Input: llm_markdown_table_results_<tag>.parquet
# Output: llm_tables_<tag>/*.csv, final_integration_with_paths_v2_<tag>.parquet
python -m src.data_preprocess.step2_llm_save --tag 251117 > logs/step2_llm_save_251117.log 2>&1
```

### 4\. Label Ground Truth for Unionable Search Baselines


This section details the process of generating ground truth labels for table unionability.
```bash
python -m src.data_preprocess.step2_merge_tables --tag 251117 > logs/step2_merge_tables_251117.log 2>&1  # Merge all table lists from 4 resources (HuggingFace, GitHub, HTML, LLM) into a unified model ID file. Input: final_integration_with_paths_v2_<tag>.parquet, modelcard_all_title_list_<tag>.parquet, modelcard_step2_v2_<tag>.parquet. Output: modelcard_step3_merged_v2_<tag>.parquet
python -m src.data_gt.paper_citation_overlap --tag 251117 > logs/paper_citation_overlap_251117.log 2>&1  # Compute paper-pair citation overlap scores for ground truth. Input: extracted_annotations_<tag>.parquet. Output: modelcard_citation_all_matrices_<tag>.pkl.gz (REQUIRED for step3_gt)
python -m src.data_analysis.paper_relatedness_distribution --tag 251117 > logs/paper_relatedness_distribution_251117.log 2>&1  # (Optional) Plot violin figures of paper relatedness distribution. Input: modelcard_citation_all_matrices_<tag>.pkl.gz. Output: overlap_violin_by_mode_<tag>.pdf
python -m src.data_analysis.paper_relatedness_threshold --tag 251117 > logs/paper_relatedness_threshold_251117.log 2>&1  # (Optional) Determine paper relatedness thresholds. Input: modelcard_citation_all_matrices_<tag>.pkl.gz. Output: score_*.pdf files in data/analysis/
```

### Quality Control \!\!\! | Run some analysis

Ensure data quality and consistency before generating final ground truth.

```bash
python -m src.data_preprocess.step2_dedup_tables --tag 251117 > logs/step2_dedup_tables_251117.log 2>&1  # Deduplicate raw tables, prioritizing Hugging Face > GitHub > HTML > LLM. Input: modelcard_step3_merged_v2_<tag>.parquet. Output: modelcard_step3_dedup_v2_<tag>.parquet
python -m src.data_analysis.qc_dedup_fig --tag 251117 > logs/qc_dedup_fig_251117.log 2>&1  # Generate heatmaps from dedup results. Input: deduped_<tag>/dup_matrix.pkl, deduped_<tag>/stats.json. Output: heatmaps in data/analysis/
python -m src.data_analysis.qc_stats --tag 251117 > logs/qc_stats_251117.log 2>&1  # Print table #rows #cols. Input: modelcard_step3_dedup_v2_<tag>.parquet. Output: benchmark_results_v2_<tag>.parquet
python -m src.data_analysis.qc_stats_fig --tag 251117 > logs/qc_stats_fig_251117.log 2>&1  # Plot benchmark results. Input: benchmark_results_v2_<tag>.parquet. Output: benchmark_metrics_vertical_v2_<tag>.pdf/png
python -m src.data_analysis.qc_anomaly --recursive > logs/qc_anomaly.log 2>&1 #(Optional)
python -m src.tools.show_table_diff_md 0ae65809ffffa20a2e5ead861e7408ac_table_0.csv > logs/show_table_diff.log 2>&1 #(Optional) # compare v1 and v2 table diff
# (Optional) Double-check deduplication and mapping logic.
# python -m src.data_analysis.qc_dc > logs/qc_dc.log 2>&1
# Obtain file counts directly from folders to verify against statistics.
bash src/data_analysis/count_files.sh > logs/count_files.log 2>&1
```

### Final Ground Truth Generation
Generate the definitive ground truth files for evaluation.

```bash
# (Depre) python -m src.data_gt.step3_create_symlinks --tag 251117 > logs/step3_create_symlinks_251117.log 2>&1  # Create symbolic links for organizing processed tables. Input: modelcard_step3_dedup_v2_<tag>.parquet. Output: modelcard_step4_v2_<tag>.parquet, sym_*_csvs_* (symbolic links)
bash src/data_gt/step3_gt.sh 251117 > logs/step3_gt_251117.log 2>&1  # Build ground truth (paper-level, model-level, dataset-level). Input: modelcard_citation_all_matrices_<tag>.pkl.gz, modelcard_step3_dedup_v2_<tag>.parquet, final_integration_with_paths_v2_<tag>.parquet, modelcard_all_title_list_<tag>.parquet. Output: data/gt/* (no versioning)
python -m src.tools.check_gt_coverage --csv-name 1910.09700_table0.csv --levels direct --mode both > logs/check_gt_coverage.log 2>&1 # (Optional)
python -m src.data_gt.debug_npz --gt-dir data/gt/ > logs/debug_npz.log 2>&1 # (Optional) Debug NPZ ground truth files to ensure valid conditions.
# Process SQLite ground truth into pickle files (if applicable from other benchmarks).
python -m src.data_localindexing.turn_tus_into_pickle > logs/turn_tus_into_pickle.log 2>&1
# (deprecate) python -m src.data_gt.gt_combine > logs/gt_combine.log 2>&1
python -m src.data_gt.modelcard_matrix --tag 251117 > logs/modelcard_matrix_251117.log 2>&1  # Add other two levels of citation graphs (modelcard and dataset). Input: modelcard_step1_<tag>.parquet, modelcard_step3_dedup_v2_<tag>.parquet, modelcard_step3_merged_v2_<tag>.parquet. Output: modelcard_gt_related_model_<tag>.parquet, data/gt/scilake_gt_modellink_*_<tag>.npz
python -m src.data_gt.merge_union --level direct --tag 251117 > logs/merge_union_251117.log 2>&1  # Merge union ground truth. Input: data/gt/*_<tag>.npz, *_<tag>.pkl. Output: data/gt/csv_pair_union_*_<tag>_processed.npz
python -m src.data_analysis.gt_distri --tag 251117 > logs/gt_distri_251117.log 2>&1  # Plot GT length distribution (boxplot/violin). Input: data/gt/*_<tag>.npz
python -m src.data_gt.nonzeroedge --gt_dir data/gt --tag 251117 > logs/nonzeroedge_251117.log 2>&1  # Compute non-zero edge statistics for citation graphs. Input: data/gt/*_<tag>.npz
# (test)python -m src.data_gt.test_modelcard_update --mode dataset > logs/test_modelcard_update.log 2>&1 # check whether matrix multiplication and for loop obtain the same results
#(test)python -m src.data_gt.convert_adj_to_npz --input data/gt/scilake_gt_modellink_dataset_adj_processed.pkl --output-prefix data/gt/scilake_gt_modellink_dataset > logs/convert_adj_to_npz.log 2>&1 # pkl2npz
python -m src.data_gt.create_csvlist_variants --level direct --tag 251117 > logs/create_csvlist_variants_251117.log 2>&1  # Update CSV lists for various ground truth variants. Input: data/gt/*_<tag>.pkl
# (depreate) python -m src.data_gt.create_gt_variants data/gt/csv_pair_adj_overlap_rate_processed.pkl # produce _s, _t, _s_t for pkl files
# (deprecate) python -m src.data_gt.print_relations_stats data/tmp/relations_all.pkl # print stats for matrix
# (deprecate) python -m src.data_analysis.gt_fig # plot stats
```

### 5\. Create Symlinks for Starmie Integration

Prepare data and augmentations for integration with the Starmie benchmark framework.

**Two main steps:**

1. **Create augmented table folders (tr/str)**: Generate transpose and string-augmented versions of tables
2. **Create symlinks**: Link CitationLake tables to starmie_internal/data/scilake_final_<tag>/datalake

```bash
# Step 1: Create augmented table folders (tr/str/str_tr)
# This creates folders like: deduped_hugging_csvs_v2_251117_tr, deduped_hugging_csvs_v2_251117_str
python -m src.data_symlink.trick_aug --repo_root /u1/z6dong/Repo --mode tr --tag 251117 > logs/trick_aug_tr_251117.log 2>&1   # Create transpose augmented folders
python -m src.data_symlink.trick_aug --repo_root /u1/z6dong/Repo --mode str --tag 251117 > logs/trick_aug_str_251117.log 2>&1  # Create string augmented folders
# Or process all modes:
#python -m src.data_symlink.trick_aug --repo_root /u1/z6dong/Repo --mode str_tr --tag 251117 > logs/trick_aug_str_tr_251117.log 2>&1  # Create both str and tr augmented folders

# Step 2: Create symlinks from CitationLake to starmie_internal
# This creates symlinks in starmie_internal/data/scilake_final_<tag>/datalake
python -m src.data_symlink.ln_scilake --repo_root /u1/z6dong/Repo --mode base --dir-name scilake_final_251117 > logs/ln_scilake_base_251117.log 2>&1  # Base mode
python -m src.data_symlink.ln_scilake --repo_root /u1/z6dong/Repo --mode str --dir-name scilake_final_251117_str > logs/ln_scilake_str_251117.log 2>&1  # Str mode
python -m src.data_symlink.ln_scilake --repo_root /u1/z6dong/Repo --mode tr --dir-name scilake_final_251117_tr > logs/ln_scilake_tr_251117.log 2>&1  # Tr mode
# Or process all modes at once:
python -m src.data_symlink.ln_scilake --repo_root /u1/z6dong/Repo --mode all --dir-name scilake_final_251117 > logs/ln_scilake_all_251117.log 2>&1  # All modes (base, str, tr, tr_str)

# Note: For default v2 version (no tag), omit --tag and use scilake_final_v2 as dir-name
python -m src.data_symlink.trick_aug --repo_root /u1/z6dong/Repo --mode tr > logs/trick_aug_tr_v2.log 2>&1  # No tag, uses v2 folders
python -m src.data_symlink.ln_scilake --repo_root /u1/z6dong/Repo --mode base --dir-name scilake_final_v2 > logs/ln_scilake_base_v2.log 2>&1  # No tag, uses v2
```

### 6\. Run Updated Starmie Scripts

Execute Starmie's pipeline for contrastive learning, embedding extraction, and search

**Versioning with TAG:**
All Starmie scripts support a `TAG` environment variable for versioning:
- `TAG=v2` (or omit TAG) for the default v2 version
- `TAG=<date>` (e.g., `TAG=251117`) for date-based versions
- `TAG=` (empty) will default to `v2`

```bash
# bash prepare_sample.sh > logs/prepare_sample.log 2>&1  # Sample 1000 tables from each resource folder for evaluation
# python -m src.data_symlink.prepare_sample_server --root_dir /u4/z6dong/Repo --output scilake_final --output_file scilake_final_filelist.txt --limit 2000 --seed 42 > logs/prepare_sample_server.log 2>&1  # Alternative for server
python -m src.data_symlink.prepare_sample --root_dir /u1/z6dong/Repo --output_file scilake_final_filelist.txt --limit 1000 --seed 42 > logs/prepare_sample.log 2>&1  # Another substitution
# (deprecated) python -m src.data_symlink.prepare_sample_tricks --input_file scilake_final_filelist.txt > logs/prepare_sample_tricks.log 2>&1  # Create file lists for trick-augmented files (Input: scilake_final_filelist.txt, Output: scilake_final_filelist_{tricks}_filelist.txt)
# (deprecated) python -m src.data_symlink.ln_scilake_final_link --filelist scilake_final_filelist.txt scilake_final_filelist_val.txt > logs/ln_scilake_final_link.log 2>&1  # Create validation file lists
# bash check_empty.sh > logs/check_empty.log 2>&1  # (deprecate) (already processed in QC step) filter out empty files (or low quality files later)
bash scripts/step1_pretrain.sh > logs/step1_pretrain.log 2>&1  # Fine-tune contrastive learning model

# Using default v2 version (backward compatible)
bash scripts/step2_extractvectors.sh > logs/step2_extractvectors_v2.log 2>&1  # Encode embeddings for query and datalake items
bash scripts/step3_hnsw_search.sh > logs/step3_hnsw_search_v2.log 2>&1  # Perform data lake search (retrieval)
bash scripts/step3_processmetrics.sh > logs/step3_processmetrics_v2.log 2>&1  # Extract metrics based on ground truth and retrieval results; plot figures
bash eval_per_resource.sh > logs/eval_per_resource_v2.log 2>&1  # Run ablation study on different resources (after getting results)

# Using date-based tag (e.g., 251117)
TAG=251117 bash scripts/step2_extractvectors.sh > logs/step2_extractvectors_251117.log 2>&1
TAG=251117 bash scripts/step3_hnsw_search.sh > logs/step3_hnsw_search_251117.log 2>&1
TAG=251117 bash scripts/step3_processmetrics.sh > logs/step3_processmetrics_251117.log 2>&1
TAG=251117 bash scripts/step3_processmetrics_all.sh <EXPERIMENT_INDEX> > logs/step3_processmetrics_all_251117.log 2>&1
TAG=251117 bash eval_per_resource.sh > logs/eval_per_resource_251117.log 2>&1
# bash eval_per_resource.sh  # (Alternatively, run before getting results)
# bash scripts/step4_discovery.sh  # (Optional)
```

### 7\. Baseline: Dense Search, Sparse Search, Hybrid Search

Run baseline table embedding and retrieval methods for comparison, for faiss cpu/gpu installation, see [FAISS GitHub repository](https://github.com/facebookresearch/faiss).

**Tag Support**: All baseline scripts support `TAG` environment variable for versioning:
- Use `TAG=251117` (or any date/tag) to use tagged versions of input/output files
- All file paths automatically include the tag suffix when TAG is set
- Example: `all_valid_title_valid.txt` → `all_valid_title_valid_251117.txt` when `TAG=251117`

```bash
### 1. Baseline1: Dense Search
# Unified script - supports base/str/tr modes
# Note: All three modes use the same Python script (table_retrieval_pipeline.py) with different --mode arguments
# The unified script replaces the separate pipeline_str.sh and pipeline_tr.sh scripts
TAG=251117 bash src/baseline1/table_retrieval_pipeline_unified.sh base > logs/baseline1_pipeline_base_251117.log 2>&1  # base mode: full pipeline (filter + encode + build_faiss + search + postprocess)
TAG=251117 bash src/baseline1/table_retrieval_pipeline_unified.sh str --skip-search > logs/baseline1_pipeline_str_251117.log 2>&1  # str mode: filter + encode only (for mixed experiments)
TAG=251117 bash src/baseline1/table_retrieval_pipeline_unified.sh tr --skip-search > logs/baseline1_pipeline_tr_251117.log 2>&1   # tr mode: filter + encode only (for mixed experiments)
TAG=251117 bash src/baseline1/combine_embedding.sh > logs/baseline1_combine_embedding_251117.log 2>&1  # for augmented ablation studies: step2 combine embedding and jsonl for ori+tr, ori+str, ori+tr+str
TAG=251117 bash src/baseline1/build_aug_faiss.sh > logs/baseline1_build_aug_faiss_251117.log 2>&1  # step3: build faiss
TAG=251117 bash src/baseline1/augment_search.sh > logs/baseline1_augment_search_251117.log 2>&1  # step4: search
TAG=251117 bash src/baseline1/postprocess_general.sh > logs/baseline1_postprocess_general_251117.log 2>&1  # step5: postprocess: split into ori / tr / str json
TAG=251117 bash src/baseline1/standardize_filenames.sh > logs/baseline1_standardize_filenames_251117.log 2>&1  # step6: postprocess: all files back to ori csv name
TAG=251117 bash scripts/step3_processmetrics_all.sh <index> > logs/baseline1_processmetrics_251117.log 2>&1  # compute metrics under starmie: run baseline metrics computation

### 2. Baseline2: Sparse search
TAG=251117 bash src/baseline2/get_metadata.sh > logs/baseline2_get_metadata_251117.log 2>&1 # Baseline2: Sparse search get metadata
TAG=251117 bash src/baseline2/sparse_search.sh > logs/baseline2_sparse_search_251117.log 2>&1 # Baseline2: Sparse search

### 3. Baseline3: Hybrid (Sparse + Dense search)
# Note: Hybrid search uses Python scripts with command-line arguments
# Use tagged file paths when calling the scripts
python src/baseline2/search_with_pyserini_hybrid.py \
  --sparse-index data/tmp/index_251117 \
  --dense-index data/tmp/index_dense_251117 \
  --queries data/tmp/queries_table.tsv \
  --mapping data/tmp/queries_table_mapping.json \
  --k 11 --alpha 0.45 --device cpu > logs/baseline2_hybrid_search_251117.log 2>&1
```

<!-- ### 8. Model Search - Dense first: 
```bash
bash src/modelsearch/base_densesearch.sh
python -m src.modelsearch.compare_baselines \
  --model_id Salesforce/codet5-base \
  --relationship_parquet data/processed/modelcard_step3_dedup.parquet \
  --table_search_result ~/Repo/starmie_internal/results/scilake_final/test_hnsw_search_drop_cell_tfidf_entity_full.json \
  --modelsearch_base_result output/modelsearch/modelsearch_neighbors.json \
  --output_md output/compare_Salesforce_codet5-base.md

# llm feedback
``` -->

<details>
<summary><strong>9. GPT Evaluation of Table Relatedness and Model Relatedness</strong></summary>

**Script**: `src/gpt_evaluation/step1_table_sampling.py`
**Purpose**: Sample balanced table pairs for GPT evaluation across three ground truth levels (Paper, ModelCard, Dataset).
**Key Features**:
- **Multi-level balanced sampling**: Each GT level maintains 50/50 positive/negative ratio
- **8-way combination analysis**: Analyzes all 8 possible label combinations (2³)
- **Efficient batch querying**: Uses sparse matrix indexing for O(log deg) queries
- **Large pool sampling**: Samples from 100k+ pool, then filters for balance
**Usage**:
```bash
python src/gpt_evaluation/step1_table_sampling.py \
    --total-target 200 \
    --seed 42 > logs/step1_table_sampling.log 2>&1
```
**Notes**:
- `--total-target`: Number of unique pairs to output (default: 500)
- Pool size is auto-set to `max(6x target, 10000)` for better diversity
- No need to specify `--n-samples-pool` anymore

**Output**: 500 unique pairs with 8-way combination statistics
```bash
[5/5] Post-selection statistics:
  8-Way (selected):
    (0, 0, 0): None                 -   21 (10.61%)
    (0, 0, 1): Dataset only         -   11 ( 5.56%)
    (0, 1, 0): ModelCard only       -   21 (10.61%)
    (0, 1, 1): ModelCard + Dataset  -   12 ( 6.06%)
    (1, 0, 0): Paper only           -   35 (17.68%)
    (1, 0, 1): Paper + Dataset      -   24 (12.12%)
    (1, 1, 0): Paper + ModelCard    -   24 (12.12%)
    (1, 1, 1): All three            -   50 (25.25%)

  Per-level (selected):
    Paper     :  133 pos (67.17%) /   65 neg (32.83%)
    Modelcard :  107 pos (54.04%) /   91 neg (45.96%)
    Dataset   :   97 pos (48.99%) /  101 neg (51.01%)
```

**Visualization**: Generate heatmap visualization for paper figures
```bash
python src/gpt_evaluation/visualize_sampling_2x4_horizontal.py > logs/visualize_sampling.log 2>&1
```

#### Step2: Query OpenRouter for table relatedness evaluation.
```bash
python -m src.gpt_evaluation.step2_query_openrouter --input output/gpt_evaluation/table_1M_fix_unique_pairs.jsonl --output output/gpt_evaluation/step2_full_198.jsonl 2>&1 | tee logs/step2_query_openrouter.log &
# retry
# python -m src.gpt_evaluation.step2_retry_failed \
#     --input output/gpt_evaluation/step2_openrouter_results_full.jsonl \
#     --output output/gpt_evaluation/step2_openrouter_results_retried.jsonl > logs/step2_retry_failed.log 2>&1
# add merge please
python -m src.gpt_evaluation.step2_merge_results --main output/gpt_evaluation/step2_full_198.jsonl --additional output/gpt_evaluation/step2_gpt4mini_full.jsonl --output output/gpt_evaluation/step2_all_5models.jsonl > logs/step2_merge_results.log 2>&1

# crowdsourcing metrics analysis
python -m src.gpt_evaluation.visualize_crowdsourcing_metrics > logs/visualize_crowdsourcing_metrics.log 2>&1 # generate figures for subset
python -m src.gpt_evaluation.visualize_crowdsourcing_metrics_full > logs/visualize_crowdsourcing_metrics_full.log 2>&1 # generate figures for full dataset
```

</details>

---

### (Deprecated)Model Relatedness Sampling

**Script**: `src/gpt_evaluation/step1_model_sampling.py`

**Purpose**: Sample model pairs for model relatedness evaluation.

**Usage**:
```bash
python src/gpt_evaluation/step1_model_sampling.py --n-samples 200 --seed 42 > logs/step1_model_sampling.log 2>&1
```


### 10. Table Integration:
```bash

```

### Analysis on Results

Tools for analyzing the retrieval results and ground truth.

```bash
# Get top-10 results from step3_search_hnsw.
python -m src.data_analysis.report_generation --json_path ~/Repo/starmie_internal/tmp/test_hnsw_search_scilake_large_full.json > logs/report_generation.log 2>&1
# then gen markdown
python -m src.data_analysis.report_generation --json_path data/baseline/baseline1_dense.json --query_table 1810.04805_table4.csv > logs/report_generation_baseline.log 2>&1
# --show_model_titles

# Check if a specific CSV pair is related in the ground truth.
python -m src.data_gt.check_pair_in_gt --gt-dir data/gt --csv1 0ab2d85d37_table1.csv --csv2 096d51652d_table1.csv > logs/check_pair_in_gt.log 2>&1

# Count unique CSVs in retrieval results.
python count_unique_csvs.py --results /u1/z6dong/Repo/starmie_internal/results/scilake_final/test_hnsw_search_drop_cell_tfidf_entity_full.json --gt /u1/z6dong/Repo/ModelLake/data/gt/csv_pair_adj_overlap_rate_processed.pkl > logs/count_unique_csvs.log 2>&1

# CSV to ModelID Mapping
# Get modelIDs from CSV files (supports GitHub, HuggingFace, HTML, LLM sources)
python batch_process_tables.py -i tmp/top_tables.txt -o tmp/top_tables_with_keywords.csv > logs/batch_process_tables.log 2>&1
# Analyze HTML table files and compare column counts between v1 and v2
# Input: tmp/top_tables_with_keywords.csv (filtered for HTML source)
# Output: tmp/html_table_analysis.csv with column count comparison
# Purpose: Compare HTML table v2 vs v1 column counts and analyze reduction
python src/data_analysis/analyze_html_tables.py > logs/analyze_html_tables.log 2>&1
python src/data_analysis/analyze_huggingface_tables.py > logs/analyze_huggingface_tables.log 2>&1
```

### Additional Statistics Analysis

Scripts for generating further insights into the dataset.

```bash
python -m src.data_preprocess.step1_analysis > logs/step1_analysis.log 2>&1 # get analysis on proportion of different links (sql)
python -m src.data_preprocess.query_compare_API_local > logs/query_compare_API_local.log 2>&1 # compare the query results among local and API from s2orc
TODO: add statistics analysis from dataset_processed.ipynb
```

(Deprecated scripts: Previously we download pdfs and try to parse them. However, we find semantic scholar dataset includes it.)
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

```bash
# get parquet schema
python src.data_analysis.list_parquet_schemas > logs/parquet_schema.log 2>&1
# get attributes duplicate analysis
#python complete_duplicate_analysis.py > logs/complete_duplicate_analysis_results.txt 2>&1
python src.data_analysis.column_size_analysis --include-modelid > logs/parquet_storage.log 2>&1

# load all files into duckdb
python src.data_analysis.load_sv_to_db --engine sqlite --db-path deduped_hugging_csvs_v2.sqlite --input-dir data/processed/deduped_hugging_csvs_v2 > logs/load_sv_to_db.log 2>&1
# get relational keys from other key automatically (require logs/parquet_schema.log)
python -m src.data_analysis.get_from --target html_table_list_mapped_dedup --source modelId --value google-bert/bert-base-uncased > logs/get_from.log 2>&1
python -m src.data_analysis.get_from --target readme_path --source csv_paths --value "64dc62e53f_table2.csv" >> logs/get_from.log 2>&1
python -m src.data_analysis.get_from --target modelId --source hugging_table_list --value data/processed/deduped_hugging_csvs/021f09961f_table1.csv >> logs/get_from.log 2>&1
python -m src.data_analysis.get_from --target modelId --source pdf_link --value https://arxiv.org/pdf/0803.1019 >> logs/get_from.log 2>&1
python get_modelid_from_arxiv_comprehensive.py "0803.1019" --search-all --debug > logs/get_modelid_from_arxiv.log 2>&1
# step by step filtering img
python -m src.data_analysis.card_statistics > logs/card_statistics.log 2>&1 # get statistics of model cards
python -m src.data_analysis.hf_models_analysis > logs/hf_models_analysis.log 2>&1 # get statistics of models in Hugging Face
# after carefully examining
python -m src.data_analysis.filtered_gt_visualization > logs/filtered_gt_visualization.log 2>&1
python -m src.data_analysis.quick_visualization_final > logs/quick_visualization_final.log 2>&1
# the count v.s. time visualization
python src/data_analysis/table_model_counts_over_time.py --use-v2 --output-dir data/analysis > logs/table_model_counts_over_time.log 2>&1
# TODO: top-10!
```

```bash
# periodically upload to hugging face : Files: modelcard_step3_dedup_v2_<tag>.parquet, hugging, github, html, llm.zip
# huggingface-cli login
# huggingface-cli repo create your-username/your-dataset --type dataset
python -m src.data_preprocess.upload_to_hf_dataset --tag 251117 --dataset-name your-username/your-dataset --no-dry-run > logs/upload_to_hf_dataset_251117.log 2>&1
# TODO: versioning for zip
```
