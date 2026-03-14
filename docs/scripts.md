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
# Download modelcards/datasetcards dataset
python -m src.data_preprocess.download_hf_dataset --date 251117 --type modelcard/datasetcard > logs/download_hf_dataset_251117.log 2>&1
```

### 1\. Parse Initial Elements

This step extracts key metadata from model cards and associated links.
```bash
# Split readme and tags, parse URLs, parse BibTeX entries.
# Output: ['modelId', 'author', 'last_modified', 'downloads', 'likes', 'library_name', 'tags', 'pipeline_tag', 'createdAt', 'card', 'card_tags', 'card_readme', 'pdf_link', 'github_link', 'all_links', 'extracted_bibtex', 'extracted_bibtex_tuple', 'parsed_bibtex_tuple_list', 'successful_parse_count']

#python -m src.data_preprocess.step1_parse --raw-date 251117 --versioning --baseline-step1 data/processed/modelcard_step1.parquet > logs/step1_parse_251117.log 2>&1 # incremental mode, based on the previous step1 result
# or 
python -m src.data_preprocess.step1_parse --raw-date 251117 > logs/step1_parse_251117.log 2>&1 # output: modelcard_step1_251117.parquet
python -m src.data_preprocess.step1_down_giturl --tag 251117 --versioning --baseline-cache data/processed/github_readme_cache.parquet > logs/step1_down_giturl_251117.log 2>&1 # Download GitHub READMEs; Input: modelcard_step1_251117.parquet. Download: only new files to data/downloaded_github_readmes_251117/. Output: (1) dir data/downloaded_github_readmes_251117/, (2) github_readmes_info_251117.parquet, (3) github_readme_cache_251117.parquet. All saved paths in (2)(3) are unified as data/downloaded_github_readmes_251117/ (reused from baseline are not re-downloaded; run ln_giturl to symlink baseline into this dir).
python -m src.data_preprocess.ln_giturl --source-dir data/downloaded_github_readmes --target-dir data/downloaded_github_readmes_251117 > logs/ln_giturl_251117.log 2>&1 # Symlink all .md from source into target; skip if name already in target. In case that we try to analyze github readmes folder in the future
# (Optional) find data/downloaded_github_readmes -type f -exec stat -f "%z %N" {} + | sort -nr | head -n 50 | awk '{printf "%.2f MB %s\n", $1/1024/1024, $2}' > logs/find_large_readmes.log 2>&1 # some readme files are too large, they are actually model files
```

### 2\. Download and Build Database for Faster Querying

This step sets up local databases for efficient querying of Semantic Scholar data.

I don't update this section anymore, as the semantic scholar dataset is too large to maintain.

<details>
<summary>Click to expand database setup commands</summary>

```bash
# TODO: add command from privatecommonscript to here, for downloading the semantic scholar here
# Requirement: cd to the path of downloaded dataset, e.g.: cd ~/shared_data/se_s2orc_250218
python -m src.data_localindexing.build_mini_s2orc build --directory /u501/z6dong/shared_data/se_s2orc_250218/ # After downloading semantic scholar dataset, build database based on it.
python -m src.data_localindexing.build_mini_s2orc query --title "BioMANIA: Simplifying bioinformatics data analysis through conversation" --directory /u501/z6dong/shared_data/se_s2orc_250218/ # After building up database, query title based on db file.
python -m src.data_localindexing.build_mini_s2orc query_cid --corpusid 248779963 --directory /u501/z6dong/shared_data/se_s2orc_250218

# issue: citation edge is hard to store, it is too much ... Solution: I think we better using the API to query citation relationship? Or use cypher to query over graph condensely
# python -m src.data_localindexing.build_complete_citation build --directory ./ # build db for citation dataset
# python -m src.data_localindexing.build_complete_citation query --citationid 169 --directory ./
# (Optional) if you don't have key, use public API for querying citations instead
# python -m src.data_preprocess.step1_citationAPI # get citations through bibtex only by API. TODO: Update for bibtex + url, not bibtex only. TODO: Update for all bibtex, not the first bibtex

# Optional solution: we use kuzu database to store node and edge
#python -m src.data_localindexing.build_mini_citation_kuzu --mode build --directory /u501/z6dong/shared_data/se_citations_250218/
#python -m src.data_localindexing.test_node_edge_db # test how many nodes and edges are in built database
# issue: slow for our 300G ndjson files, not suitable for this stage

# Optional solution: we use neo4j database to store and query
#python src.data_localindexing.build_mini_citation_neo4j --mode build --directory ./ --fields minimal
#python src.data_localindexing.build_mini_citation_neo4j --mode query --citationid 248811336
# for slurm run this script to keep neo4j open in another terminal
# sbatch src.data_localindexing.neo4j_slurm

# fuzzy matching: elastic search for s2orc
python -m src.data_localindexing.build_mini_s2orc_es --mode build --directory /u501/z6dong/shared_data/se_s2orc_250218 --index_name papers_index --db_file /u501/z6dong/shared_data/se_s2orc_250218/paper_index_mini.db
python -m src.data_localindexing.build_mini_s2orc_es --mode query --directory /u501/z6dong/shared_data/se_s2orc_250218 --index_name papers_index --query "BioMANIA: Simplifying bioinformatics data analysis through conversation"
python -m src.data_localindexing.build_mini_s2orc_es --mode test --directory /u501/z6dong/shared_data/se_s2orc_250218 --index_name papers_index --db_file /u501/z6dong/shared_data/se_s2orc_250218/paper_index_mini.db
```
</details>

### 3\. Extract Tables to Local Folder

This step extracts tabular data from various sources and processes it.
```bash
# Extract tables from Hugging Face Model Cards and GitHub READMEs. Saves CSVs to local folder.
# Versioning mode (with tag):
# Process downloaded GitHub HTML files to Markdown. Skips when output file already exists (e.g. ln -s into _processed). 
# Input: data/downloaded_github_readmes_<tag>/
# Output: data/downloaded_github_readmes_<tag>_processed/, data/processed/md_parsing_results_v2_<tag>.parquet
# (Optional) python -m src.data_preprocess.ln_giturl --source-dir data/downloaded_github_readmes_processed --target-dir data/downloaded_github_readmes_251117_processed > logs/ln_giturl_processed_251117.log 2>&1
python -m src.data_preprocess.step2_git_md2text --tag 251117 > logs/step2_git_md2text_251117.log 2>&1

# Extract tables from Hugging model cards + GitHub READMEs. Input: modelcard_step1, github_readmes_info, downloaded_github_readmes_<tag>/ (not _processed).
# Input: data/processed/modelcard_step1_<tag>.parquet, github_readmes_info_<tag>.parquet, downloaded_github_readmes_<tag>/
# Output: data/processed/modelcard_step2_v2_<tag>.parquet, data/processed/deduped_hugging_csvs_v2_<tag>/, data/processed/hugging_deduped_mapping_v2_<tag>.json, data/processed/deduped_github_csvs_v2_<tag>/, md_to_csv_mapping.json
############################################### Here we only keep v2 version for extracting table as this is more accurate; see v1 extracting, check the previous packaged version on github
python -m src.data_preprocess.step2_hugging_github_extract --tag 251117 > logs/step2_hugging_github_extract_251117.log 2>&1

# Extract titles from arXiv and GitHub URLs (not S2ORC). For BibTeX entries and PDF URLs.
# Input: modelcard_step1_<tag>.parquet, github_readme_cache_<tag>.parquet, downloaded_github_readmes_<tag>_processed/, PDF/GitHub URLs
# Output: modelcard_all_title_list_<tag>.parquet, all_title_list_intra_row_dedup_groups_<tag>.json
# (Output but not used anymore) github_readme_cache_update_<tag>.parquet, github_extraction_cache_<tag>.json, all_links_with_category_<tag>.csv
python -m src.data_preprocess.step2_arxiv_github_title --tag 251117 > logs/step2_arxiv_github_title_251117.log 2>&1 # This one is slow..
# (One-time fix without run all: PYTHONPATH=. python bak/dedup_all_title_list_intra_row_251117.py)

# Save deduplicated titles for querying Semantic Scholar (S2ORC). Cross-row dedup: same normalize. Output: modelcard_dedup_titles_<tag>.json, s2orc_cross_row_dedup_groups_<tag>.json
python -m src.data_preprocess.step2_s2orc_save --tag 251117 > logs/step2_s2orc_save_251117.log 2>&1

# non-main pipeline scripts are documented in `docs/depre_scripts.md`.
<details>
#### Option1:
# save some searched results, only search the missing titles
#cp -r data/processed/s2orc_titles2ids.parquet data/processed/s2orc_titles2ids_251117.parquet
#cp -r data/processed/s2orc_citations_cache.parquet data/processed/s2orc_citations_cache_251117.parquet
#cp -r data/processed/s2orc_references_cache.parquet data/processed/s2orc_references_cache_251117.parquet

# Query Semantic Scholar API for citation information (alternative to local database if no key, but may hit rate limits).
# Why API over local (build_mini_citation_es): (1) API provides fresher citations/references; (2) API's title fuzzy matching is more accurate (commercialized) than our local ES fuzzy match.
python -m src.data_preprocess.s2orc_title2ids_API --tag 251117 > logs/s2orc_title2ids_API_251117.log 2>&1 #  Input: modelcard_dedup_titles_<tag>.json  Output: s2orc_titles2ids_<tag>.parquet
python -m src.data_preprocess.s2orc_refcit_API --tag 251117 > logs/s2orc_refcit_API_251117.log 2>&1 # Input: s2orc_titles2ids_<tag>.parquet Output: s2orc_citations_cache_<tag>.parquet, s2orc_references_cache_<tag>.parquet # issue: reference could be queried, but citation not
# Or 
# (local corpus version) python -m src.data_localindexing.s2orc_refcit_local --tag 251117 --src_dir /u501/z6dong/shared_data/se_citations_250218 > logs/extract_full_records.log 2>&1 # Input: batch_results + hit_ids_<tag>.txt, output: full_hits_<tag>.jsonl
# (local corpus version) python -m src.data_localindexing.s2orc_refcit_local_post --tag 251117 > logs/s2orc_local_query_ref_cit_251117.log 2>&1 # Input: full_hits_<tag>.jsonl (or fallback full_hits.jsonl), Output: s2orc_*_<tag>.parquet
# (deprecate) - bash src/data_localindexing/build_mini_s2orc_es.sh # choose dump data to setup and batch query | I: paper_index_mini.db, modelcard_dedup_titles.json → O: Elasticsearch index (e.g., papers_index), query_cache.parquet
python -m src.data_preprocess.s2orc_merge --tag 251117 > logs/s2orc_merge_251117.log 2>&1 # parse refs/cits | I: s2orc_cit/ref_cache_251117.parquet, O: s2orc_rerun_251117.parquet
 #- bash src/data_localindexing/build_mini_citation_es.sh > logs/build_mini_citation_es.log 2>&1 # I: xx | O: batch_results
# (local corpus version) bash src/data_preprocess/s2orc_fulltext_local.sh # extract fulltext -> ref/cit info
# I: query_cache.parquet/s2orc_rerun.parquet, paper_index_mini.db, NDJSON files in /se_s2orc_250218 → O: extracted_annotations.parquet, tmp_merged_df.parquet, tmp_extracted_lines.parquet
### Option2: batch querying papers_index
# (local corpus version) python -m src.data_localindexing.build_mini_s2orc_es --mode batch_query --directory /u501/z6dong/shared_data/se_s2orc_250218 --index_name papers_index --titles_file data/processed/modelcard_dedup_titles_251117.json --cache_file data/processed/query_cache_251117.json # getting full tables
</details>

# Download arXiv HTML, extract tables from arXiv HTML files.
#python -m bak.analyze_bibtex_arxiv_ids --tag 251117 > logs/analyze_bibtex_arxiv_ids_251117.log 2>&1 # Input: s2orc_titles2ids_<tag>.parquet, modelcard_all_title_list_<tag>.parquet, Output: bibte_title_arxiv_s2orc_<tag>.parquet  # try saving some title:arxiv from bibtex

# Resolve title→arxiv_id. Init: s2orc + arxiv_titles_cache concat; then bibtex + OAI rescue; sync html_path from folder.
# Input: s2orc_titles2ids_<tag>.parquet (query_title, retrieved_title), arxiv_titles_cache_<tag>.json (url→title, init only)
#       bibtex_title_arxiv_<tag>.parquet, title2arxiv_oai_index_<tag>.parquet
# Output: title2arxiv_cache_<tag>.parquet, final_missing_titles_from_cache_<tag>.txt
python -m src.data_preprocess.arxiv_title2ids_oai --tag 251117 > logs/arxiv_title2ids_oai_251117_5.log 2>&1
bash scripts/ln_arxiv_html.sh 251117  # ln data/arxiv_fulltext_html/*.html → data/arxiv_fulltext_html_<tag>/, save downloading
# Download HTML for arxiv_ids in cache. Input: title2arxiv_cache_<tag>.parquet. Output: data/arxiv_fulltext_html_<tag>/*.html (no parquet write)
python -m src.data_preprocess.arxiv_fulltext_api --tag 251117 > logs/arxiv_fulltext_api_251117.log 2>&1
# run python -m src.data_preprocess.arxiv_title2ids_oai --tag 251117 > logs/arxiv_title2ids_oai_251117_4.log 2>&1 again to sync html_path from folder to cache.


# Extract tables from arXiv HTML files (v2: rowspan/colspan, ltx_table).
# Input: arxiv_fulltext_html_<tag>/*.html. Output: tables_output_v2_<tag>/*.csv, html_parsing_results_v2_<tag>.parquet
# Incremental by default (skips paper_ids already in parquet). Use --overwrite for full reprocess.
###############################################
#python -m src.data_preprocess.step2_arxiv_parse --tag 251117 > logs/step2_arxiv_parse_251117.log 2>&1  # deprecated v1
# we don't ln v1 to v2, because we change parsing logic
python -m src.data_preprocess.step2_arxiv_parse_v2 --n_jobs 16 --tag 251117 --save_mode csv > logs/step2_arxiv_parse_v2_251117.log 2>&1  # --overwrite for full run; save_mode: csv|duckdb 

# Integrate all processed table data (arXiv HTML + S2ORC extracted annotations) and process with LLM.
# Input: title2arxiv_cache_<tag>.parquet, html_parsing_results_v2_<tag>.parquet, extracted_annotations_<tag>.parquet, pdf_download_cache_<tag>.json
# Output: llm_markdown_table_results_v2_<tag>.parquet (optional: batch_input_v2_<tag>.jsonl/output_v2_<tag>.jsonl if running LLM)
# Use --skip-llm to skip LLM entirely (merge only, empty llm_response_raw) when not updating LLM
# (s2orc+LLM table source, deprecated) python -m src.data_preprocess.step2_integration_s2orc_llm --tag 251117 --skip-llm --v2_mode > logs/step2_integration_s2orc_llm_251117.log 2>&1
# Check OpenAI batch job status (if using LLM for table processing)
# bash src/data_preprocess/openai_batchjob_status.sh > logs/openai_batchjob_status.log 2>&1

# If the sequence is wrong, reproduce from the log...
#python -m src.data_preprocess.quick_repro
#cp -r llm_outputs/llm_markdown_table_results_aligned.parquet llm_outputs/llm_markdown_table_results_v2_<tag>.parquet
# Extract LLM-processed tables. Input: llm_markdown_table_results_v2_<tag>.parquet; Output: llm_tables_<tag>/*.csv, final_integration_with_paths_v2_<tag>.parquet
# (s2orc+LLM table source, deprecated) python -m src.data_preprocess.step2_llm_save --tag 251117 --v2_mode > logs/step2_llm_save_251117.log 2>&1
```

Finally, we merge table list from different sources back to modelID level.
```bash
# (merge after s2orc + LLM, depreated) python -m src.data_preprocess.step2_merge_tables --tag 251117 --v2_mode > logs/step2_merge_tables_v2_251117.log 2>&1  # Merge all table lists from 4 resources (HuggingFace, GitHub, HTML, LLM) into a unified model ID file.
# Input: final_integration_with_paths_v2_<tag>.parquet, modelcard_all_title_list_<tag>.parquet, modelcard_step2_v2_<tag>.parquet.
# Output: modelcard_step3_merged_v2_<tag>.parquet
```

To substitute step2_integration_s2orc_llm, step2_llm_save (we skip llm tables as it is unstable), step2_merge_tables, we can use step2_merge_tables_simplify to directly generate the final merged table list at modelID level (without running the LLM table extraction pipeline).
```bash
python -m src.data_preprocess.step2_merge_tables_simplify --tag 251117 --v2_mode > logs/step2_merge_tables_simplify_251117.log 2>&1 
  # Input: s2orc_rerun_<tag>.parquet, title2arxiv_cache_<tag>.parquet, modelcard_all_title_list_<tag>.parquet, 
  #        hugging_deduped_mapping_v2_<tag>.json, deduped_github_csvs_v2_<tag>/md_to_csv_mapping.json. html_parsing_results_v2_<tag>.parquet, modelcard_step2_v2_<tag>.parquet,
  # Output: modelcard_step3_merged_v2_<tag>.parquet
```

### Quality Control \!\!\! | Run some analysis

Ensure data quality and consistency before generating final ground truth.

```bash
# Umm, dedup better happen before merge, e.g. in s2orc_rerun.parquet.
python -m src.data_preprocess.step2_dedup_tables --tag 251117 --v2_mode > logs/step2_dedup_tables_v2_251117.log 2>&1  # Deduplicate raw tables, prioritizing Hugging Face > GitHub > HTML > LLM. Input: modelcard_step3_merged_v2_<tag>.parquet. Output: modelcard_step3_dedup_v2_<tag>.parquet, and others
python -m src.data_analysis.qc_dedup_fig --tag 251117 --v2_mode > logs/qc_dedup_fig_v2_251117.log 2>&1  # Generate heatmaps from dedup results. Input: deduped_v2_<tag>/dup_matrix_v2_<tag>.pkl, deduped_v2_<tag>/stats_v2_<tag>.json. Output: heatmaps heatmap_overlap_v2_<tag>.pdf / heatmap_percentage_v2_<tag>.pdf in data/analysis/
python -m src.data_analysis.qc_stats --tag 251117 --v2_mode > logs/qc_stats_v2_251117.log 2>&1  # Print table #rows #cols. Input: modelcard_step3_dedup_v2_<tag>.parquet. s2orc_rerun_<tag>.parquet. Output: benchmark_results_v2_<tag>.parquet, all_title_list_valid_v2_<tag>.parquet, all_valid_title_valid_v2_<tag>.txt. Here we filter out over large tables (max_cols=100, max_rows=200)
python -m src.data_analysis.qc_stats_fig --tag 251117 --v2_mode --exclude_resources llm > logs/qc_stats_fig_v2_251117.log 2>&1  # Plot benchmark results. Input: benchmark_results_v2_<tag>.parquet. Output: benchmark_metrics_vertical_v2_<tag>.pdf/png

# python -m src.data_analysis.qc_anomaly --recursive > logs/qc_anomaly.log 2>&1 # this one is without tag, as we don't run v1 with 251117 anymore.
# python -m src.data_analysis.show_table_diff_md 0ae65809ffffa20a2e5ead861e7408ac_table_0.csv > logs/show_table_diff.log 2>&1 # compare v1 and v2 table diff
# python -m src.data_analysis.qc_dc > logs/qc_dc.log 2>&1 # Double-check deduplication and mapping logic.
```

We could go for starmie searching and baselines searching. We need groundtruth for evaluation based on searched results and groundtruth results.

### 4\. Label Ground Truth for Unionable Search Baselines

This section details the process of generating ground truth labels for table unionability.
```bash
python -m src.data_gt.paper_citation_overlap --tag 251117 > logs/paper_citation_overlap_251117.log 2>&1  # Compute paper-pair citation overlap scores for ground truth. Input: s2orc_rerun_<tag>.parquet. Output: modelcard_citation_all_matrices_<tag>.pkl.gz (REQUIRED for step3_gt)
python -m src.data_analysis.paper_relatedness_distribution --tag 251117 > logs/paper_relatedness_distribution_251117.log 2>&1  # (Optional) Plot violin figures of paper relatedness distribution. Input: modelcard_citation_all_matrices_<tag>.pkl.gz. Output: overlap_violin_by_mode_<tag>.pdf
# (Deprecated) python -m src.data_analysis.paper_relatedness_threshold --tag 251117 > logs/paper_relatedness_threshold_251117.log 2>&1  # (Optional) Determine paper relatedness thresholds. Input: modelcard_citation_all_matrices_<tag>.pkl.gz. Output: score_*.pdf files in data/analysis/
```

### Final Ground Truth Generation
Generate the definitive ground truth files for evaluation.

```bash
bash src/data_gt/step3_gt.sh 251117 > logs/step3_gt_v2_251117.log 2>&1  # Build ground truth (paper-level, model-level, dataset-level). Input: modelcard_citation_all_matrices_<tag>.pkl.gz, modelcard_step3_dedup_v2_<tag>.parquet, s2orc_rerun_<tag>.parquet, modelcard_all_title_list_<tag>.parquet. Output: data/gt/* (no versioning)
# (Optional) python -m src.data_gt.check_gt_coverage --csv-name 1910.09700_table0.csv --levels direct --mode both > logs/check_gt_coverage.log 2>&1 
# (Optional) python -m src.data_gt.debug_npz --gt-dir data/gt/ > logs/debug_npz.log 2>&1 # Debug NPZ ground truth files to ensure valid conditions.
# Process SQLite ground truth into pickle files (if applicable from other benchmarks).
python -m src.data_localindexing.turn_tus_into_pickle > logs/turn_tus_into_pickle.log 2>&1
# (deprecate) python -m src.data_gt.gt_combine > logs/gt_combine.log 2>&1
python -m src.data_gt.modelcard_matrix --tag 251117 --v2_mode > logs/modelcard_matrix_v2_251117.log 2>&1  # Add other two levels of citation graphs (modelcard and dataset). Input: modelcard_step1_<tag>.parquet, modelcard_step3_dedup_v2_<tag>.parquet, modelcard_step3_merged_v2_<tag>.parquet. Output: modelcard_gt_related_model_v2_<tag>.parquet, data/gt/scilake_gt_modellink_*_v2_<tag>.npz
python -m src.data_gt.merge_union --level direct --tag 251117 --v2_mode > logs/merge_union_v2_251117.log 2>&1  # Merge union ground truth. Input: data/gt/*_v2_<tag>.npz, *_v2_<tag>.pkl. Output: data/gt/csv_pair_union_*_v2_<tag>_processed.npz
python -m src.data_analysis.gt_distri --tag 251117 --v2_mode > logs/gt_distri_251117.log 2>&1  # Plot GT length distribution (boxplot/violin). Input: data/gt/*_v2_<tag>.npz and *_v2_<tag>_processed.npz (requires merge_union first). Use same --tag as merge_union.
python -m src.data_gt.nonzeroedge --gt_dir data/gt --tag 251117 --v2_mode > logs/nonzeroedge_v2_251117.log 2>&1  # Compute non-zero edge statistics for citation graphs. Input: data/gt/*_v2_<tag>.npz
python -m src.data_gt.create_csvlist_variants --level direct --tag 251117 --v2_mode > logs/create_csvlist_variants_251117.log 2>&1  # Update CSV lists for various ground truth variants. Input: data/gt/*_v2_<tag>.pkl
# (deprecate) python -m src.data_analysis.gt_fig # plot stats
```

### 5\. Create Symlinks for Starmie Integration

Prepare data and augmentations for integration with the Starmie benchmark framework.

**Two main steps:**

0. zip and transfer the data to the server
1. **Create augmented table folders (tr/str)**: Generate transpose and string-augmented versions of tables
2. **Create symlinks**: Link ModelTables tables to starmie_internal/data/scilake_final_<tag>/datalake

```bash
bash src/postprocess/zip_with_mask.sh 251117 # Step 0: zip with mask
# Step 1: Create augmented table folders (tr/str) deduped_hugging_csvs_v2_251117_tr, deduped_hugging_csvs_v2_251117_str
python -m src.data_symlink.trick_aug --repo_root /u1/z6dong/Repo/ModelTables/data/processed --mode tr --tag 251117 --v2_mode > logs/trick_aug_tr_v2_251117.log 2>&1   
python -m src.data_symlink.trick_aug --repo_root /u1/z6dong/Repo/ModelTables/data/processed --mode str --tag 251117 --v2_mode > logs/trick_aug_str_v2_251117.log 2>&1  
# Step 2: Create symlinks from ModelTables to starmie_internal/data/scilake_final_<tag>/datalake
# python -m src.data_symlink.ln_scilake --repo_root /u1/z6dong/Repo --mode base --tag 251117 --v2_mode > logs/ln_scilake_base_251117.log 2>&1  
# python -m src.data_symlink.ln_scilake --repo_root /u1/z6dong/Repo --mode str --tag 251117 --v2_mode > logs/ln_scilake_str_251117.log 2>&1  
# python -m src.data_symlink.ln_scilake --repo_root /u1/z6dong/Repo --mode tr --tag 251117 --v2_mode > logs/ln_scilake_tr_251117.log 2>&1 
python -m src.data_symlink.ln_scilake_new --repo_root /u1/z6dong/Repo --tag 251117 --v2_mode --n_jobs 32 > logs/ln_scilake_new_251117.log 2>&1
```

### 6\. Run Updated Starmie Scripts

Execute Starmie's pipeline for contrastive learning, embedding extraction, and search

```bash
python -m src.data_symlink.prepare_sample --tag 251117 --v2_mode --root_dir /u1/z6dong/Repo --output_file data/analysis/scilake_final_filelist_v2_251117.txt --limit 1000 --seed 42 > logs/prepare_sample_v2_251117.log 2>&1
# hands to starmie
bash scripts/step1_pretrain.sh > logs/step1_pretrain.log 2>&1  # Fine-tune contrastive learning model
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
```

### 7\. Baseline: Dense Search, Sparse Search, Hybrid Search

Run baseline table embedding and retrieval methods for comparison, for faiss cpu/gpu installation, see [FAISS GitHub repository](https://github.com/facebookresearch/faiss).

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
# Note: Requires pyserini and Java/JDK (for pyserini to work)
# Recommended conda environment: faiss_gpu_env (or any environment with pyserini installed)
# Output: data/tmp/baseline2_sparse_results_251117.json
TAG=251117 bash src/baseline2/get_metadata.sh > logs/baseline2_get_metadata_251117.log 2>&1 # Baseline2: Sparse search get metadata
TAG=251117 bash src/baseline2/sparse_search.sh > logs/baseline2_sparse_search_251117.log 2>&1 # Baseline2: Sparse search

### 3. Baseline3: Hybrid (Sparse + Dense search)
# Note: Hybrid search uses Python scripts with command-line arguments
# Requires: 
#   - Sparse index from Baseline2: data/tmp/index_251117
#   - Dense index directory: data/tmp/index_dense_251117/ (must contain index.faiss or index file)
#   - To create dense index, first encode corpus and build faiss index:
#     mkdir -p data/tmp/index_dense_251117
#     python src/baseline1/table_retrieval_pipeline.py encode \
#       --jsonl data/tmp/corpus/collection.jsonl \
#       --model_name sentence-transformers/all-MiniLM-L6-v2 \
#       --batch_size 256 --output_npz data/tmp/index_dense_251117/embeddings.npz --device cuda
#     python src/baseline1/table_retrieval_pipeline.py build_faiss \
#       --emb_npz data/tmp/index_dense_251117/embeddings.npz \
#       --output_index data/tmp/index_dense_251117/index.faiss
# Output: data/tmp/search_result_hybrid_251117.json (then postprocess to baseline3_hybrid_results_251117.json)
TAG=251117 python src/baseline2/search_with_pyserini_hybrid.py \
  --sparse-index data/tmp/index_251117 \
  --dense-index data/tmp/index_dense_251117 \
  --queries data/tmp/queries_table.tsv \
  --mapping data/tmp/queries_table_mapping.json \
  --k 11 --alpha 0.45 --device cpu > logs/baseline2_hybrid_search_251117.log 2>&1
# Postprocess hybrid results (if postprocess.py supports hybrid results)
# TAG=251117 python src/baseline2/postprocess.py \
#   --input data/tmp/search_result_hybrid_251117.json \
#   --output data/tmp/baseline3_hybrid_results_251117.json \
#   --top1-list data/tmp/hybrid_queries_with_top1_matches.txt
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

### 8\. Figure post-analysis
```bash
# figure1 + 2: qc_stats_fig + gt_distri's output figures
# figure3: count v.s. time. Based on step3_dedup_<tag>.parquet, all_title_list_valid_<tag>.parquet, output table_model_counts_over_time_<tag>.pdf/png
python -m src.data_analysis.table_model_counts_over_time --tag 251117 --v2_mode > logs/table_model_counts_over_time_v2_251117.log 2>&1  # step3_dedup_<tag>.parquet, all_title_list_valid_<tag>.parquet, output table_model_counts_over_time_<tag>.pdf/png

# step by step filtering img
python -m src.data_analysis.card_statistics --tag 251117 > logs/card_statistics_251117.log 2>&1 # get statistics of model cards
python -m src.data_analysis.hf_models_analysis --tag 251117 --v2_mode > logs/hf_models_analysis_v2_251117.log 2>&1 # get statistics of models in Hugging Face: hf_models_analysis.png and hf_cross_analysis.png
python -m src.data_analysis.model_snapshot_overlap > logs/model_snapshot_overlap_251117.log 2>&1 # compare modelId overlap between two fixed snapshots: V1 (no tag, no v2) vs V2 (tag 251117 + v2): data/analysis/model_snapshot_overlap.png

python -m src.data_analysis.align_tables_output_versions --dir-a data/processed/tables_output --dir-b data/processed/tables_output_v2_251117 > logs/align_tables_output_arxiv.log 2>&1  
python -m src.data_analysis.compare_tables_by_content 2503.03556v1 > logs/compare_tables_by_content.log 2>&1 # compare tables by content for base id 2409.19581

# after carefully examining
python -m src.data_analysis.filtered_gt_visualization > logs/filtered_gt_visualization.log 2>&1
python -m src.data_analysis.quick_visualization_final > logs/quick_visualization_final.log 2>&1

# get relational keys from other key automatically (require logs/parquet_schema.log)
python -m src.data_analysis.get_from --target html_table_list_mapped_dedup --source modelId --value google-bert/bert-base-uncased > logs/get_from.log 2>&1
python -m src.data_analysis.get_from --target readme_path --source csv_paths --value "64dc62e53f_table2.csv" >> logs/get_from.log 2>&1
python -m src.data_analysis.get_from --target modelId --source hugging_table_list --value data/processed/deduped_hugging_csvs/021f09961f_table1.csv >> logs/get_from.log 2>&1
python -m src.data_analysis.get_from --target modelId --source pdf_link --value https://arxiv.org/pdf/0803.1019 >> logs/get_from.log 2>&1
# or 
python -m src.postprocess.relational_parquet_strategies --tag 251117 --v2_mode > logs/relational_parquet_strategies_251117.log 2>&1  # 
```

```bash
python -m src.data_analysis.valid_table_shapes --tag 251117 --v2_mode > logs/valid_table_shapes_v2_251117.log 2>&1  # table shapes from valid table list, Input: all_valid_title_valid_v2_<tag>.txt, Output: valid_table_shapes_v2_<tag>.parquet; We double check the qc_stats filtering
#could execute sql on valid_table_shapes.parquet, get anomaly tables with extremely large rows or columns
python -m src.data_analysis.table_usage_stats --tag 251117 --v2_mode > logs/table_usage_stats_v2_251117.log 2>&1  # table usage value counts, Input: valid_table_shapes_v2_<tag>.parquet, Output: table_usage_stats_v2_<tag>.parquet
```
