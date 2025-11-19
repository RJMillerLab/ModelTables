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
# Download modelcards
python -m src.data_preprocess.download_hf_dataset --date 251117 --type modelcard/datasetcard
```

### 1\. Parse Initial Elements

This step extracts key metadata from model cards and associated links.
```bash
# Split readme and tags, parse URLs, parse BibTeX entries.
# Output: ['modelId', 'author', 'last_modified', 'downloads', 'likes', 'library_name', 'tags', 'pipeline_tag', 'createdAt', 'card', 'card_tags', 'card_readme', 'pdf_link', 'github_link', 'all_links', 'extracted_bibtex', 'extracted_bibtex_tuple', 'parsed_bibtex_tuple_list', 'successful_parse_count']
# (Optional)
#python -m src.data_preprocess.load_raw_to_db sqlite/duckdb # save raw to DuckDB, but will explode the memory
python -m src.data_preprocess.step1_parse --raw-date 251117 --versioning --baseline-step1 data/processed/modelcard_step1.parquet
# or 
python -m src.data_preprocess.step1_parse --raw-date 251117
python -m src.data_preprocess.step1_down_giturl --tag 251117 --versioning --baseline-cache data/processed/github_readme_cache.parquet # Download GitHub READMEs and HTMLs from extracted URLs; Input: modelcard_step1.parquet, Output: giturl_info.parquet, downloaded_github_readmes/
#python -m src.data_preprocess.step1_down_giturl_fake # if program has leakage but finished downloading, then re-run this code to save final parquet and cache files.
find data/downloaded_github_readmes -type f -exec stat -f "%z %N" {} + | sort -nr | head -n 50 | awk '{printf "%.2f MB %s\n", $1/1024/1024, $2}' # some readme files are too large
# Query specific GitHub URL content (example). Input: local path to a downloaded README, Output: URL content
python -m src.data_analysis.query_giturl load --query "data/downloaded_github_readmes/0a0c3d247213c087eb2472c3fe387292.md" # sql. # (Optional)
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
python -m src.data_preprocess.step2_hugging_github_extract --tag 251117

# Process downloaded GitHub HTML files to Markdown.
# Input: data/downloaded_github_readmes_<tag>/
# Output: data/downloaded_github_readmes_<tag>_processed/, data/processed/md_parsing_results_v2_<tag>.parquet
python -m src.data_preprocess.step2_git_md2text --tag 251117
#python -m src.data_preprocess.step2_git_md2text_v2 --n_jobs 8 --output_dir data/processed/md_processed_v2 --save_mode csv/duckdb/sqlite
# Extract titles from arXiv and GitHub URLs (not S2ORC). For BibTeX entries and PDF URLs.
# Input: modelcard_step1_<tag>.parquet, github_readme_cache_<tag>.parquet, downloaded_github_readmes_<tag>_processed/, PDF/GitHub URLs
# Output: modelcard_all_title_list_<tag>.parquet, github_readme_cache_update_<tag>.parquet, github_extraction_cache_<tag>.json, all_links_with_category_<tag>.csv
python -m src.data_preprocess.step2_arxiv_github_title --tag 251117

# Save deduplicated titles for querying Semantic Scholar (S2ORC).
# Input: modelcard_all_title_list_<tag>.parquet
# Output: modelcard_dedup_titles_<tag>.json, modelcard_title_query_results_<tag>.json, modelcard_all_title_list_mapped_<tag>.parquet
python -m src.data_preprocess.step2_s2orc_save --tag 251117

<details>
<summary>Optional: LLM/S2ORC pipelines (currently skipped)</summary>
#### Option1:
# Query Semantic Scholar API for citation information (alternative to local database if no key, but may hit rate limits).
# Input: modelcard_dedup_titles.json
# Output: s2orc_query_results.parquet, s2orc_citations_cache.parquet, s2orc_references_cache.parquet, s2orc_titles2ids.parquet
python -m src.data_preprocess.s2orc_API_query > logs/s2orc_API_query_v3_429.log
 - python -m src.data_preprocess.s2orc_log_429 # incase some title meet 429 error (API rate error)
 - python -m src.data_preprocess.s2orc_retry_missing # make up for the missing items
 - python -m src.data_preprocess.s2orc_merge # parse the references and citations from retrieved results | I: s2orc*.parquet, O: s2orc_rerun.parquet
 - bash src/data_localindexing/build_mini_citation_es.sh # I: xx | O: batch_results
# (Deprecate: Old method) bash src.data_localindexing/build_mini_citation_es.sh
# Extract full records from batch query results.
# Input: batch_results + hit_ids.txt
# Output: full_hits.jsonl
python -m src.data_localindexing.extract_full_records
# Merge extracted full records.
# Input: full_hits.jsonl
# Output: s2orc_citations_cache, s2orc_references_cache, s2orc_query_results
python -m src.data_localindexing.extract_full_records_to_merge
- python -m src.data_preprocess.s2orc_merge # I: s2orc*.parquet, O: s2orc_rerun.parquet
 # (deprecate) - bash src/data_localindexing/build_mini_s2orc_es.sh # choose dump data to setup and batch query |
  # I: paper_index_mini.db, modelcard_dedup_titles.json → O: Elasticsearch index (e.g., papers_index), query_cache.parquet
 - bash src/data_preprocess/step2_se_url_tab.sh # extract fulltext -> ref/cit info
# I: query_cache.parquet/s2orc_rerun.parquet, paper_index_mini.db, NDJSON files in /se_s2orc_250218 → O: extracted_annotations.parquet, tmp_merged_df.parquet, tmp_extracted_lines.parquet
</details>

# Download arXiv HTML content for table extraction.
# Input: extracted_annotations_<tag>.parquet, arxiv_titles_cache_<tag>.json
# Output: title2arxiv_new_cache_<tag>.json, arxiv_html_cache_<tag>.json, missing_titles_tmp_<tag>.txt, arxiv_fulltext_html_<tag>/*.html
# cp -r /Users/doradong/Repo/CitationLake/data/processed/extracted_annotations.parquet /Users/doradong/Repo/CitationLake/data/processed/extracted_annotations_251117.parquet
python -m src.data_preprocess.step2_arxiv_get_html --tag 251117

# Extract tables from arXiv HTML files.
# Input: arxiv_html_cache.json, arxiv_fulltext_html/*.html, html_table.parquet (optional)
# Output: html_table.parquet, tables_output/*.csv
python -m src.data_preprocess.step2_arxiv_parse --tag 251117
python -m src.data_preprocess.step2_arxiv_parse_v2 --n_jobs 16 --output_dir data/processed/tables_output_v2_251117 --tag 251117 --save_mode csv  #/duckdb/sqlite 

mkdir logs
# Integrate all processed table data (arXiv HTML + S2ORC extracted annotations) and process with LLM.
# Input: title2arxiv_new_cache_<tag>.json, html_table_<tag>.parquet/html_parsing_results_v2_<tag>.parquet, extracted_annotations_<tag>.parquet, pdf_download_cache_<tag>.json
# Output: batch_input_<tag>.jsonl, batch_output_<tag>.jsonl, llm_markdown_table_results_<tag>.parquet
python -m src.data_preprocess.step2_integration_s2orc_llm --tag 251117 > logs/step2_integration_s2orc_llm_251117.log
# (Optional) Check OpenAI batch job status (if using LLM for table processing)
bash src/data_preprocess/openai_batchjob_status.sh

# (Optional) If the sequence is wrong, reproduce from the log...
#python -m src.data_preprocess.quick_repro
#cp -r llm_outputs/llm_markdown_table_results_aligned.parquet llm_outputs/llm_markdown_table_results.parquet

# Save LLM-processed tables into local CSVs.
# Input: llm_markdown_table_results_<tag>.parquet
# Output: llm_tables_<tag>/*.csv, final_integration_with_paths_v2_<tag>.parquet
python -m src.data_preprocess.step2_llm_save --tag 251117 > logs/step2_llm_save_251117.log
```

### 4\. Label Ground Truth for Unionable Search Baselines


This section details the process of generating ground truth labels for table unionability.
```bash
python -m src.data_preprocess.step2_merge_tables --tag 251117  # Merge all table lists from 4 resources (HuggingFace, GitHub, HTML, LLM) into a unified model ID file. Input: final_integration_with_paths_v2_<tag>.parquet, modelcard_all_title_list_<tag>.parquet, modelcard_step2_v2_<tag>.parquet. Output: modelcard_step3_merged_v2_<tag>.parquet
python -m src.data_gt.paper_citation_overlap --tag 251117  # Compute paper-pair citation overlap scores for ground truth. Input: extracted_annotations_<tag>.parquet. Output: modelcard_citation_all_matrices_<tag>.pkl.gz (REQUIRED for step3_gt)
python -m src.data_analysis.paper_relatedness_distribution --tag 251117  # (Optional) Plot violin figures of paper relatedness distribution. Input: modelcard_citation_all_matrices_<tag>.pkl.gz. Output: overlap_violin_by_mode_<tag>.pdf
python -m src.data_analysis.paper_relatedness_threshold --tag 251117  # (Optional) Determine paper relatedness thresholds. Input: modelcard_citation_all_matrices_<tag>.pkl.gz. Output: score_*.pdf files in data/analysis/
```

### Quality Control \!\!\! | Run some analysis

Ensure data quality and consistency before generating final ground truth.

```bash
python -m src.data_preprocess.step2_dedup_tables --tag 251117 > logs/step2_dedup_tables_251117.log  # Deduplicate raw tables, prioritizing Hugging Face > GitHub > HTML > LLM. Input: modelcard_step3_merged_v2_<tag>.parquet. Output: modelcard_step3_dedup_v2_<tag>.parquet
python -m src.data_analysis.qc_dedup_fig --tag 251117  # Generate heatmaps from dedup results. Input: deduped_<tag>/dup_matrix.pkl, deduped_<tag>/stats.json. Output: heatmaps in data/analysis/
python -m src.data_analysis.qc_stats --tag 251117 > logs/qc_stats_251117.log  # Print table #rows #cols. Input: modelcard_step3_dedup_v2_<tag>.parquet. Output: benchmark_results_v2_<tag>.parquet
python -m src.data_analysis.qc_stats_fig --tag 251117  # Plot benchmark results. Input: benchmark_results_v2_<tag>.parquet. Output: benchmark_metrics_vertical_v2_<tag>.pdf/png
python -m src.data_analysis.qc_anomaly --recursive #(Optional)
python -m src.tools.show_table_diff_md 0ae65809ffffa20a2e5ead861e7408ac_table_0.csv #(Optional) # compare v1 and v2 table diff
# (Optional) Double-check deduplication and mapping logic.
# python -m src.data_analysis.qc_dc
# Obtain file counts directly from folders to verify against statistics.
bash src/data_analysis/count_files.sh
```

### Final Ground Truth Generation
Generate the definitive ground truth files for evaluation.

```bash
# (Depre) python -m src.data_gt.step3_create_symlinks --tag 251117  # Create symbolic links for organizing processed tables. Input: modelcard_step3_dedup_v2_<tag>.parquet. Output: modelcard_step4_v2_<tag>.parquet, sym_*_csvs_* (symbolic links)
bash src/data_gt/step3_gt.sh 251117  # Build ground truth (paper-level, model-level, dataset-level). Input: modelcard_citation_all_matrices_<tag>.pkl.gz, modelcard_step3_dedup_v2_<tag>.parquet, final_integration_with_paths_v2_<tag>.parquet, modelcard_all_title_list_<tag>.parquet. Output: data/gt/* (no versioning)
python -m src.tools.check_gt_coverage --csv-name 1910.09700_table0.csv --levels direct --mode both # (Optional)
python -m src.data_gt.debug_npz --gt-dir data/gt/ # (Optional) Debug NPZ ground truth files to ensure valid conditions.
# Process SQLite ground truth into pickle files (if applicable from other benchmarks).
python -m src.data_localindexing.turn_tus_into_pickle
# (deprecate) python -m src.data_gt.gt_combine
python -m src.data_gt.modelcard_matrix --tag 251117  # Add other two levels of citation graphs (modelcard and dataset). Input: modelcard_step1_<tag>.parquet, modelcard_step3_dedup_v2_<tag>.parquet, modelcard_step3_merged_v2_<tag>.parquet. Output: modelcard_gt_related_model_<tag>.parquet, data/gt/scilake_gt_modellink_*_<tag>.npz
python -m src.data_gt.merge_union --level direct --tag 251117  # Merge union ground truth. Input: data/gt/*_<tag>.npz, *_<tag>.pkl. Output: data/gt/csv_pair_union_*_<tag>_processed.npz
python -m src.data_analysis.gt_distri --tag 251117  # Plot GT length distribution (boxplot/violin). Input: data/gt/*_<tag>.npz
python -m src.data_gt.nonzeroedge --gt_dir data/gt --tag 251117  # Compute non-zero edge statistics for citation graphs. Input: data/gt/*_<tag>.npz
# (test)python -m src.data_gt.test_modelcard_update --mode dataset # check whether matrix multiplication and for loop obtain the same results
#(test)python -m src.data_gt.convert_adj_to_npz --input data/gt/scilake_gt_modellink_dataset_adj_processed.pkl --output-prefix data/gt/scilake_gt_modellink_dataset # pkl2npz
python -m src.data_gt.create_csvlist_variants --level direct --tag 251117  # Update CSV lists for various ground truth variants. Input: data/gt/*_<tag>.pkl
# (depreate) python -m src.data_gt.create_gt_variants data/gt/csv_pair_adj_overlap_rate_processed.pkl # produce _s, _t, _s_t for pkl files
# (deprecate) python -m src.data_gt.print_relations_stats data/tmp/relations_all.pkl # print stats for matrix
# (deprecate) python -m src.data_analysis.gt_fig # plot stats
```

### 5\. Create Symlinks for Starmie Integration

Prepare data and augmentations for integration with the Starmie benchmark framework.

```bash
# go to starmie folder, and copy this sh file to run 
python -m src.data_symlink.trick_aug --repo_root /u1/z6dong/Repo --mode str/transpose/str_transpose # trick: header-str(value)
python -m src.data_symlink.ln_scilake --repo_root /u1/z6dong/Repo --mode str/tr/tr_str --dir-name scilake_final_v2_str/tr/tr_str
python -m src.data_symlink.ln_scilake --repo_root /u1/z6dong/Repo --mode base --dir-name scilake_final_v2 # symlink csvs to the target folder
# bash src/data_analysis/count_files.sh check whether the symlink path include some files
```

### 6\. Run Updated Starmie Scripts

Execute Starmie's pipeline for contrastive learning, embedding extraction, and search

```bash
# bash prepare_sample.sh  # Sample 1000 tables from each resource folder for evaluation
# python -m src.data_symlink.prepare_sample_server --root_dir /u4/z6dong/Repo --output scilake_final --output_file scilake_final_filelist.txt --limit 2000 --seed 42  # Alternative for server
python -m src.data_symlink.prepare_sample --root_dir /u1/z6dong/Repo --output_file scilake_final_filelist.txt --limit 1000 --seed 42  # Another substitution
# python -m src.data_symlink.prepare_sample_tricks --input_file scilake_final_filelist.txt  # Create file lists for trick-augmented files (Input: scilake_final_filelist.txt, Output: scilake_final_filelist_{tricks}_filelist.txt)
python -m src.data_symlink.ln_scilake_final_link --filelist scilake_final_filelist.txt scilake_final_filelist_val.txt  # Create validation file lists
# bash check_empty.sh  # (deprecate) (already processed in QC step) filter out empty files (or low quality files later)
bash scripts/step1_pretrain.sh  # Fine-tune contrastive learning model
bash scripts/step2_extractvectors.sh  # Encode embeddings for query and datalake items
bash scripts/step3_search_hnsw.sh  # Perform data lake search (retrieval)
bash scripts/step3_processmetrics.sh  # Extract metrics based on ground truth and retrieval results; plot figures
bash eval_per_resource.sh  # Run ablation study on different resources (after getting results)
# bash eval_per_resource.sh  # (Alternatively, run before getting results)
# bash scripts/step4_discovery.sh  # (Optional)
```

### 7\. Baseline1: Dense Search

Run baseline table embedding and retrieval methods for comparison.

```bash
bash src/baseline1/table_retrieval_pipeline.sh  # build corpus jsonl/encode(SBERT)/build faiss/search/postprocess
bash src/baseline1/table_retrieval_pipeline_str.sh  # for augmented tables: step1 get embedding
bash src/baseline1/table_retrieval_pipeline_tr.sh  # for augmented tables: step1 get embedding
bash src/baseline1/combine_embedding.sh  # for augmented ablation studies: step2 combine embedding and jsonl for ori+tr, ori+str, ori+tr+str
bash src/baseline1/build_aug_faiss.sh  # step3: build faiss
bash src/baseline1/augment_search.sh  # step4: search
bash src/baseline1/postprocess_general.sh  # step5: postprocess: split into ori / tr / str json
bash src/baseline1/standardize_filenames.sh  # step6: postprocess: all files back to ori csv name
bash scripts/step3_processmetrics_all.sh <index>  # compute metrics under starmie: run baseline metrics computation
```
for faiss cpu/gpu installation, see [FAISS GitHub repository](https://github.com/facebookresearch/faiss).

____
### 8. Baseline2: Sparse search, Baseline3: Hybrid (Sparse + Dense search)
```bash
bash src/baseline2/get_metadata.sh # Baseline2: Sparse search get metadata
bash src/baseline2/sparse_search.sh # Baseline2: Sparse search
bash src/baseline2/hybrid_search.sh # Baseline3: Hybrid (Sparse + Dense search)
```

<!-- ### 10. Model Search - Dense first: 
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
<summary><strong>11. GPT Evaluation of Table Relatedness and Model Relatedness</strong></summary>

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
    --seed 42
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
python src/gpt_evaluation/visualize_sampling_2x4_horizontal.py
```

#### Step2: Query OpenRouter for table relatedness evaluation.
```bash
python -m src.gpt_evaluation.step2_query_openrouter --input output/gpt_evaluation/table_1M_fix_unique_pairs.jsonl --output output/gpt_evaluation/step2_full_198.jsonl 2>&1 | tee step2_full.log &
# retry
# python -m src.gpt_evaluation.step2_retry_failed \
#     --input output/gpt_evaluation/step2_openrouter_results_full.jsonl \
#     --output output/gpt_evaluation/step2_openrouter_results_retried.jsonl
# add merge please
python -m src.gpt_evaluation.step2_merge_results --main output/gpt_evaluation/step2_full_198.jsonl --additional output/gpt_evaluation/step2_gpt4mini_full.jsonl --output output/gpt_evaluation/step2_all_5models.jsonl

# crowdsourcing metrics analysis
python -m src.gpt_evaluation.visualize_crowdsourcing_metrics # generate figures for subset
python -m src.gpt_evaluation.visualize_crowdsourcing_metrics_full # generate figures for full dataset
```

</details>

---

### 2. Model Relatedness Sampling

**Script**: `src/gpt_evaluation/step1_model_sampling.py`

**Purpose**: Sample model pairs for model relatedness evaluation.

**Usage**:
```bash
python src/gpt_evaluation/step1_model_sampling.py --n-samples 200 --seed 42
```


### 12. Table Integration:
```bash

```

### Analysis on Results

Tools for analyzing the retrieval results and ground truth.

```bash
# Get top-10 results from step3_search_hnsw.
python -m src.data_analysis.report_generation --json_path ~/Repo/starmie_internal/tmp/test_hnsw_search_scilake_large_full.json
# then gen markdown
python -m src.data_analysis.report_generation --json_path data/baseline/baseline1_dense.json --query_table 1810.04805_table4.csv
# --show_model_titles

# Check if a specific CSV pair is related in the ground truth.
python -m src.data_gt.check_pair_in_gt --gt-dir data/gt --csv1 0ab2d85d37_table1.csv --csv2 096d51652d_table1.csv

# Count unique CSVs in retrieval results.
python count_unique_csvs.py --results /u1/z6dong/Repo/starmie_internal/results/scilake_final/test_hnsw_search_drop_cell_tfidf_entity_full.json --gt /u1/z6dong/Repo/ModelLake/data/gt/csv_pair_adj_overlap_rate_processed.pkl

# CSV to ModelID Mapping
# Get modelIDs from CSV files (supports GitHub, HuggingFace, HTML, LLM sources)
python batch_process_tables.py -i tmp/top_tables.txt -o tmp/top_tables_with_keywords.csv
# Analyze HTML table files and compare column counts between v1 and v2
# Input: tmp/top_tables_with_keywords.csv (filtered for HTML source)
# Output: tmp/html_table_analysis.csv with column count comparison
# Purpose: Compare HTML table v2 vs v1 column counts and analyze reduction
python src/data_analysis/analyze_html_tables.py
python src/data_analysis/analyze_huggingface_tables.py
```

### Additional Statistics Analysis

Scripts for generating further insights into the dataset.

```bash
python -m src.data_preprocess.step1_analysis # get analysis on proportion of different links (sql)
python -m src.data_preprocess.query_compare_API_local # compare the query results among local and API from s2orc
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
python src.data_analysis.list_parquet_schemas > logs/parquet_schema.log    
# get attributes duplicate analysis
#python complete_duplicate_analysis.py > logs/complete_duplicate_analysis_results.txt
python src.data_analysis.column_size_analysis --include-modelid > logs/parquet_storage.log

# load all files into duckdb
python src.data_analysis.load_sv_to_db --engine sqlite --db-path deduped_hugging_csvs_v2.sqlite --input-dir data/processed/deduped_hugging_csvs_v2
# get relational keys from other key automatically (require logs/parquet_schema.log)
python -m src.data_analysis.get_from --target html_table_list_mapped_dedup --source modelId --value google-bert/bert-base-uncased
python -m src.data_analysis.get_from --target readme_path --source csv_paths --value "64dc62e53f_table2.csv" 
python -m src.data_analysis.get_from --target modelId --source hugging_table_list --value data/processed/deduped_hugging_csvs/021f09961f_table1.csv
python -m src.data_analysis.get_from --target modelId --source pdf_link --value https://arxiv.org/pdf/0803.1019
python get_modelid_from_arxiv_comprehensive.py "0803.1019" --search-all --debug
# step by step filtering img
python -m src.data_analysis.card_statistics # get statistics of model cards
python -m src.data_analysis.hf_models_analysis # get statistics of models in Hugging Face
# after carefully examining
python -m src.data_analysis.filtered_gt_visualization
python -m src.data_analysis.quick_visualization_final
# the count v.s. time visualization
python src/data_analysis/table_model_counts_over_time.py --use-v2 --output-dir data/analysis
# TODO: top-10!
```

```bash
# periodically upload to hugging face : Files: modelcard_step3_dedup_v2_<tag>.parquet, hugging, github, html, llm.zip
# huggingface-cli login
# huggingface-cli repo create your-username/your-dataset --type dataset
python -m src.data_preprocess.upload_to_hf_dataset --tag 251117 --dataset-name your-username/your-dataset --no-dry-run
# TODO: versioning for zip
```
