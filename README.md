# CitationLake
## Features
- Extract text, figures, and tables from PDFs and links.
- Build and manage citation graphs.
- Define and execute tasks such as model retrieving, table integration, and data lake operations.
- User-friendly UI with pipeline visualization and task management.# CitationLake

## Requirements

```bash
git clone https://github.com/DoraDong-2023/CitationLake.git
cd CitationLake/
pip install -r requirements.txt
```

```bash
echo "OPENAI_API_KEY='your_api_key'" > .env
# echo "SEMANTIC_SCHOLAR_API_KEY='your_api_key'" > .env # Optional, use this to download semantic scholar dataset, and faster querying 
```


## Download Data

This project uses datasets hosted on Hugging Face. Use the following commands to download the necessary data:
```bash
mkdir data
mkdir data/processed
mkdir data/raw
# Downloading huggingface model card dataset
git lfs install
git clone https://huggingface.co/datasets/librarian-bots/model_cards_with_metadata data/raw
git clone https://huggingface.co/datasets/librarian-bots/dataset_cards_with_metadata data/raw
# Downloading semantic scholar data
# TODO: add command from privatecommonscript to here, for downloading the semantic scholar here
```

---

## Scripts

1. Parse every elements required:
```bash
# Here I -> Input, O -> Output | skip writting data/processed
python -m src.data_preprocess.step1 # Split readme and tags, parse urls, parse bibtex, 
# get:  ['modelId', 'author', 'last_modified', 'downloads', 'likes', 'library_name', 'tags', 'pipeline_tag', 'createdAt', 'card', 'card_tags', 'card_readme', 'pdf_link', 'github_link', 'all_links', 'extracted_bibtex', 'extracted_bibtex_tuple', 'parsed_bibtex_tuple_list', 'successful_parse_count']
python -m src.data_preprocess.step1_down_giturl # Download github URL README & HTMLs. I: modelcard_step1.parquet, O: giturl_info.parquet
#python -m src.data_preprocess.step1_down_giturl_fake # if program has leakage but finished downloading, then re-run this code to save final parquet and cache files.
find data/downloaded_github_readmes -type f -exec stat -f "%z %N" {} + | sort -nr | head -n 50 | awk '{printf "%.2f MB %s\n", $1/1024/1024, $2}' # some readme files are too large, we fix this issue
python -m src.data_preprocess.step1_query_giturl load --query data/downloaded_github_readmes/7166b2cb378b3740c3b212bc0657dd11.md # Input: local path, Output: URL
```

2. Download and build database for faster querying:
```bash
# TODO: add command from privatecommonscript to here, for downloading the semantic scholar here
# Requirement: cd to the path of downloaded dataset, e.g.: cd ~/shared_data/se_s2orc_250218
python -m src.data_localindexing.build_mini_s2orc build --directory /u4/z6dong/shared_data/se_s2orc_250218/ # After downloading semantic scholar dataset, build database based on it.
python -m src.data_localindexing.build_mini_s2orc query --title "BioMANIA: Simplifying bioinformatics data analysis through conversation" --directory /u4/z6dong/shared_data/se_s2orc_250218/ # After building up database, query title based on db file.

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

3. Extract tables to local folder:
```bash
python -m src.data_preprocess.step2_gitcard_tab # extract table from git + modelcards | save csvs to folder
# I: data/processed + modelcard_step1.parquet, github_readmes_info.parquet, downloaded_github_readmes/, config.yaml | O: modelcard_step2.parquet, deduped_hugging_csvs/, hugging_deduped_mapping.json, deduped_github_csvs/, md_to_csv_mapping.json
python -m src.data_preprocess.step2_md2text # process downloaded github html (if any) to markdown
# I: data/downloaded_github_readmes/ → O: data/downloaded_github_readmes_processed/
python -m src.data_preprocess.step2_se_url_title # fetching title from bibtex, PDF url.
# I: modelcard_step1.parquet, github_readme_cache.parquet, github_readmes_processed/, pdf_link + github_link URLs → O: modelcard_all_title_list.parquet, github_readme_cache_update.parquet, github_extraction_cache.json, all_links_with_category.csv
python -m src.data_preprocess.step2_se_url_save # save the deduplicate titles
# I: modelcard_all_title_list.parquet → O: modelcard_dedup_titles.json, modelcard_title_query_results.json, modelcard_all_title_list_mapped.parquet
 - python -m src.data_preprocess.s2orc_API_query > logs/s2orc_API_query_v3_429.log # choose API to query |
 # I: modelcard_dedup_titles.json O: s2orc_query_results.parquet, s2orc_citations_cache.parquet, s2orc_references_cache.parquet, s2orc_titles2ids.parquet
 - python -m src.data_preprocess.s2orc_log_429 # incase some title meet 429 error (API rate error)
 - python -m src.data_preprocess.s2orc_retry_missing # make up for the missing items
 - python -m src.data_preprocess.s2orc_merge # parse the references and citations from retrieved results | I: s2orc*.parquet, O: s2orc_rerun.parquet

 - bash src/data_localindexing/build_mini_s2orc_es.sh # choose dump data to setup and batch query |
  # I: paper_index_mini.db, modelcard_dedup_titles.json → O: Elasticsearch index (e.g., papers_index), query_cache.parquet
 - bash src/data_preprocess/step2_se_url_tab.sh # extract fulltext & openaccessurl | use title/id to fetch table from s2orc
# I: query_cache.parquet/s2orc_rerun.parquet, paper_index_mini.db, NDJSON files in /se_s2orc_250218 → O: extracted_annotations.parquet, merged_df.parquet
python -m src.data_preprocess.step2_get_html # download html
# I: extracted_annotations.parquet, arxiv_titles_cache.json → O: title2arxiv_new_cache.json, arxiv_html_cache.json, missing_titles_tmp.txt, arxiv_fulltext_html/*.html
python -m src.data_preprocess.step2_html_parsing # extract tables from html
# I: arxiv_html_cache.json, arxiv_fulltext_html/*.html, html_table.parquet (optional) → O: html_table.parquet, tables_output/*.csv
mkdir logs
TODO: python -m src.data_preprocess.step2_integration_order > logs/step2_integration_order_0414.log # first html | then PDF? (no) | finally llm polished table text
# I: title2arxiv_new_cache.json, html_table.parquet, extracted_annotations.parquet, pdf_download_cache.json → O: before_llm_output.parquet, batch_input.jsonl, batch_output.jsonl, llm_markdown_table_results.parquet
bash src/data_preprocess/openai_batchjob_status.sh # query batch job status
# (Optional) If the sequence is wrong, reproduce from the log...
#python -m src.data_preprocess.quick_repro
#cp -r llm_outputs/llm_markdown_table_results_aligned.parquet llm_outputs/llm_markdown_table_results.parquet
python -m src.data_preprocess.step2_llm_save > logs/step2_llm_save.log # save table into local# csv
# I: llm_markdown_table_results.parquet → O: llm_tables/*.csv, final_integration_with_paths.parquet
```

4. Label groundtruth for unionable search baselines:

```bash
python -m src.data_gt.step3_pre_merge # merge all the table list into modelid file
# I: final_integration_with_paths.parquet, modelcard_all_title_list.parquet → O: modelcard_step3_merged.parquet
# (Only need if we not run s2orc_API_query) python -m src.data_gt.step3_API_query # paper level: get citations relation by API | Tips: notify the timing issue, this is the updated real-time query, your local corpus data might be outdated
# I: final_integration_with_paths.parquet. O: modelcard_citation_enriched.parquet
# (Optional: only need when build db locally) bash src/data_localindexing/build_mini_citation_es.sh # build up citation graph | Then get citations relation from graph edge .db
python -m src.data_gt.step3_overlap_rate # paper level: compute paper-pair overlap score | 
# I: extracted_annotations/modelcard_citation_enriched O: modelcard_rate/label.pickle
python -m src.data_analysis.overlap_fig # plot stats
python -m src.data_gt.overlap # check whether thresholding for overlapping rate is reasonable | I: pickle from step3_overlap_rate
```
Quality Control !!! | Run some analysis
```bash
# this must be run before gt
python -m src.data_analysis.qc_dedup > logs/qc_dedup.log # dedup raw tables, keep dedup in order hugging>github>html>llm | I: modelcard_step3_merged, O: modelcard_step3_dedup
python -m src.data_analysis.qc_dedup_fig
python -m src.data_analysis.qc_stats # stats | I: modelcard_step4_dedup, O: benchmark_results
python -m src.data_analysis.qc_stats_fig
# (Optional) python -m src.data_analysis.qc_dc # double check whether the dedup and mapped logic is correct
bash src/data_analysis/count_files.sh # obtain files count directly from folder | The count above should be smaller than files here.
```
Final gt!
```bash
python -m src.data_gt.step3_create_symlinks # create the symbolic link on different device
# I: modelcard_step3_dedup → O: modelcard_step4 + sym_*_csvs_*
python -m src.data_gt.step3_gt # build up groundtruth
python -m src.data_gt.modelcard_matrix --stats 0 # (add modelcard level citation graph)
python -m src.data_gt.print_relations_stats data/tmp/relations_all.pkl # print stats for matrix

python -m src.data_analysis.gt_fig # plot stats
```
5. Create symlink for combining them into starmie/data/scilake_large/datalake/*
```bash
# go to starmie folder, and copy this sh file to run 
python -m src.data_symlink.trick_aug --repo_root /Users/doradong/Repo --mode str # trick: header-str(value)
python -m src.data_symlink.trick_aug --repo_root /Users/doradong/Repo --mode transpose # trick: permutation
python -m src.data_symlink.trick_aug --repo_root /Users/doradong/Repo --mode str_transpose # trick: tr + str 

python -m src.data_symlink.ln_scilake --repo_root /u1/z6dong/Repo --mode str
python -m src.data_symlink.ln_scilake --repo_root /u1/z6dong/Repo --mode tr
python -m src.data_symlink.ln_scilake --repo_root /u1/z6dong/Repo --mode str_tr
python -m src.data_symlink.ln_scilake --repo_root /u1/z6dong/Repo --mode base # symlink csvs to the target folder
# bash src/data_analysis/count_files.sh check whether the symlink path include some files
```

6. Run updated [starmie](https://github.com/DoraDong-2023/starmie_internal) scripts for finetuning and check performance
```bash
bash prepare_sample.sh # sample 1000 samples from each resources folder
# or python -m src.data_symlink.prepare_sample_server --root_dir /u4/z6dong/Repo --output scilake_final --output_file scilake_final_filelist.txt --limit 2000 --seed 42
# another substitution
python -m src.data_symlink.prepare_sample --root_dir /Users/doradong/Repo --output_file scilake_final_filelist.txt --limit 1000 --seed 42
# create for tricks augmented files
#python -m src.data_symlink.prepare_sample_tricks --input_file scilake_final_filelist.txt
# Input: scilake_final_filelist.txt ; Output: scilake_final_filelist_{tricks}_filelist.txt, 
python -m src.data_symlink.ln_scilake_final_link --filelist scilake_final_filelist.txt scilake_final_filelist_val.txt # create other 3 files
# (deprecate) (already processed in QC step) bash check_empty.sh # filter out empty files (or low quality files later)
bash scripts/step1_pretrain.sh # finetune contrastive learning
bash scripts/step2_extractvectors.sh # encode embeddings for query and datalake items
bash scripts/step3_search_hnsw.sh # data lake search (retrieve)! notice we move the gt compare to step3_processmetrics, because we need to compare multiple groundtruth
bash scripts/step3_processmetrics.sh # extract metrics based on searched results
bash scripts/step4_discovery.sh
```

Analysis on results
```bash  
# get top-10 results from step3_search_hnsw
python -m src.data_analysis.report_generation --json_path ~/Repo/starmie_internal/tmp/test_hnsw_search_scilake_large_full.json
python -m src.data_analysis.starmie_metrics_topk --input ~/Repo/starmie_internal/tmp/metrics_scilake_large_hnsw.json --output data/analysis/metrics_plot.pdf
# get distribution of groundtruth
python -m src.data_analysis.gt_distri # get csv with gt list > 1000 # Input: /Users/doradong/Repo/CitationLake/data/gt/scilake_large_gt__direct_label.pickle # Output: figures
python -m src.data_analysis.check_related --csv 201646309_table4.csv > logs/check_related_csv.log # check the related model of csv
```

7. Analyze some statistics
```bash
python -m src.data_preprocess.step1_analysis # get analysis on proportion of different links
python -m src.data_analysis.query_compare_API_local # compare the query results among local and API from s2orc
TODO: add statistics analysis from dataset_processed.ipynb
```

(Deprecated scripts: Previously we download pdfs and try to parse them. However, we find semantic scholar dataset includes it.)
```bash
# deprecated as we don't use PDF for extraction at this time
#python -m src.data_preprocess.step2_get_pdf #TODO: wait se_url_tab and then test
python -m src.data_preprocess.step_down_pdf
python -m src.data_preprocess.step_add_pdftab # Issue: deterministic PDF2table is not accurate enough. Try LLM based image extraction (not implemented here)
python -m src.data_preprocess.step_down_tex # Issue: IP rate limit on accessing tex files, Possible solution: use arxiv bulk downloading
python -m src.data_preprocess.step_add_textab
python -m src.data_preprocess.step_add_gittab
python -m src.data_ingestion.tmp_extract_url # Update PDF url from extracted url (some don't have .pdf, need to extract from html or add)
python -m src.data_ingestion.tmp_extract_table # Extract table/figures caption and cited text from s2orc dumped data, but don't contain text and figure detailed content!
python -m src.data_preprocess.step4 # process groundtruth (work for API, not work for dump data)
python -m src.data_preprocess.step2_Citation_Info
# (Optional) python -m src.data_preprocess.step3_statistic_table # get statistic tables
python -m src.data_preprocess.step1_parsetags # Parse tags into columns with name start with `card_tag_xx`
bash src/data_symlink/trick_str.sh # too slow
bash src/data_symlink/trick_tr.sh # too slow
bash src/data_symlink/trick_tr_str.sh # too slow
bash src/data_symlink/ln_scilake_large.sh # too slow
bash src/data_symlink/ln_scilake_final.sh
```


---
What dataset_processed should addressed
---

### Objective:
The goal is to create a reliable and minimal dataset by filtering and validating model cards from multiple sources. We aim to ensure that every card retained in the dataset:
1. Contains valid and verified metadata (BibTeX entries, paper links, GitHub repositories).
2. Has no duplicate entries—retain only the card with the highest `downloads` count if duplicates exist.
3. Downloads the associated data locally and reads it for further processing.
4. Annotates the dataset with citation information to build a **citation graph** that reflects inter-card relationships (i.e., a card’s associated paper cites another card's associated paper).  

We will leverage a **Citation Graph API** to annotate and establish these relationships.

### Step-by-Step Plan:

#### 1. **Data Filtering Phase**
   - **Input Sources:** 
     - card_readme content
     - BibTeX entries
     - GitHub repo READMEs
   - **Filtering Criteria:**
     - Remove any card that lacks all of the following:
       - Paper link (arXiv, DOI, or PDF)
       - BibTeX entry
       - GitHub link
     - Eliminate duplicate cards; keep the one with the highest `downloads` count.
     - Validate remaining links to ensure they are functional and non-placeholder.

#### 2. **Data Downloading Phase**
   - Download the related resources (PDFs, code repos, etc.).
   - Organize and read the downloaded content from the local directory for further processing.

#### 3. **Citation Annotation Phase**
   - Use the **Citation Graph API** to analyze and annotate each card by identifying citations between associated papers.
   - Establish inter-card relationships (i.e., identify which card’s paper cites another).

#### 4. **Final Dataset Construction**
   - Create a structured and annotated dataset that includes:
     - Valid and non-duplsicate card data.
     - Citation relationships between cards.
     - Metadata for each card (downloads, likes, BibTeX, and link status).
---