# CitationLake
## Features
- Extract text, figures, and tables from PDFs and links.
- Build and manage citation graphs.
- Define and execute tasks such as model retrieving, table integration, and data lake operations.
- User-friendly UI with pipeline visualization and task management.# CitationLake

## Download Data

This project uses datasets hosted on Hugging Face. Use the following commands to download the necessary data:
```bash
git clone https://github.com/DoraDong-2023/CitationLake.git
cd CitationLake/
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
python -m src.data_preprocess.step1 # Split readme and tags, parse urls, parse bibtex, 
# get:  ['modelId', 'author', 'last_modified', 'downloads', 'likes', 'library_name', 'tags', 'pipeline_tag', 'createdAt', 'card', 'card_tags', 'card_readme', 'pdf_link', 'github_link', 'all_links', 'extracted_bibtex', 'extracted_bibtex_tuple', 'parsed_bibtex_tuple_list', 'successful_parse_count']
python -m src.data_preprocess.step1_down_giturl # Download github URL README & HTMLs. Input: modelcard_step1.parquet, Output: data/processed/giturl_info.parquet
#python -m src.data_preprocess.step1_down_giturl_fake # if program has leakage but finished downloading, then re-run this code to save final parquet and cache files.
```


2. Download and build database for faster querying:
```bash
# TODO: add command from privatecommonscript to here, for downloading the semantic scholar here
# Requirement: cd to the path of downloaded dataset, e.g.: cd ~/shared_data/se_s2orc_250218
python -m src.data_localindexing.build_mini_s2orc build --directory ./ # After downloading semantic scholar dataset, build database based on it.
python -m src.data_localindexing.build_mini_s2orc query --title "estimating the resilience to natural disasters by using call detail records" --directory ./ # After building up database, query title based on db file.

# issue: citation edge is hard to store, it is too much ... Solution: I think we better using the API to query citation relationship? Or use cypher to query over graph condensely
# python -m src.data_localindexing.build_complete_citation build --directory ./ # build db for citation dataset
# python -m src.data_localindexing.build_complete_citation query --citationid 169 --directory ./
# (Optional) if you don't have key, use public API for querying citations instead
# python -m src.data_preprocess.step1_citationAPI # get citations through bibtex only by API. TODO: Update for bibtex + url, not bibtex only. TODO: Update for all bibtex, not the first bibtex
```


3. Extract tables to local folder:
```bash
python -m src.data_preprocess.step2 # extract table from .md git, from modelcards | save csvs to folder
python -m src.data_preprocess.step1_CitationInfo #  get citations relation from .db |, parse tables from citations
# TODO: get tags arxiv id
# TODO: get title
# TODO: get table from whole text in semantic scholar
# TODO: download pdf ? formalize table text?
# TODO: pdf2table? good or not?
python -m src.data_preprocess.step_add_gittab
```


4. Label groundtruth for unionable search baselines:
```bash
python -m src.data_preprocess.step4 # process groundtruth
```


5. Analyze some statistics
```bash
python -m src.data_preprocess.step1_analysis # get analysis on proportion of different links
python -m src.data_preprocess.step1_parsetags # Parse tags into columns with name start with `card_tag_xx`
python -m src.data_preprocess.step3_statistic_table # get statistic tables, not need to run step4, but require to run step3
python -m src.data_preprocess.step3_save_starmie_results # analyze searching results of starmie
```

(Deprecated scripts: Previously we download pdfs and try to parse them. However, we find semantic scholar dataset includes it.)
```bash
python -m src.data_preprocess.step_down_pdf
python -m src.data_preprocess.step_add_pdftab # Issue: deterministic PDF2table is not accurate enough. Try LLM based image extraction (not implemented here)
python -m src.data_preprocess.step_down_tex # Issue: IP rate limit on accessing tex files, Possible solution: use arxiv bulk downloading
python -m src.data_preprocess.step_add_textab
python -m src.data_ingestion.tmp_extract_url # Update PDF url from extracted url (some don't have .pdf, need to extract from html or add)
python -m src.data_ingestion.tmp_extract_table # Extract table/figures caption and cited text from s2orc dumped data, but don't contain text and figure detailed content!
```

---
Then run in [starmie]()


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