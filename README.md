# CitationLake
## Features
- Extract text, figures, and tables from PDFs and links.
- Build and manage citation graphs.
- Define and execute tasks such as model retrieving, table integration, and data lake operations.
- User-friendly UI with pipeline visualization and task management.# CitationLake

## Download Data

This project uses datasets hosted on Hugging Face. Use the following commands to download the necessary data:
```bash
# Install Git LFS (Large File Storage)
$ git lfs install

# Clone required datasets
$ git clone https://huggingface.co/datasets/librarian-bots/model_cards_with_metadata data/model_cards
$ git clone https://huggingface.co/datasets/librarian-bots/dataset_cards_with_metadata data/dataset_cards
```

## Analysis

Run this command
```bash
# cd src

# get url, bibtex, citations relationship, extract tables
python -m src.data_preprocess.step1 # Split readme and tags, parse urls
# TODO: Parse tags into columns with name start with `card_tag_xx` | extract markdown table from card_readme | 
python -m src.data_preprocess.step2 # process bibtex from card_readme | download markdown csvs
python -m src.data_preprocess.step3 # get citations through bibtex by API | 
python -m src.data_preprocess.step4 # process groundtruth
python -m src.data_preprocess.step_download_github_readme
python -m src.data_preprocess.step_add_github_tables
python -m src.data_preprocess.step_download_pdf 
python -m src.data_preprocess.step_add_pdf_tables 
python -m src.data_preprocess.step_download_tex
python -m src.data_preprocess.step_add_tex_tables


# get statistics
python -m src.data_preprocess.step1_analysis # get analysis on proportion of different links
python -m src.data_preprocess.step3_statistic_table # get statistic tables, not need to run step4, but require to run step3
python -m src.data_preprocess.step3_save_starmie_results # analyze searching results of starmie

# tmp:
python -m src.data_ingestion.tmp_extract_url # Update PDF url from extracted url (some don't have .pdf, need to extract from html or add)
python -m src.data_ingestion.tmp_extract_table # Extract table/figures caption and cited text from s2orc dumped data, but don't contain text and figure detailed content!

```

Then run in [starmie]()





What dataset_processed should addressed
---

### Objective:
The goal is to create a reliable and minimal dataset by filtering and validating model cards from multiple sources. We aim to ensure that every card retained in the dataset:
1. Contains valid and verified metadata (BibTeX entries, paper links, GitHub repositories).
2. Has no duplicate entries—retain only the card with the highest `downloads` count if duplicates exist.
3. Downloads the associated data locally and reads it for further processing.
4. Annotates the dataset with citation information to build a **citation graph** that reflects inter-card relationships (i.e., a card’s associated paper cites another card's associated paper).  

We will leverage a **Citation Graph API** to annotate and establish these relationships.

---

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
     - Valid and non-duplicate card data.
     - Citation relationships between cards.
     - Metadata for each card (downloads, likes, BibTeX, and link status).
---