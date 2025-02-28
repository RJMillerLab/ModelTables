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
python -m data_preprocess.step1 # Split readme and tags, Parse tags into columns with name start with `card_tag_xx`, extract markdown table
python -m data_preprocess.step2 # process bibtex and download markdown csvs
python -m data_preprocess.step3 # download citations, produce real annotation dataset
python -m data_preprocess.step3_statistic_table # get statistic tables, not need to run step4, but require to run step3
python -m data_preprocess.step3_save_starmie_results # analyze searching results of starmie
python -m data_preprocess.step4 # process groundtruth

# TODO: add link extraction!
python -m data_preprocess.step_download_github_readme
python -m data_preprocess.step_add_github_tables
python -m data_preprocess.step_download_pdf 
python -m data_preprocess.step_add_pdf_tables 
python -m data_preprocess.step_download_tex
python -m data_preprocess.step_add_tex_tables
```

Then run in [starmie]()
