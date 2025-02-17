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

```markdown
# Project Directory Structure
├── data/                  # All data files and indexes
│   ├── raw/               # Raw data (downloaded files)
│   │   ├── papers/        # Raw paper files
│   │   ├── citations/     # Raw citation data
│   │   └── tables/        # Raw table files (if separate)
│   ├── processed/         # Processed (standardized) data
│   │   ├── papers/        # Processed paper metadata
│   │   ├── tables/        # Processed table data
│   │   └── citations/     # Processed citation data
│   ├── tmp_inference/     # Temporary data for inference
│   └── index.json         # Metadata index file
├── examples/              # Example datasets and demonstration files
├── src/                   # Source code
│   ├── data_ingestion/    # Stage 1: Data ingestion and processing
│   │   ├── paper_parsers/ # Paper parsers for various file types
│   │   ├── citation_ingestion/  # Citation ingestion modules (API/dump)
│   │   └── converters/    # Table converters
│   ├── evaluation/        # Stage 3: Evaluation modules
│   ├── visualization/     # Stage 4: Visualization (text reporting)
│   ├── factory/           # Factory modules (unified creation of components)
│   ├── storage/           # Data storage interfaces (read/write processed data)
│   └── main.py            # Main pipeline entry point
├── tests/                 # Unit tests for various modules
└── README.md              # Project documentation
```
