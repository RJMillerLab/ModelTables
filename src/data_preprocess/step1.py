
"""
Author: Zhengyuan Dong
Created: 2025-02-23
Last Modified: 2025-02-25

Description: Extract BibTeX entries from the 'card_readme' column and save to CSV files.
"""
import pandas as pd
import numpy as np
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
from joblib import Parallel, delayed, parallel_backend
import re, os, json, time
from concurrent.futures import ThreadPoolExecutor
from src.utils import load_data, load_config, load_combined_data, safe_json_dumps
from urllib.parse import urlparse
import pyarrow as pa
import pyarrow.parquet as pq
from src.data_ingestion.readme_parser import BibTeXExtractor
from src.data_ingestion.bibtex_parser import BibTeXFactory

tqdm.pandas()

# List of valid PDF link domains
VALID_PDF_LINKS = [
    "arxiv.org", "biorxiv.org", "medrxiv.org", "dl.acm.org",
    "dblp.uni-trier.de", "scholar.google.com", "pubmed.ncbi.nlm.nih.gov",
    "frontiersin.org", "mdpi.com", "cvpr.thecvf.com", "nips.cc",
    "icml.cc", "ijcai.org", "webofscience.com", "journals.plos.org",
    "nature.com", "semanticscholar.org", "chemrxiv.org", "link.springer.com",
    "ieeexplore.ieee.org", "aaai.org", "openaccess.thecvf.com",
]

ARXIV_IGNORE_IDS = ["arxiv:1910.09700"]

URL_REGEX = re.compile(r"(https?://\S+|www\.\S+)")

def is_valid_pdf_link(link):
    """
    Check if the given link is a valid PDF link.
    It is valid if the URL's domain is in the allowed list
    or the link ends with '.pdf'.
    """
    try:
        parsed_url = urlparse(link)
        domain = parsed_url.netloc.lstrip("www.")
    except Exception:
        return False
    return (domain in VALID_PDF_LINKS or link.lower().endswith(".pdf"))

def extract_links(text):
    """
    Extract all URLs from the given text and filter out PDF and GitHub links.
    Additional URL types (e.g., bibtex, arXiv identifiers) can be added as needed.
    """
    if pd.isna(text):
        return {"pdf_link": None, "github_link": None, "all_links": []}
    # Find all URLs (matching http(s):// and www.)
    all_links = [link.strip(".,)") for link in URL_REGEX.findall(text)]
    # Filter PDF and GitHub links
    pdf_links = [link for link in all_links if is_valid_pdf_link(link)]
    github_links = [link for link in all_links if "github.com" in link or "github.io" in link]
    # Special handling: if any ignored arXiv ID is found and it's the only PDF link, ignore it.
    ignore_found = any(ignore_id in link for link in all_links for ignore_id in ARXIV_IGNORE_IDS)
    if ignore_found and len(pdf_links) == 1 and any(ignore_id in pdf_links[0] for ignore_id in ARXIV_IGNORE_IDS):
        pdf_links = []
    return {
        "pdf_link": pdf_links if pdf_links else None,
        "github_link": github_links if github_links else None,
        "all_links": all_links if all_links else None
    }

def process_basic_elements_parallel(df, n_jobs=-1):
    """
    Process the 'card_readme' column in parallel to extract URL information.
    """
    # Ensure card_readme is filled with empty string for NaN values
    texts = df['card_readme'].fillna('').tolist()
    # Process each text using the extract_links function in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(extract_links)(text) for text in tqdm(texts, desc="Extracting URLs")
    )
    # Update the DataFrame with new columns from the results
    df['pdf_link'] = [res["pdf_link"] if res["pdf_link"] else None for res in results]
    df['github_link'] = [res["github_link"] if res["github_link"] else None for res in results]
    df['all_links'] = [', '.join(res["all_links"]) if res["all_links"] else None for res in results]
    return df

# Function to clean up line breaks and whitespace
def clean_content(content):
    if content is None:
        return None
    # Normalize line endings (but keep everything else intact)
    return content.replace('\r\n', '\n').replace('\r', '\n')

# Separate tags and README using split and replace
def separate_tags_and_readme(card_content):
    tags, readme = None, None
    try:
        # Clean up content minimally
        card_content = clean_content(card_content)
        if card_content.startswith("---\n"):
            # Split only on the first two "---\n"
            parts = card_content.split("---\n", 2)
            if len(parts) > 2:
                tags = parts[1]  # Keep tags part intact
                readme = parts[2]  # Keep readme part intact
            else:
                readme = parts[1]  # Handle case where only readme exists
        else:
            readme = card_content  # No tags part, entire content is readme
    except Exception as e:
        print(f"Error parsing content: {e}")
    return tags, readme

# Process to extract tags and README using tqdm and apply
def extract_tags_and_readme_parallel(df, n_jobs=-1):
    # Process a single row
    def process_card(card_content):
        return separate_tags_and_readme(card_content)
    # Use joblib to parallelize the map function
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_card)(content) for content in tqdm(df["card"], desc="Extracting Tags and README")
    )
    # Split the results into separate columns
    df["card_tags"] = [x[0] for x in results]
    df["card_readme"] = [x[1] for x in results]
    return df

# Clean and compare restored content using replace and split
def clean_for_comparison(content):
    if content is None:
        return ""
    return content.replace("\n", "").replace("\r", "").replace("---", "").replace(" ", "").strip()

# Validate parsed content using tqdm
def validate_parsing(df):
    inconsistencies = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Validating Parsed Cards"):
        original_card = row["card"]  # Preserve the original structure
        restored_card = ""
        if row["card_tags"] is not None:
            restored_card = f"---\n{row['card_tags']}\n---\n{row['card_readme']}"
        else:
            restored_card = row["card_readme"]
        if original_card.strip() != restored_card.strip():
            inconsistencies.append({
                "original_card": row["card"],
                "restored_card": restored_card,
                "card_tags": row["card_tags"],
                "card_readme": row["card_readme"]
            })
    return pd.DataFrame(inconsistencies)

def extract_bibtex(df, readme_key="card_readme", new_key="extracted_bibtex"):
    """Extract BibTeX entries from the 'card_readme' column."""
    #df["extracted_bibtex"] = df["card_readme"].apply(lambda x: BibTeXExtractor().extract(x) if isinstance(x, str) else [], meta=('extracted_bibtex', 'object'))
    df[new_key] = df[readme_key].apply(lambda x: BibTeXExtractor().extract(x) if isinstance(x, str) else [])
    print(f"New attributes added: {new_key}")

def process_bibtex_tuple(entry):
    """Process a BibTeX entry and return parsed results."""
    parsed_results = []
    success_count = 0
    if isinstance(entry, (tuple, np.ndarray)):
        entry = list(entry)
    if isinstance(entry, list):
        for single_entry in entry:
            single_entry = ensure_string(single_entry)
            if single_entry:
                parsed_entry, flag = BibTeXFactory.parse_bibtex(single_entry)
                parsed_results.append(parsed_entry)
                if flag:
                    success_count += 1
    elif isinstance(entry, str):
        single_entry = ensure_string(entry)
        parsed_entry, flag = BibTeXFactory.parse_bibtex(single_entry)
        parsed_results.append(parsed_entry)
        if flag:
            success_count += 1
    return parsed_results, success_count

def parse_bibtex_entries(df, key="extracted_bibtex_tuple", output_key="parsed_bibtex_tuple_list", count_key = "successful_parse_count"):
    """Process BibTeX entries in the DataFrame and return results."""
    non_null_entries = df[df[key].notnull()]
    # Initialize tqdm progress bar
    results = []
    with tqdm(total=len(non_null_entries), desc="Processing BibTeX tuples") as pbar:
        results = Parallel(n_jobs=-1)(
            delayed(process_bibtex_tuple)(entry) for entry in non_null_entries[key]
        )
        pbar.update(len(non_null_entries))  # Update progress bar after processing
    # Create new columns for parsed results
    non_null_entries[output_key] = [result[0] for result in results]
    non_null_entries[count_key] = [result[1] for result in results]
    df.loc[non_null_entries.index, output_key] = non_null_entries[output_key]
    df.loc[non_null_entries.index, count_key] = non_null_entries[count_key]
    print(f"New attributes added: '{output_key}', '{count_key}'.")
    return df

def ensure_string(entry):
    """Ensure the input is a string."""
    return str(entry) if entry is not None else None

def main():
    data_type = "modelcard"
    config = load_config('config.yaml')
    raw_base_path = os.path.join(config.get('base_path'), 'raw')
    
    print("⚠️ Step 1: Loading data...")
    start_time = time.time()
    df = load_combined_data(data_type, file_path=raw_base_path)
    print("✅ Done. Time cost: {:.2f} seconds.".format(time.time() - start_time))

    print("⚠️ Step 2: Splitting readme and tags...")
    start_time = time.time()
    df = extract_tags_and_readme_parallel(df)
    #inconsistencies_df = validate_parsing(df_split)
    print("✅ Done. Time cost: {:.2f} seconds.".format(time.time() - start_time))

    print("⚠️ Step 3: Extracting URLs...")
    start_time = time.time()
    df = process_basic_elements_parallel(df)
    print(f"✅ Done. Time cost: {time.time() - start_time:.2f} seconds.")

    print("⚠️Step 4: Extracting BibTeX entries...")
    start_time = time.time()
    extract_bibtex(df)
    df["extracted_bibtex_tuple"] = df["extracted_bibtex"].apply(lambda x: tuple(x) if isinstance(x, list) else (x,))
    print("✅ Done. Time cost: {:.2f} seconds.".format(time.time() - start_time))

    print("⚠️Step 5: Parsing BibTeX entries...")
    start_time = time.time()
    processed_entries = parse_bibtex_entries(df)
    print("✅ done. Time cost: {:.2f} seconds.".format(time.time() - start_time))

    print("all attributes: ", list(df.columns))

    #print("⚠️ Step 4: Convert list to str...")
    #start_time = time.time()
    #for col in df.columns:
    #    if df[col].apply(lambda x: isinstance(x, (list, tuple, np.ndarray))).any():
    #        df[col] = [safe_json_dumps(x) if isinstance(x, (list, tuple, np.ndarray)) else x for x in df[col]]
    #print("✅ Done. Time cost: {:.2f} seconds.".format(time.time() - start_time))

    print("⚠️ Step 5: Saving results to Parquet file...")
    start_time = time.time()
    pq.write_table(pa.Table.from_pandas(df), os.path.join(config.get('base_path'), 'processed', f"{data_type}_step1.parquet"))
    #df.to_parquet(os.path.join(config.get('base_path'), 'processed', f"{data_type}_step1.parquet"))
    print("✅ Done. Time cost: {:.2f} seconds.".format(time.time() - start_time))
    print("Sampled data: ", df.head(5))

if __name__ == "__main__":
    main()

"""
Exampled output:

⚠️ Step 1: Loading data...
✅ Done. Time cost: 6.00 seconds.
⚠️ Step 2: Splitting readme and tags...
Extracting Tags and README: 100%|█████████████████████████████████████████████| 1108759/1108759 [00:43<00:00, 25606.19it/s]
✅ Done. Time cost: 48.46 seconds.
⚠️ Step 3: Extracting URLs...
Extracting URLs: 100%|████████████████████████████████████████████████████████| 1108759/1108759 [00:46<00:00, 24050.98it/s]
✅ Done. Time cost: 48.67 seconds.
⚠️Step 4: Extracting BibTeX entries...
New attributes added: extracted_bibtex
✅ Done. Time cost: 50.67 seconds.
⚠️Step 5: Parsing BibTeX entries...
Processing BibTeX tuples: 100%|███████████████████████████████████████████████| 1108759/1108759 [00:28<00:00, 39524.73it/s]
New attributes added: 'parsed_bibtex_tuple_list', 'successful_parse_count'.
✅ done. Time cost: 46.78 seconds.
all attributes:  ['modelId', 'author', 'last_modified', 'downloads', 'likes', 'library_name', 'tags', 'pipeline_tag', 'createdAt', 'card', 'card_tags', 'card_readme', 'pdf_link', 'github_link', 'all_links', 'extracted_bibtex', 'extracted_bibtex_tuple', 'parsed_bibtex_tuple_list', 'successful_parse_count']
⚠️ Step 5: Saving results to Parquet file...
✅ Done. Time cost: 158.64 seconds.
"""
