
"""
Author: Zhengyuan Dong
Created: 2025-02-23
Last Modified: 2025-02-25

Description: Extract BibTeX entries from the 'card_readme' column and save to CSV files.
"""
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
from joblib import Parallel, delayed, parallel_backend
import re, os, json, time
from concurrent.futures import ThreadPoolExecutor
from src.utils import load_config, load_combined_data, safe_json_dumps, load_table_from_duckdb, load_table_from_sqlite, to_parquet
from urllib.parse import urlparse
import pyarrow as pa
import pyarrow.parquet as pq
import duckdb
from src.data_ingestion.readme_parser import BibTeXExtractor
from src.data_ingestion.bibtex_parser import BibTeXFactory
import hashlib
from typing import Tuple, Optional, Set

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

def compute_card_hash(card_content):
    """Compute hash of card content for quick comparison"""
    if pd.isna(card_content) or card_content is None:
        return None
    return hashlib.sha256(str(card_content).encode('utf-8')).hexdigest()

def compare_cards_incremental(df_new, baseline_step1_path, baseline_raw_path=None, baseline_date=None):
    """
    Compare new raw data with baseline to identify unchanged/updated/new models.
    
    Args:
        df_new: New raw dataframe with modelId and card columns
        baseline_step1_path: Path to previous step1 parquet file
        baseline_raw_path: Path to baseline raw data directory (optional, will try to infer)
        baseline_date: Date string of baseline snapshot (optional)
    
    Returns:
        Tuple of (unchanged_ids: Set, updated_ids: Set, new_ids: Set, baseline_df: DataFrame)
    """
    print("🔄 Versioning mode: Comparing with baseline...")
    
    # Load baseline step1 result
    if not os.path.exists(baseline_step1_path):
        print(f"⚠️  Baseline step1 file not found: {baseline_step1_path}")
        print("   Running in full mode (no baseline)")
        return set(), set(), set(df_new['modelId'].unique()), None
    
    print(f"   Loading baseline: {baseline_step1_path}")
    baseline_step1 = pd.read_parquet(baseline_step1_path)
    baseline_model_ids = set(baseline_step1['modelId'].unique())
    print(f"   Baseline has {len(baseline_model_ids)} models")
    
    # Load baseline raw data to get card content
    if baseline_raw_path is None:
        # Try to infer from baseline_step1_path or use default
        if baseline_date:
            from src.utils import load_config
            config = load_config('config.yaml')
            base_path = config.get('base_path')
            baseline_raw_path = os.path.join(base_path, f"raw_{baseline_date}")
            baseline_has_date = True
        else:
            # Default: use data/raw (no date tag) as baseline
            from src.utils import load_config
            config = load_config('config.yaml')
            base_path = config.get('base_path')
            baseline_raw_path = os.path.join(base_path, 'raw')
            baseline_has_date = False
            print(f"   Using default baseline raw data: {baseline_raw_path}")
    
    baseline_raw_path = os.path.expanduser(baseline_raw_path)
    
    if not os.path.exists(baseline_raw_path):
        print(f"⚠️  Baseline raw directory not found: {baseline_raw_path}")
        print("   Running in full mode (no baseline raw data)")
        return set(), set(), set(df_new['modelId'].unique()), baseline_step1
    
    print(f"   Loading baseline raw data: {baseline_raw_path}")
    # For data/raw (no date), use date=None to use fixed file names
    # For data/raw_<date>, use date parameter to auto-detect files
    if baseline_has_date:
        # Extract date from path (e.g., "raw_251116" -> "251116")
        baseline_date_from_path = os.path.basename(baseline_raw_path).replace('raw_', '')
        baseline_raw = load_combined_data('modelcard', file_path=baseline_raw_path, columns=['modelId', 'card'], date=baseline_date_from_path)
    else:
        # Use date=None for data/raw to use fixed file names (old rule)
        baseline_raw = load_combined_data('modelcard', file_path=baseline_raw_path, columns=['modelId', 'card'], date=None)
    print(f"   Baseline raw has {len(baseline_raw)} models")
    
    # Compute hashes for comparison (vectorized)
    print("   Computing card hashes...")
    df_new['card_hash'] = df_new['card'].apply(compute_card_hash)
    baseline_raw['card_hash'] = baseline_raw['card'].apply(compute_card_hash)
    
    # Use pandas merge for fast comparison (much faster than loops)
    print("   Comparing cards using merge...")
    # Create comparison dataframe
    df_new_hash = df_new[['modelId', 'card_hash']].copy()
    baseline_hash_df = baseline_raw[['modelId', 'card_hash']].copy()
    baseline_hash_df = baseline_hash_df.rename(columns={'card_hash': 'baseline_hash'})
    
    # Merge to compare
    comparison_df = df_new_hash.merge(baseline_hash_df, on='modelId', how='left')
    
    # Classify models using vectorized operations
    # New models: not in baseline
    new_ids = set(df_new['modelId'].unique()) - baseline_model_ids
    
    # Existing models: compare hashes
    existing_df = comparison_df[comparison_df['baseline_hash'].notna()].copy()
    
    # Unchanged: hashes match
    unchanged_mask = existing_df['card_hash'] == existing_df['baseline_hash']
    unchanged_ids = set(existing_df[unchanged_mask]['modelId'].unique())
    
    # Updated: hashes don't match
    updated_mask = existing_df['card_hash'] != existing_df['baseline_hash']
    updated_ids = set(existing_df[updated_mask]['modelId'].unique())
    
    # Also include models in baseline step1 but not in baseline raw
    missing_in_raw = baseline_model_ids - set(baseline_raw['modelId'].unique())
    updated_ids.update(missing_in_raw & set(df_new['modelId'].unique()))
    
    new_model_ids = set(df_new['modelId'].unique())
    print(f"   📊 Comparison results:")
    print(f"      Unchanged: {len(unchanged_ids):,} models")
    print(f"      Updated:   {len(updated_ids):,} models")
    print(f"      New:        {len(new_ids):,} models")
    print(f"      Total:      {len(new_model_ids):,} models")
    
    return unchanged_ids, updated_ids, new_ids, baseline_step1

def parse_args():
    parser = argparse.ArgumentParser(description="Step1: parse initial model card elements")
    parser.add_argument("--raw-date", dest="raw_date", default=None,
                        help="If set, load raw data from data/raw_<DATE> (e.g., 251117)")
    parser.add_argument("--tag", dest="tag", default=None,
                        help="Suffix for output parquet (default: equals --raw-date when provided)")
    parser.add_argument("--versioning", dest="versioning", action="store_true",
                        help="Enable versioning mode: reuse results for unchanged cards from baseline")
    parser.add_argument("--baseline-step1", dest="baseline_step1", default=None,
                        help="Path to baseline step1 parquet file (default: auto-detect)")
    parser.add_argument("--baseline-date", dest="baseline_date", default=None,
                        help="Date of baseline snapshot (e.g., 251116). Used to locate baseline raw data")
    return parser.parse_args()

def main():
    args = parse_args()
    data_type = "modelcard"
    config = load_config('config.yaml')
    base_path = config.get('base_path')
    if not base_path:
        raise ValueError("base_path missing in config.yaml")

    # Validate raw date if provided
    if args.raw_date:
        from src.utils import validate_raw_date
        validate_raw_date(args.raw_date, base_path=base_path, raise_error=True)
    
    raw_dir = f"raw_{args.raw_date}" if args.raw_date else "raw"
    raw_base_path = os.path.join(base_path, raw_dir)
    raw_base_path = os.path.expanduser(raw_base_path)
    if not os.path.exists(raw_base_path):
        # If no date provided, use standard error. If date provided, validate_raw_date should have caught it.
        if args.raw_date:
            from src.utils import validate_raw_date
            validate_raw_date(args.raw_date, base_path=base_path, raise_error=True)
        raise FileNotFoundError(f"Raw data directory not found: {raw_base_path}")
    tag = args.tag or args.raw_date
    output_suffix = f"_{tag}" if tag else ""
    output_path = os.path.join(base_path, 'processed', f"{data_type}_step1{output_suffix}.parquet")
    
    # Determine baseline path for versioning mode
    baseline_step1_path = args.baseline_step1
    if args.versioning and baseline_step1_path is None:
        # Auto-detect baseline: Priority: 1) modelcard_step1.parquet (no tag, default), 2) modelcard_step1_<date>.parquet
        processed_dir = os.path.join(base_path, 'processed')
        default_baseline = os.path.join(processed_dir, f"{data_type}_step1.parquet")
        if os.path.exists(default_baseline):
            baseline_step1_path = default_baseline
            print(f"📌 Auto-detected baseline (default): {baseline_step1_path}")
        else:
            # Fallback: Try to find latest dated version only if default doesn't exist
            from src.utils import list_available_raw_dates
            available_dates = list_available_raw_dates(base_path)
            if available_dates and args.raw_date:
                # Try previous date
                prev_dates = [d for d in available_dates if d < args.raw_date]
                if prev_dates:
                    prev_date = prev_dates[-1]
                    candidate = os.path.join(processed_dir, f"{data_type}_step1_{prev_date}.parquet")
                    if os.path.exists(candidate):
                        baseline_step1_path = candidate
                        print(f"📌 Auto-detected baseline (dated): {baseline_step1_path}")
    
    print(f"⚠️ Step 1: Loading data from {raw_base_path} ...")
    start_time = time.time()
    df = load_combined_data(data_type, file_path=raw_base_path, date=args.raw_date)
    #df = load_table_from_duckdb(f"raw_{data_type}", db_path="modellake_all.db")
    print("✅ Done. Time cost: {:.2f} seconds.".format(time.time() - start_time))
    
    # Versioning mode: compare with baseline
    unchanged_ids = set()
    updated_ids = set()
    new_ids = set()
    baseline_step1 = None
    
    if args.versioning and baseline_step1_path:
        unchanged_ids, updated_ids, new_ids, baseline_step1 = compare_cards_incremental(
            df, baseline_step1_path, baseline_date=args.baseline_date
        )
        
        if baseline_step1 is not None and len(unchanged_ids) > 0:
            print(f"\n⚡ Versioning mode: Reusing {len(unchanged_ids):,} unchanged models from baseline")
            # Filter to only process changed/new models
            changed_ids = updated_ids | new_ids
            df_to_process = df[df['modelId'].isin(changed_ids)].copy()
            print(f"   Processing {len(df_to_process):,} changed/new models")
        else:
            print("\n⚠️  Versioning mode enabled but no baseline found, running in full mode")
            df_to_process = df
    else:
        df_to_process = df
    
    # Store original df for merging later
    df_original = df.copy()

    print("⚠️ Step 2: Splitting readme and tags...")
    start_time = time.time()
    df_to_process = extract_tags_and_readme_parallel(df_to_process)
    #inconsistencies_df = validate_parsing(df_split)
    print("✅ Done. Time cost: {:.2f} seconds.".format(time.time() - start_time))

    print("⚠️ Step 3: Extracting URLs...")
    start_time = time.time()
    df_to_process = process_basic_elements_parallel(df_to_process)
    print(f"✅ Done. Time cost: {time.time() - start_time:.2f} seconds.")

    print("⚠️Step 4: Extracting BibTeX entries...")
    start_time = time.time()
    extract_bibtex(df_to_process)
    df_to_process["extracted_bibtex_tuple"] = df_to_process["extracted_bibtex"].apply(lambda x: tuple(x) if isinstance(x, list) else (x,))
    print("✅ Done. Time cost: {:.2f} seconds.".format(time.time() - start_time))

    print("⚠️Step 5: Parsing BibTeX entries...")
    start_time = time.time()
    processed_entries = parse_bibtex_entries(df_to_process)
    print("✅ done. Time cost: {:.2f} seconds.".format(time.time() - start_time))
    
    # Merge with baseline if in versioning mode
    if args.versioning and baseline_step1 is not None and len(unchanged_ids) > 0:
        print(f"\n🔄 Merging with baseline results...")
        # Get unchanged rows from baseline
        baseline_unchanged = baseline_step1[baseline_step1['modelId'].isin(unchanged_ids)].copy()
        
        # Ensure columns match
        # Remove 'card' column from both if present (we don't save it in step1 output)
        df_to_process = df_to_process.drop(columns=['card'], errors='ignore')
        baseline_unchanged = baseline_unchanged.drop(columns=['card'], errors='ignore')
        
        # Align columns
        common_cols = set(df_to_process.columns) & set(baseline_unchanged.columns)
        df_to_process = df_to_process[[c for c in df_to_process.columns if c in common_cols]]
        baseline_unchanged = baseline_unchanged[[c for c in baseline_unchanged.columns if c in common_cols]]
        
        # Merge
        df = pd.concat([df_to_process, baseline_unchanged], ignore_index=True)
        print(f"   Merged: {len(df_to_process):,} new/updated + {len(baseline_unchanged):,} unchanged = {len(df):,} total")
    else:
        df = df_to_process.drop(columns=['card'], errors='ignore')

    print("all attributes: ", list(df.columns))

    #print("⚠️ Step 4: Convert list to str...")
    #start_time = time.time()
    #for col in df.columns:
    #    if df[col].apply(lambda x: isinstance(x, (list, tuple, np.ndarray))).any():
    #        df[col] = [safe_json_dumps(x) if isinstance(x, (list, tuple, np.ndarray)) else x for x in df[col]]
    #print("✅ Done. Time cost: {:.2f} seconds.".format(time.time() - start_time))

    print("⚠️ Step 5: Saving results to Parquet file...")
    df.drop(columns=['card'], inplace=True, errors='ignore') # get card from raw instead of this saved file
    start_time = time.time()
    to_parquet(df, output_path)
    #df.to_parquet(os.path.join(config.get('base_path'), 'processed', f"{data_type}_step1.parquet"), compression='zstd', engine='pyarrow')
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
