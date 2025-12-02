import json
import os
import argparse
import pandas as pd
import re
from pathlib import Path

def load_corpus_ids(corpus_file):
    """Load IDs from corpus jsonl file."""
    ids = []
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                ids.append(entry['id'])
    return ids

def find_table_file(table_id, base_dir):
    """Find the corresponding table file for a given ID."""
    # Try different directories and patterns
    dirs_to_search = [
        'tables_output',  # For arXiv papers
        'deduped_github_csvs',  # For GitHub tables
        'deduped_hugging_csvs'  # For HuggingFace tables
    ]
    
    for dir_name in dirs_to_search:
        search_dir = os.path.join(base_dir, dir_name)
        if not os.path.exists(search_dir):
            continue
            
        # Try different patterns
        patterns = [
            f"{table_id}.csv",  # Exact match
            f"{table_id}_table_0.csv",  # GitHub pattern
            f"{table_id}_table1.csv"    # HuggingFace pattern
        ]
        
        for pattern in patterns:
            for file_path in Path(search_dir).glob(pattern):
                return file_path
    
    return None

def truncate_text(text, max_tokens):
    """Truncate text to a maximum number of tokens."""
    if not max_tokens:
        return text
        
    # Split by comma and count tokens
    tokens = text.split()
    if len(tokens) <= max_tokens:
        return text
        
    # Take first max_tokens tokens
    return ','.join(tokens[:max_tokens])

def table_to_text(df):
    """Convert pandas DataFrame to a single line of text using comma separation."""
    # Get column names
    columns = df.columns.tolist()
    
    # Convert each row to a string
    rows = []
    for _, row in df.iterrows():
        # Convert each cell to string, strip whitespace and tabs, and join with commas
        row_str = ','.join(str(val).strip().replace('\t', ' ') for val in row)
        rows.append(row_str)
    
    # Join everything with commas in a single line and strip any leading/trailing whitespace
    return ','.join([','.join(columns)] + rows).strip()

def load_table_data(corpus_file, base_dir, max_tokens=None):
    """Load table data based on IDs from corpus."""
    queries = []
    
    # First, get all IDs from corpus
    print("Loading IDs from corpus...")
    ids = load_corpus_ids(corpus_file)
    print(f"Found {len(ids)} IDs in corpus")
    
    # Then, find and load corresponding table files
    for table_id in ids:
        try:
            # Find the table file
            table_file = find_table_file(table_id, base_dir)
            if not table_file:
                print(f"Warning: Could not find table file for {table_id}, skipping...")
                continue
            
            # Read CSV file
            df = pd.read_csv(table_file)
            
            # Convert table to text
            query_text = table_to_text(df)
            
            # Truncate if max_tokens is specified
            if max_tokens:
                query_text = truncate_text(query_text, max_tokens)
            
            queries.append({
                'id': table_id,
                'contents': query_text
            })
            
        except Exception as e:
            print(f"Error processing {table_id}: {e}")
            continue
    
    return queries

def verify_tsv_format(file_path):
    """Verify the TSV file format is correct."""
    print(f"Verifying TSV format for {file_path}...")
    issues = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:  # Skip empty lines
                continue
                
            # Check for exactly one tab
            tabs = line.count('\t')
            if tabs != 1:
                issues.append(f"Line {i}: Found {tabs} tabs, expected 1")
                continue
                
            # Split and check parts
            parts = line.split('\t')
            if len(parts) != 2:
                issues.append(f"Line {i}: Split into {len(parts)} parts, expected 2")
                continue
                
            # Check ID and text are not empty
            query_id, query_text = parts
            if not query_id.strip():
                issues.append(f"Line {i}: Empty query ID")
            if not query_text.strip():
                issues.append(f"Line {i}: Empty query text")
                
            # Check for any remaining tabs in text
            if '\t' in query_text:
                issues.append(f"Line {i}: Found tab in query text")
                
    if issues:
        print("Found issues in TSV file:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("TSV format verification passed!")
        return True

def create_queries_tsv(entries, output_file):
    """Create queries.tsv from entries (dicts with 'id' and 'contents')."""
    # Create ID mapping
    id_mapping = {}
    valid_entries = 0
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, entry in enumerate(entries, 1):
            # Use numeric ID for pyserini
            query_id = str(i)
            
            # Clean up the text: remove tabs and extra whitespace
            query_text = entry['contents'].strip()
            # Replace tabs with spaces in the content only
            query_text = query_text.replace('\t', ' ').replace('\n', ' ').replace('\r', ' ')
            # Clean up extra spaces around commas
            query_text = ','.join(part.strip() for part in query_text.split(','))
            
            # Skip empty entries
            if not query_text:
                continue
                
            # Store original ID mapping
            id_mapping[query_id] = entry['id']
            
            # Write with a single tab
            f.write(f"{query_id}\t{query_text}\n")
            valid_entries += 1
    
    # Save ID mapping to a separate file
    mapping_file = output_file.replace('.tsv', '_mapping.json')
    with open(mapping_file, 'w', encoding='utf-8') as f:
        json.dump(id_mapping, f, indent=2, ensure_ascii=False)
    
    print(f"Created {valid_entries} valid queries in {output_file}")
    print(f"ID mapping saved to {mapping_file}")
    
    # Verify the format
    if not verify_tsv_format(output_file):
        print("Warning: TSV format verification failed!")
    else:
        print("TSV format verification passed!")

def main():
    # Support TAG environment variable for versioning
    tag = os.environ.get('TAG', '')
    suffix = f"_{tag}" if tag else ""
    
    parser = argparse.ArgumentParser(description='Create queries.tsv from local table data')
    parser.add_argument('--corpus', default='data/tmp/corpus/collection.jsonl',
                        help='Path to corpus jsonl file')
    parser.add_argument('--base-dir', default='data/processed',
                        help='Base directory containing table files')
    parser.add_argument('--output-dir', default='data/tmp/',
                        help='Output directory for query files')
    parser.add_argument('--max-tokens', type=int, default=None,
                        help='Maximum number of tokens per query')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"🔄 Loading tables from {args.base_dir} based on corpus {args.corpus}...")
    print(f"🔄 Maximum number of tokens per query: {args.max_tokens}")
    entries = load_table_data(args.corpus, args.base_dir, args.max_tokens)

    queries_file = os.path.join(args.output_dir, f'queries_table{suffix}.tsv')
    create_queries_tsv(entries, queries_file)

if __name__ == "__main__":
    main()

