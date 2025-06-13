import pandas as pd
import os
import json
import numpy as np

def load_mappings():
    """
    Load all mapping files and return a list of (csv_path, readme_path, source)
    """
    records = []

    # GitHub
    print("\nLoading GitHub mapping...")
    github_info_path = os.path.join('data', 'processed', 'github_readmes_info.parquet')
    md_map_path = os.path.join('data', 'processed', 'deduped_github_csvs', 'md_to_csv_mapping.json')
    
    if os.path.exists(github_info_path) and os.path.exists(md_map_path):
        # Load the mapping files
        df_github = pd.read_parquet(github_info_path)
        with open(md_map_path, 'r') as f:
            md_to_csv = json.load(f)
            
        print("GitHub info columns:", df_github.columns.tolist())
        print(f"Found {len(md_to_csv)} markdown files with tables")
        
        github_count = 0
        for _, row in df_github.iterrows():
            readme_paths = row['readme_path']
            if isinstance(readme_paths, (list, np.ndarray)):
                for path in readme_paths:
                    if pd.notna(path):
                        # Extract SHA256 from the filename
                        sha256 = os.path.basename(path).replace(".md", "")
                        csv_paths = md_to_csv.get(sha256)
                        if csv_paths is None:
                            csv_paths = []
                        for csv_path in csv_paths:
                            records.append({
                                'csv_path': os.path.join('data', 'processed', 'deduped_github_csvs', csv_path),
                                'readme_path': path,
                                'source': 'github'
                            })
                            github_count += 1
            elif pd.notna(readme_paths):
                # Extract SHA256 from the filename
                sha256 = os.path.basename(readme_paths).replace(".md", "")
                csv_paths = md_to_csv.get(sha256)
                if csv_paths is None:
                    csv_paths = []
                for csv_path in csv_paths:
                    records.append({
                        'csv_path': os.path.join('data', 'processed', 'deduped_github_csvs', csv_path),
                        'readme_path': readme_paths,
                        'source': 'github'
                    })
                    github_count += 1
        print(f"Added {github_count} GitHub mappings")
        
        # GitHub deduplication: deduplicate based on all three keys (csv_path, readme_path, source)
        print("\nDeduplicating GitHub mappings...")
        github_records = [r for r in records if r['source'] == 'github']
        github_df = pd.DataFrame(github_records)
        if not github_df.empty:
            before_dedup = len(github_df)
            github_df_dedup = github_df.drop_duplicates(subset=['csv_path', 'readme_path', 'source'])
            after_dedup = len(github_df_dedup)
            print(f"GitHub dedup: {before_dedup} -> {after_dedup} (removed {before_dedup - after_dedup} duplicates)")
            
            # Update records list: remove old github records and add deduplicated ones
            records = [r for r in records if r['source'] != 'github']
            records.extend(github_df_dedup.to_dict('records'))

    # arXiv
    print("\nLoading arXiv mapping...")
    html_table_path = os.path.join('data', 'processed', 'html_table.parquet')
    if os.path.exists(html_table_path):
        html = pd.read_parquet(html_table_path)
        print("arXiv columns:", html.columns.tolist())
        print("First row of arXiv data:")
        print(html.iloc[0])
        
        if 'html_path' in html.columns and 'table_list' in html.columns:
            arxiv_count = 0
            for _, row in html.iterrows():
                if pd.notna(row['html_path']):
                    table_list = row['table_list']
                    # Expand np.ndarray or list
                    if isinstance(table_list, (list, np.ndarray)):
                        for csv_path in list(table_list):
                            if isinstance(csv_path, str) and csv_path.endswith('.csv'):
                                records.append({
                                    'csv_path': csv_path,
                                    'readme_path': row['html_path'],
                                    'source': 'arxiv'
                                })
                                arxiv_count += 1
            print(f"Added {arxiv_count} arXiv mappings")
        else:
            print("Warning: Missing required columns in arXiv data")
    else:
        print(f"Warning: arXiv file not found at {html_table_path}")

    # Huggingface
    print("\nLoading Huggingface mapping...")
    step2_path = os.path.join('data', 'processed', 'modelcard_step2.parquet')
    hugging_map_path = os.path.join('data', 'processed', 'hugging_deduped_mapping.json')
    if os.path.exists(step2_path) and os.path.exists(hugging_map_path):
        step2 = pd.read_parquet(step2_path, columns=['modelId', 'readme_hash', 'card_readme'])
        with open(hugging_map_path, 'r') as f:
            hugging = json.load(f)
        
        # Use a set to track unique (csv_path, readme_path) combinations to avoid duplicates
        hugging_unique_pairs = set()
        hugging_records = []
        
        hugging_count = 0
        processed_count = 0
        for _, row in step2.iterrows():
            processed_count += 1
            if pd.notna(row['readme_hash']) and row['readme_hash'] in hugging:
                csv_paths = hugging[row['readme_hash']]
                for csv_path in csv_paths:
                    # Create unique pair identifier
                    pair_key = (csv_path, row['card_readme'])
                    if pair_key not in hugging_unique_pairs:
                        hugging_unique_pairs.add(pair_key)
                        hugging_records.append({
                            'csv_path': csv_path,
                            'readme_path': row['card_readme'],
                            'source': 'huggingface'
                        })
                        hugging_count += 1
        
        records.extend(hugging_records)
        print(f"Processed {processed_count} Huggingface models")
        print(f"Added {hugging_count} unique Huggingface mappings (deduplicated by csv_path + readme_path)")

    return pd.DataFrame(records)

def get_file_size(file_path):
    """
    Get file size in human readable format (MB/GB)
    """
    size_bytes = os.path.getsize(file_path)
    size_mb = size_bytes / (1024 * 1024)
    if size_mb > 1024:
        return f"{size_mb/1024:.2f} GB"
    return f"{size_mb:.2f} MB"

def main():
    df = load_mappings()
    print(f"\nTotal records: {len(df)}")
    print("\nRecords by source:")
    print(df['source'].value_counts())
    output_path = os.path.join('data', 'processed', 'raw_csv_to_text_mapping.parquet')
    df.to_parquet(output_path, index=False)
    
    df = pd.read_parquet(output_path)

    # Check exact duplicates
    print("\nChecking for exact duplicates in csv_path:")
    counts = df['csv_path'].value_counts()
    duplicates = counts[counts > 1]
    if duplicates.empty:
        print("No exact duplicates found - this is good!")
    else:
        print(f"Found {duplicates.sum()} total duplicate entries across {len(duplicates)} unique csv_path values")
        print("Duplicate counts per csv_path:")
        print(duplicates)
        
        # Analyze duplicates by source
        print("\nAnalyzing duplicates by source:")
        duplicate_df = df[df['csv_path'].isin(duplicates.index)]
        print("\nDuplicate records by source:")
        print(duplicate_df['source'].value_counts())
        
        # Show example of duplicates
        print("\nExample of duplicate entries:")
        example_csv = duplicates.index[0]
        print(f"\nEntries for {example_csv}:")
        print(df[df['csv_path'] == example_csv][['csv_path', 'readme_path', 'source']].head())

    # Check base table counts
    df['base_path'] = df['csv_path'].apply(lambda x: x.split('_table_')[0] if '_table_' in x else x)
    table_counts = df['base_path'].value_counts()
    num_multi = (table_counts > 1).sum()
    print(f"Number of readmes with multiple tables: {num_multi}")

if __name__ == "__main__":
    main()
