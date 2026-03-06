#!/usr/bin/env python3
"""
Combined batch script to find modelIds for CSV files and check keywords
Author: Zhengyuan Dong
Date: 2025-11
Description: 
    1. Find modelIds for CSV files using DuckDB SQL
    2. Check card fields for "Label Scheme" and "View label scheme" keywords
    3. Output results in CSV format with keyword columns on the left

Usage:
    python src/data_analysis/batch_process_tables.py
    python src/data_analysis/batch_process_tables.py -i tmp/top_tables.txt -o tmp/top_tables_with_keywords.csv
    python src/data_analysis/batch_process_tables.py --input data/tables.txt --output results/analysis.csv
"""

import os
import sys
import time
import argparse
import pandas as pd
import duckdb
import re
from pathlib import Path

def classify_table_source(fname: str) -> str:
    """
    Classify table file source based on filename pattern
    Returns: 'github', 'huggingface', 'llm', 'html', or 'unknown'
    """
    fname = fname.replace('_s.csv', '.csv').replace('_t.csv', '.csv')
    
    # GitHub: exactly 32 hex chars, then "_table_{digit}.csv"
    if re.fullmatch(r"[0-9a-f]{32}_table_\d+\.csv", fname):
        return "github"
    
    # HTML/S2ORC (ArXiv): e.g. "0705.2450v1_table39.csv" or "1234.5678v2_table3.csv"
    if re.fullmatch(r"\d+\.\d+(?:v\d+)?_table\d+\.csv", fname):
        return "html"
    
    # HuggingFace: 10 hex chars before "_table"
    if re.fullmatch(r"[0-9a-f]{10}_table\d+\.csv", fname):
        return "huggingface"
    
    # LLM-tables: purely digits before "_table" (but not HuggingFace hex patterns)
    if re.fullmatch(r"\d+_table\d+\.csv", fname):
        return "llm"
    
    # Other patterns ending in "_table{digit}.csv"
    if re.fullmatch(r".+_table\d+\.csv", fname):
        return "huggingface"
    
    return "unknown"

def get_full_path(table_file: str) -> str:
    """
    Get the full path for a table file based on its source classification
    """
    source = classify_table_source(table_file)
    
    if source == "github":
        return f"data/processed/deduped_github_csvs/{table_file}"
    elif source == "huggingface":
        return f"data/processed/deduped_hugging_csvs/{table_file}"
    elif source == "llm":
        return f"data/processed/llm_tables/{table_file}"
    elif source == "html":
        return f"data/processed/tables_output/{table_file}"
    else:
        return f"data/processed/unknown/{table_file}"

def build_modelid_sql_query():
    """Build the SQL query to find modelIds for CSV files using step3_merged.parquet"""
    
    sql_query = """
    WITH model_tables AS (
        SELECT 
            modelId,
            unnest(html_table_list_mapped) as html_table,
            unnest(llm_table_list_mapped) as llm_table,
            unnest(github_table_list) as github_table,
            unnest(hugging_table_list) as hugging_table
        FROM read_parquet('data/processed/modelcard_step3_merged.parquet')
        WHERE html_table_list_mapped IS NOT NULL 
           OR llm_table_list_mapped IS NOT NULL 
           OR github_table_list IS NOT NULL 
           OR hugging_table_list IS NOT NULL
    ),
    model_table_mapping AS (
        SELECT DISTINCT
            modelId,
            regexp_extract(html_table, '([^/\\\\]+)$', 1) as table_name
        FROM model_tables
        WHERE html_table IS NOT NULL AND html_table != ''
        
        UNION ALL
        
        SELECT DISTINCT
            modelId,
            regexp_extract(llm_table, '([^/\\\\]+)$', 1) as table_name
        FROM model_tables
        WHERE llm_table IS NOT NULL AND llm_table != ''
        
        UNION ALL
        
        SELECT DISTINCT
            modelId,
            regexp_extract(github_table, '([^/\\\\]+)$', 1) as table_name
        FROM model_tables
        WHERE github_table IS NOT NULL AND github_table != ''
        
        UNION ALL
        
        SELECT DISTINCT
            modelId,
            regexp_extract(hugging_table, '([^/\\\\]+)$', 1) as table_name
        FROM model_tables
        WHERE hugging_table IS NOT NULL AND hugging_table != ''
    )
    SELECT 
        table_name as csv_name,
        string_agg(DISTINCT modelId, '; ') as model_ids
    FROM model_table_mapping
    GROUP BY table_name
    ORDER BY table_name;
    """
    
    return sql_query

def find_modelids_for_tables(input_file):
    """Find modelIds for CSV files using SQL query"""
    print(f"🔍 Finding modelIds for tables in {input_file}...")
    
    # Read the input file to get CSV names and scores
    top_tables = {}
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split('\t')
                if len(parts) >= 2:
                    csv_name = parts[0]
                    score = parts[1]
                    top_tables[csv_name] = score
    
    # Connect to DuckDB
    conn = duckdb.connect()
    
    # Build and execute SQL query
    print("📊 Building and executing SQL query...")
    start_time = time.time()
    
    sql_query = build_modelid_sql_query()
    result = conn.execute(sql_query).fetchall()
    
    end_time = time.time()
    query_time = end_time - start_time
    
    print(f"⏱️  SQL query executed in {query_time:.3f} seconds")
    
    # Create a mapping from SQL results
    sql_mapping = {row[0]: row[1] for row in result}
    
    # Prepare data for keyword checking
    data = []
    found_count = 0
    not_found_count = 0
    
    not_found_tables = []
    for csv_name, score in top_tables.items():
        model_ids = sql_mapping.get(csv_name, 'NOT_FOUND')
        
        if model_ids == 'NOT_FOUND':
            not_found_count += 1
            not_found_tables.append(csv_name)
        else:
            found_count += 1
            # Split multiple modelIds and create separate entries
            model_list = model_ids.split('; ')
            for model_id in model_list:
                if model_id.strip():  # Only add non-empty modelIds
                    data.append({
                        'table_file': csv_name,
                        'value': score,
                        'modelId': model_id.strip(),
                        'source': classify_table_source(csv_name),
                        'full_path': get_full_path(csv_name)
                    })
    
    print(f"✅ Found modelIds for: {found_count} files")
    print(f"❌ Not found: {not_found_count} files")
    print(f"📊 Total model entries to check: {len(data)}")
    
    if not_found_tables:
        print(f"\n🔍 Tables not found in step3_merged (first 10):")
        for i, table in enumerate(not_found_tables[:10]):
            print(f"  {i+1}. {table}")
        if len(not_found_tables) > 10:
            print(f"  ... and {len(not_found_tables) - 10} more")
    
    conn.close()
    return data, query_time

def check_keywords_for_models(data, output_file):
    """Check keywords in card fields for all models"""
    
    # Data file path
    data_path = "data/processed/modelcard_step1.parquet"
    
    if not Path(data_path).exists():
        print(f"❌ Data file does not exist: {data_path}")
        return
    
    print(f"📊 Starting keyword check for {len(data)} models...")
    
    # Extract all modelIds
    model_ids = [item['modelId'] for item in data]
    
    # Use DuckDB for batch query
    conn = duckdb.connect()
    
    try:
        # Build batch query SQL
        model_ids_str = "', '".join(model_ids)
        query = f"""
        SELECT 
            modelId,
            card,
            card_readme,
            downloads
        FROM read_parquet('{data_path}')
        WHERE modelId IN ('{model_ids_str}')
        """
        
        print("🔍 Executing batch SQL query for keywords...")
        results_df = conn.execute(query).fetchdf()
        
        print(f"✅ Query completed, found data for {len(results_df)} models")
        
        # Check keywords
        results_with_keywords = []
        
        for _, row in results_df.iterrows():
            model_id = row['modelId']
            card_content = row['card'] if pd.notna(row['card']) else ""
            card_readme = row['card_readme'] if pd.notna(row['card_readme']) else ""
            downloads = row['downloads']
            
            # Check keywords
            has_label_scheme = "Label Scheme" in card_content or "Label Scheme" in card_readme
            has_view_label_scheme = "View label scheme" in card_content or "View label scheme" in card_readme
            has_both = has_label_scheme and has_view_label_scheme
            
            # Find corresponding original data
            original_data = next((item for item in data if item['modelId'] == model_id), None)
            
            if original_data:
                results_with_keywords.append({
                    'table_file': original_data['table_file'],
                    'value': original_data['value'],
                    'modelId': model_id,
                    'source': original_data['source'],
                    'full_path': original_data['full_path'],
                    'has_label_scheme': has_label_scheme,
                    'has_view_label_scheme': has_view_label_scheme,
                    'has_both': has_both,
                    'downloads': downloads,
                    'card_length': len(card_content) if card_content else 0,
                    'readme_length': len(card_readme) if card_readme else 0
                })
        
        # Save results to CSV file - keyword columns on the left!
        with open(output_file, 'w', encoding='utf-8') as f:
            # Write CSV header - keyword columns on the left
            f.write("table_file,value,modelId,source,full_path,has_label_scheme,has_view_label_scheme,has_both,downloads,card_length,readme_length\n")
            
            # Write CSV data
            for result in results_with_keywords:
                # Handle commas and quotes in CSV
                table_file = result['table_file'].replace(',', ';')  # Replace commas to avoid conflicts
                model_id = result['modelId'].replace(',', ';')  # Replace commas to avoid conflicts
                source = result['source'].replace(',', ';')  # Replace commas to avoid conflicts
                full_path = result['full_path'].replace(',', ';')  # Replace commas to avoid conflicts
                f.write(f"{table_file},{result['value']},{model_id},{source},{full_path},{result['has_label_scheme']},{result['has_view_label_scheme']},{result['has_both']},{result['downloads']},{result['card_length']},{result['readme_length']}\n")
        
        print(f"💾 Results saved to: {output_file}")
        
        # Statistics
        total_checked = len(results_with_keywords)
        with_label_scheme = sum(1 for r in results_with_keywords if r['has_label_scheme'])
        with_view_label_scheme = sum(1 for r in results_with_keywords if r['has_view_label_scheme'])
        with_both = sum(1 for r in results_with_keywords if r['has_both'])
        
        # Source statistics
        source_counts = {}
        for r in results_with_keywords:
            source = r['source']
            source_counts[source] = source_counts.get(source, 0) + 1
        
        print("\n📈 Statistics:")
        print(f"  Total models checked: {total_checked}")
        print(f"  Contains 'Label Scheme': {with_label_scheme}")
        print(f"  Contains 'View label scheme': {with_view_label_scheme}")
        print(f"  Contains both keywords: {with_both}")
        
        print("\n📊 Source Distribution:")
        for source, count in sorted(source_counts.items()):
            print(f"  {source}: {count}")
        
        # Show models with keywords (top 10)
        keyword_models = [r for r in results_with_keywords if r['has_label_scheme'] or r['has_view_label_scheme']]
        if keyword_models:
            print(f"\n🎯 Models with keywords (top 10):")
            for i, model in enumerate(keyword_models[:10]):
                print(f"  {i+1}. {model['modelId']}")
                print(f"     - Label Scheme: {model['has_label_scheme']}")
                print(f"     - View label scheme: {model['has_view_label_scheme']}")
                print(f"     - Downloads: {model['downloads']}")
        
    except Exception as e:
        print(f"❌ Query error: {e}")
    finally:
        conn.close()

def main():
    """Main function with command line argument parsing"""
    parser = argparse.ArgumentParser(
        description="Combined batch script to find modelIds and check keywords",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python batch_process_tables.py
  python batch_process_tables.py -i tmp/top_tables.txt -o tmp/top_tables_with_keywords.csv
  python batch_process_tables.py --input data/tables.txt --output results/analysis.csv
        """
    )
    
    parser.add_argument(
        '-i', '--input',
        default='tmp/top_tables.txt',
        help='Input file containing CSV names and scores (default: tmp/top_tables.txt)'
    )
    
    parser.add_argument(
        '-o', '--output',
        default='tmp/top_tables_with_keywords.csv',
        help='Output CSV file for results (default: tmp/top_tables_with_keywords.csv)'
    )
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"❌ Error: Input file '{args.input}' not found!")
        sys.exit(1)
    
    print("🚀 Starting combined batch processing...")
    print(f"📁 Input file: {args.input}")
    print(f"📁 Output file: {args.output}")
    
    start_time = time.time()
    
    # Step 1: Find modelIds for tables
    data, modelid_time = find_modelids_for_tables(args.input)
    
    if not data:
        print("❌ No modelIds found. Exiting.")
        sys.exit(1)
    
    # Step 2: Check keywords for models
    check_keywords_for_models(data, args.output)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\n🎉 Done! Check {args.output} for results.")
    print(f"⚡ Total processing time: {total_time:.3f} seconds")
    print(f"   - ModelId lookup: {modelid_time:.3f} seconds")
    print(f"   - Keyword checking: {total_time - modelid_time:.3f} seconds")

if __name__ == "__main__":
    main()
