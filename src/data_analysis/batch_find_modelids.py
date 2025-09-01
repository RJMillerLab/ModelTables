#!/usr/bin/env python3
"""
Author: Zhengyuan Dong
Date: 2025-09-01
Description: Batch script to find modelIds for CSV files using DuckDB SQL
Usage:
    python batch_find_modelids.py
    python batch_find_modelids.py -i tmp/top_tables.txt -o tmp/top_tables_with_modelids.txt
"""

import os
import sys
import time
import argparse
import duckdb
from pathlib import Path

def build_sql_query():
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

def process_top_tables_sql(input_file, output_file):
    """Process the input file using SQL query"""
    print(f"🔍 Processing {input_file} using SQL...")
    
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
    
    sql_query = build_sql_query()
    result = conn.execute(sql_query).fetchall()
    
    end_time = time.time()
    query_time = end_time - start_time
    
    print(f"⏱️  SQL query executed in {query_time:.3f} seconds")
    
    # Create a mapping from SQL results
    sql_mapping = {row[0]: row[1] for row in result}
    
    # Write results to output file
    found_count = 0
    not_found_count = 0
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for csv_name, score in top_tables.items():
            model_ids = sql_mapping.get(csv_name, 'NOT_FOUND')
            f.write(f"{csv_name}\t{score}\t{model_ids}\n")
            
            if model_ids == 'NOT_FOUND':
                not_found_count += 1
            else:
                found_count += 1
    
    print(f"\n📝 Results saved to: {output_file}")
    print(f"✅ Found modelIds for: {found_count} files")
    print(f"❌ Not found: {not_found_count} files")
    print(f"⏱️  Total processing time: {query_time:.3f} seconds")
    
    conn.close()
    return output_file, query_time

def main():
    """Main function with command line argument parsing"""
    parser = argparse.ArgumentParser(
        description="Batch script to find modelIds for CSV files using DuckDB SQL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python batch_find_modelids.py
  python batch_find_modelids.py -i my_input.txt -o my_output.txt
  python batch_find_modelids.py --input data/tables.txt --output results/models.txt
        """
    )
    
    parser.add_argument(
        '-i', '--input',
        default='tmp/top_tables.txt',
        help='Input file containing CSV names and scores (default: tmp/top_tables.txt)'
    )
    
    parser.add_argument(
        '-o', '--output',
        default='tmp/top_tables_with_modelids.txt',
        help='Output file for results (default: tmp/top_tables_with_modelids.txt)'
    )
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"❌ Error: Input file '{args.input}' not found!")
        sys.exit(1)
    
    print("🚀 Starting SQL-based batch modelId lookup...")
    print(f"📁 Input file: {args.input}")
    print(f"📁 Output file: {args.output}")
    
    result_file, query_time = process_top_tables_sql(args.input, args.output)
    print(f"🎉 Done! Check {result_file} for results.")
    print(f"⚡ SQL query performance: {query_time:.3f} seconds")

if __name__ == "__main__":
    main()
