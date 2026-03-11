#!/usr/bin/env python3
"""
Quick script to get modelID from a single CSV path
Usage: python quick_csv_to_modelid.py --csv "a1b2c3d4e5_table1.csv"
"""

import argparse
import duckdb
import sys

def get_modelid_from_csv(csv_name):
    """Get modelID for a single CSV file"""
    
    # Build SQL query (same as batch_process_tables.py)
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
    WHERE table_name = ?
    GROUP BY table_name
    ORDER BY table_name;
    """
    
    try:
        conn = duckdb.connect()
        result = conn.execute(sql_query, [csv_name]).fetchone()
        conn.close()
        
        if result is None:
            print(f"❌ No modelID found for CSV: {csv_name}")
            return None
        
        csv_name_found, model_ids = result
        print(f"✅ CSV: {csv_name_found}")
        print(f"📋 ModelID(s): {model_ids}")
        return model_ids
        
    except Exception as e:
        print(f"❌ Error querying data: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Quick CSV to ModelID lookup")
    parser.add_argument("--csv", required=True, help="CSV filename (e.g., 'a1b2c3d4e5_table1.csv')")
    
    args = parser.parse_args()
    
    print(f"🔍 Looking up modelID for: {args.csv}")
    print("-" * 50)
    
    model_ids = get_modelid_from_csv(args.csv)
    
    if model_ids:
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
