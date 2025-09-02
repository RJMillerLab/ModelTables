#!/usr/bin/env python3
"""
Author: Zhengyuan Dong
Created: 2025-02-23
Last Edited: 2025-09-01
Description: SQL version of step1_analysis.py - Link proportion analysis using DuckDB
Usage:
    python -m src.data_preprocess.step1_analysis_sql
"""

import time
import duckdb
from src.utils import load_config

def run_step1_analysis_sql():
    """Run step1 analysis using SQL queries"""
    print("🔍 Running step1 analysis using SQL...")
    
    # Load config
    config = load_config('config.yaml')
    processed_base_path = config.get('base_path', 'data') + '/processed'
    parquet_file = f"{processed_base_path}/modelcard_step1.parquet"
    
    # Connect to DuckDB
    conn = duckdb.connect()
    
    start_time = time.time()
    
    # SQL query to get all the statistics in one go
    sql_query = f"""
    WITH link_analysis AS (
        SELECT 
            COUNT(*) as total_count,
            SUM(CASE 
                WHEN all_links IS NOT NULL 
                 AND all_links != '' 
                 AND all_links != '[]' 
                THEN 1 ELSE 0 
            END) as all_link_count,
            SUM(CASE 
                WHEN pdf_link IS NOT NULL 
                 AND array_length(pdf_link) > 0
                THEN 1 ELSE 0 
            END) as pdf_link_count,
            SUM(CASE 
                WHEN github_link IS NOT NULL 
                 AND array_length(github_link) > 0
                THEN 1 ELSE 0 
            END) as github_link_count,
            SUM(CASE 
                WHEN (pdf_link IS NULL OR array_length(pdf_link) = 0)
                 AND (github_link IS NOT NULL AND array_length(github_link) > 0)
                THEN 1 ELSE 0 
            END) as no_pdf_has_github_count
        FROM read_parquet('{parquet_file}')
    )
    SELECT 
        total_count,
        all_link_count,
        pdf_link_count,
        github_link_count,
        no_pdf_has_github_count,
        ROUND((all_link_count * 100.0 / total_count), 2) as all_link_ratio,
        ROUND((pdf_link_count * 100.0 / total_count), 2) as pdf_link_ratio,
        ROUND((github_link_count * 100.0 / total_count), 2) as github_link_ratio,
        ROUND((no_pdf_has_github_count * 100.0 / total_count), 2) as no_pdf_has_github_ratio
    FROM link_analysis;
    """
    
    result = conn.execute(sql_query).fetchone()
    
    end_time = time.time()
    query_time = end_time - start_time
    
    # Extract results
    total_count, all_link_count, pdf_link_count, github_link_count, no_pdf_has_github_count, \
    all_link_ratio, pdf_link_ratio, github_link_ratio, no_pdf_has_github_ratio = result
    
    # Output the results
    print(f"Model cards with all links: {all_link_count}/{total_count} = {all_link_ratio:.2f}%")
    print(f"Model cards with GitHub links: {github_link_count}/{total_count} = {github_link_ratio:.2f}%")
    print(f"Model cards with PDF links: {pdf_link_count}/{total_count} = {pdf_link_ratio:.2f}%")
    print(f"Model cards with NO PDF but HAS GitHub links: {no_pdf_has_github_count}/{total_count} = {no_pdf_has_github_ratio:.2f}%")
    print(f"⏱️  SQL query executed in {query_time:.3f} seconds")
    
    conn.close()
    
    return {
        'total_count': total_count,
        'all_link_count': all_link_count,
        'pdf_link_count': pdf_link_count,
        'github_link_count': github_link_count,
        'no_pdf_has_github_count': no_pdf_has_github_count,
        'all_link_ratio': all_link_ratio,
        'pdf_link_ratio': pdf_link_ratio,
        'github_link_ratio': github_link_ratio,
        'no_pdf_has_github_ratio': no_pdf_has_github_ratio,
        'query_time': query_time
    }

if __name__ == "__main__":
    print("🚀 Starting SQL-based step1 analysis...")
    results = run_step1_analysis_sql()
    print("🎉 Analysis completed!")
