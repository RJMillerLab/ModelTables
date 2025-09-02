"""
Author: Zhengyuan Dong
Created: 2025-02-24
Last Modified: 2025-03-11
Description: Load the GitHub README cache data from a parquet file,
store it in an in-memory SQLite database, and allow SQL query.
Usage:
    (get abnormal README.md file) find data/downloaded_github_readmes -type f -exec stat -f "%z %N" {} \; | sort -nr | head -n 10
    python -m src.data_preprocess.step1_query_giturl load --query "data/downloaded_github_readmes/5345c30da0de8c1151b223e13b34dd37.md"
"""

import sys
import duckdb

def query_direct(query_value):
    """Direct query without loading parquet to memory"""
    parquet_file = "data/processed/github_readme_cache.parquet"
    
    sql = f"""
    SELECT raw_url 
    FROM read_parquet('{parquet_file}')
    WHERE downloaded_path = '{query_value}'
    LIMIT 1;
    """
    
    result = duckdb.execute(sql).fetchone()
    
    if result:
        print(result[0])
    else:
        print("Not found")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python query_giturl_direct.py 'your_query_here'")
        sys.exit(1)
    
    query_direct(sys.argv[1])
