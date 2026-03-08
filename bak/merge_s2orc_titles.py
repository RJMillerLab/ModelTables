"""
Author: Zhengyuan Dong
Created: 2026-03-08
Description: Merge two s2orc_titles2ids parquet files, prioritizing successful queries over failures. (save some searched results, avoid re-querying)
Usage:
    python -m src.data_preprocess.merge_s2orc_titles --file1 data/processed/s2orc_titles2ids_251117.parquet --file2 data/processed/s2orc_titles2ids_251117_2.parquet --output data/processed/s2orc_titles2ids_251117_3.parquet
"""

import pandas as pd
import argparse
from src.utils import to_parquet

def merge_s2orc_titles(file1, file2, output_file):
    df1 = pd.read_parquet(file1)
    df2 = pd.read_parquet(file2)
    
    # Ensure df1 has query_status
    if 'query_status' not in df1.columns:
        df1['query_status'] = 'success'
    
    # Create dicts for quick lookup
    dict1 = {row['query_title']: row for _, row in df1.iterrows()}
    dict2 = {row['query_title']: row for _, row in df2.iterrows()}
    
    all_titles = set(dict1.keys()) | set(dict2.keys())
    merged_rows = []
    
    status_priority = {'success': 3, '404': 2, 'no_results': 1, 'exceeded_retries': 1, '429': 1, 'timeout': 1, 'request_error': 1, 'no_paper_id': 1}
    
    for title in all_titles:
        row1 = dict1.get(title)
        row2 = dict2.get(title)
        
        if row1 is None:
            merged_rows.append(row2.to_dict())
        elif row2 is None:
            merged_rows.append(row1.to_dict())
        else:
            # Both have the title
            status1 = row1.get('query_status', 'success')  # assume success if no status
            status2 = row2.get('query_status', 'success')
            
            pri1 = status_priority.get(status1, 0)
            pri2 = status_priority.get(status2, 0)
            
            if pri1 > pri2:
                merged_rows.append(row1.to_dict())
            elif pri2 > pri1:
                merged_rows.append(row2.to_dict())
            else:
                # Same priority, prefer success, or check ID
                if status1 == 'success' and status2 == 'success':
                    # Check if IDs match
                    id1 = row1['paperId'] if pd.notna(row1['paperId']) else row1['corpusId']
                    id2 = row2['paperId'] if pd.notna(row2['paperId']) else row2['corpusId']
                    if id1 == id2:
                        merged_rows.append(row1.to_dict())  # or row2, same
                    else:
                        print(f"ID mismatch for {title}: {id1} vs {id2}, using file1")
                        merged_rows.append(row1.to_dict())
                else:
                    # Same status, use file1
                    merged_rows.append(row1.to_dict())
    
    df_merged = pd.DataFrame(merged_rows)
    print(f"Merged DataFrame has {len(df_merged)} rows")
    to_parquet(df_merged, output_file)
    print(f"Merged {len(all_titles)} titles into {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge two s2orc_titles2ids parquet files")
    parser.add_argument("--file1", required=True, help="Path to first parquet file")
    parser.add_argument("--file2", required=True, help="Path to second parquet file")
    parser.add_argument("--output", required=True, help="Path to output parquet file")
    args = parser.parse_args()
    
    merge_s2orc_titles(args.file1, args.file2, args.output)