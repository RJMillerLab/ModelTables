#!/usr/bin/env python3
"""
Author: AI added via assistant
Date: 2025-07-18

Utility: Given a modelId (e.g. "bert"), print the entire row from the parquet file using DuckDB SQL.

Data source: data/processed/modelcard_step3_merged.parquet

Usage:
    python -m src.data_analysis.get_csvs_by_model --model bert

Exit code is non-zero if the modelId is not found.
"""

import argparse
import sys
import duckdb
import json

DATA_PATH = "data/processed/modelcard_step3_merged.parquet"

def main(model_id: str):
    # 使用DuckDB SQL查询
    query = f"""
    SELECT *
    FROM read_parquet('{DATA_PATH}')
    WHERE modelId = '{model_id}'
    LIMIT 1;
    """
    
    try:
        result = duckdb.execute(query).fetchone()
        if result is None:
            sys.stderr.write(f"❌ modelId '{model_id}' not found in {DATA_PATH}\n")
            sys.exit(2)
        
        # 获取列名
        columns = duckdb.execute(f"DESCRIBE SELECT * FROM read_parquet('{DATA_PATH}') LIMIT 0").fetchall()
        column_names = [col[0] for col in columns]
        
        # 创建字典并打印
        row_dict = dict(zip(column_names, result))
        print(json.dumps(row_dict, indent=2, default=str))
        
    except Exception as e:
        sys.stderr.write(f"❌ Error querying data: {e}\n")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print the entire dataframe row for the modelId using DuckDB SQL.")
    parser.add_argument("--model", required=True, help="modelId to query, e.g. 'bert'")
    args = parser.parse_args()
    main(args.model)