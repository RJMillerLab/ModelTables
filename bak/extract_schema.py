#!/usr/bin/env python3
"""
Quickly extract Parquet schema using DuckDB via SQL.
"""

import sys
import duckdb

def extract_parquet_schema(path: str):
    """
    Use DuckDB to extract the schema of a Parquet file via SQL.
    Returns a DuckDB DataFrame with columns: name, type, null.
    """
    con = duckdb.connect()  
    # DESCRIBE SELECT * FROM '...' 会返回列名、类型、是否可空等信息
    schema_df = con.execute(f"DESCRIBE SELECT * FROM '{path}'").df()
    return schema_df

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python extract_schema.py <parquet_path>")
        sys.exit(1)
    parquet_path = sys.argv[1]
    schema = extract_parquet_schema(parquet_path)
    # 打印到控制台
    print(schema.to_string(index=False))

