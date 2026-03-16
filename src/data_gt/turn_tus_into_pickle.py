"""
Process the groundtruth sqlite file to get the query->candidate dict
Data source: Download from https://github.com/RJMillerLab/table-union-search-benchmark/tree/master, two groundtruth.sqlite files.

Usage:
python -m src.data_localindexing.turn_tus_into_pickle
"""
import sqlite3
import pandas as pd
import os
import pickle

def extract_query_to_candidates(db_path, table_name="att_groundtruth", output_path=None):
    print(f"Processing DB: {db_path}")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    query = f"""
    SELECT query_table, GROUP_CONCAT(DISTINCT candidate_table) as candidates
    FROM {table_name}
    GROUP BY query_table
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    
    result = {}
    for _, row in df.iterrows():
        candidates = row["candidates"]
        if candidates:
            candidate_list = candidates.split(",")
            result[row["query_table"]] = candidate_list
        else:
            result[row["query_table"]] = []
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump(result, f)
        print(f"Saved query->candidate dict to {output_path}")

    return result

if __name__ == "__main__":
    dbs = {
        "tus_small": "tus_small_groundtruth.sqlite",
        "tus_large": "tus_large_groundtruth.sqlite",
    }
    output_dir = "./"

    for name, path in dbs.items():
        out_path = os.path.join(output_dir, f"{name}_query_candidate.pkl")
        extract_query_to_candidates(path, output_path=out_path)

