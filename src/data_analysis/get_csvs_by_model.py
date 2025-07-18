#!/usr/bin/env python3
"""
Author: AI added via assistant
Date: 2025-07-18

Utility: Given a modelId (e.g. "bert"), print the entire row from the parquet file.

Data source: data/processed/modelcard_step3_dedup.parquet

Usage:
    python -m src.data_analysis.get_csvs_by_model --model bert

Exit code is non-zero if the modelId is not found.
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

DATA_PATH = Path("data/processed/modelcard_step3_dedup.parquet")

def load_dataframe(path: Path) -> pd.DataFrame:
    if not path.is_file():
        sys.stderr.write(f"❌ Parquet file not found: {path}\n")
        sys.exit(1)
    return pd.read_parquet(path)

def main(model_id: str):
    df = load_dataframe(DATA_PATH)
    row_match = df.loc[df["modelId"] == model_id]
    if row_match.empty:
        sys.stderr.write(f"❌ modelId '{model_id}' not found in {DATA_PATH}\n")
        sys.exit(2)
    
    print(row_match.to_dict('records')[0])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print the entire dataframe row for the modelId.")
    parser.add_argument("--model", required=True, help="modelId to query, e.g. 'bert'")
    args = parser.parse_args()
    main(args.model) 