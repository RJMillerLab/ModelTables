#!/usr/bin/env python3
"""
show_step4_schema.py
--------------------
Print schema information (column names & data types) for
`data/processed/modelcard_step4.parquet`.

Usage:
    python -m src.data_analysis.show_step4_schema

If pyarrow is available, prints full Arrow schema; otherwise falls back to
pandas dtypes.
"""
from __future__ import annotations

import sys
from pathlib import Path

DATA_FILE = Path("data/processed/modelcard_step3_dedup.parquet")

if not DATA_FILE.is_file():
    sys.stderr.write(f"❌ File not found: {DATA_FILE}\n")
    sys.exit(1)

# Prefer pyarrow schema for richer info.
try:
    import pyarrow.parquet as pq  # type: ignore

    schema = pq.read_schema(DATA_FILE)
    print("Arrow schema for", DATA_FILE)
    print(schema)
except Exception as e:  # noqa: BLE001
    # Fallback to pandas dtypes (fast, but less detailed)
    try:
        import pandas as pd  # type: ignore

        df = pd.read_parquet(DATA_FILE, nrows=0)
        print("pandas dtypes for", DATA_FILE)
        print(df.dtypes)
    except Exception as e2:  # noqa: BLE001
        sys.stderr.write(f"Failed to read schema: {e} | fallback: {e2}\n")
        sys.exit(2) 