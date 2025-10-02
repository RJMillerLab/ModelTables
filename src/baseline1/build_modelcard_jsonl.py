"""
Author: Zhengyuan Dong
Created: 2025-10-02
Last Edited: 2025-10-02
Description: Build JSONL corpus from model cards for baseline1 encode/build_faiss/search.

Each line: {"id": <modelId>, "contents": <text>}.
"""

import os
import sys
import json
import argparse
from typing import Optional

import pandas as pd
import duckdb
import pyarrow as pa

from src.utils import load_combined_data  # type: ignore

def _ensure_text(x: Optional[str]) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x.strip()
    try:
        return str(x).strip()
    except Exception:
        return ""


def build_jsonl_from_parquet(parquet_path: str, field: str, output_jsonl: str) -> None:
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Parquet not found: {parquet_path}")
    if field not in {"card", "card_readme"}:
        raise ValueError("field must be 'card' or 'card_readme'")

    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)
    con = duckdb.connect()
    try:
        query = f"""
        SELECT CAST(modelId AS VARCHAR) AS id,
               {field} AS contents
        FROM read_parquet('{parquet_path}')
        WHERE {field} IS NOT NULL AND length(trim({field})) > 0
        """
        table = con.execute(query).fetch_arrow_table()

        written = 0
        with open(output_jsonl, "w", encoding="utf-8") as fout:
            for batch in table.to_batches():
                ids = batch.column("id").to_pylist()
                contents = batch.column("contents").to_pylist()
                for mid, text in zip(ids, contents):
                    text_s = _ensure_text(text)
                    if not mid or not text_s:
                        continue
                    doc = {"id": str(mid), "contents": text_s}
                    fout.write(json.dumps(doc, ensure_ascii=False) + "\n")
                    written += 1
        print(f"Wrote {written} documents to {output_jsonl}")
    finally:
        con.close()


def build_jsonl_from_raw(raw_dir: str, field: str, output_jsonl: str) -> None:
    if field not in {"card", "card_readme"}:
        raise ValueError("field must be 'card' or 'card_readme'")

    # Prefer using load_combined_data to read raw shards
    df = load_combined_data(
        data_type="modelcard",
        file_path=raw_dir,
        columns=[],  # let it load default columns; we'll project below
    )

    cols = set(df.columns)
    # Resolve id column
    id_col = None
    for cand in ("modelId", "datasetId", "model_id", "id"):
        if cand in cols:
            id_col = cand
            break
    if id_col is None:
        raise ValueError("Could not find an ID column in raw shards (tried modelId/datasetId/model_id/id)")

    # Resolve text field
    field_col = field if field in cols else None
    if field_col is None:
        alt = "card" if field == "card_readme" else "card_readme"
        if alt in cols:
            field_col = alt
        else:
            raise ValueError(f"Neither '{field}' nor its alternative found in raw shards. Columns seen: {sorted(cols)}")

    df = df[[id_col, field_col]].copy()
    df = df[df[field_col].notna()]

    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)
    written = 0
    with open(output_jsonl, "w", encoding="utf-8") as fout:
        for _, row in df.iterrows():
            mid = str(row[id_col])
            text_s = _ensure_text(row[field_col])
            if not mid or not text_s:
                continue
            doc = {"id": mid, "contents": text_s}
            fout.write(json.dumps(doc, ensure_ascii=False) + "\n")
            written += 1
    print(f"Wrote {written} documents to {output_jsonl}")


def main():
    parser = argparse.ArgumentParser(description="Build JSONL corpus from model cards for baseline1")
    parser.add_argument("--field", choices=["card", "card_readme"], required=True,
                        help="Which text field to use as contents. 'card' uses raw shards; 'card_readme' uses step1.parquet")
    parser.add_argument("--output_jsonl", required=True, help="Output JSONL path")
    # Optional overrides (not required)
    parser.add_argument("--parquet", default="data/processed/modelcard_step1.parquet",
                        help="Path to modelcard_step1.parquet (used when field=card_readme)")
    parser.add_argument("--raw_dir", default="data/raw",
                        help="Directory to raw shards (used when field=card)")

    args = parser.parse_args()

    if args.field == "card_readme":
        build_jsonl_from_parquet(args.parquet, args.field, args.output_jsonl)
    else:
        build_jsonl_from_raw(args.raw_dir, args.field, args.output_jsonl)


if __name__ == "__main__":
    main()


