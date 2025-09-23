"""Build corpus jsonl for model card retrieval.

Reads modelcard_step1.parquet and writes jsonl containing selected field.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="output/baseline_mc/corpus.jsonl")
    args = parser.parse_args()

    df = load_combined_data("modelcard", file_path="~/Repo/CitationLake/data/raw", columns=['modelId', 'card'])

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fout:
        for _, row in df.iterrows():
            contents = str(row[args.field])
            if not contents or contents.lower() == "nan":
                continue
            obj = {"id": row["modelId"], "contents": contents}
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Wrote corpus jsonl with {len(df)} entries to {out_path}")


if __name__ == "__main__":
    main()
