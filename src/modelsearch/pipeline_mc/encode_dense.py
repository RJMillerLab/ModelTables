"""Encode model card corpus to numpy embeddings using SBERT.

Usage examples:
python encode_dense.py --jsonl output/baseline_mc/corpus.jsonl --output output/baseline_mc/embeddings.npy
python encode_dense.py --parquet data/processed/modelcard_step1.parquet --field card_readme --output output/baseline_mc/embeddings_readme.npy
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from src.modelsearch.common.encode import encode_texts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", help="Path to corpus jsonl (id, contents)")
    parser.add_argument("--parquet", help="Parquet file path (contains field column)")
    parser.add_argument("--field", default="card", help="Column name to read from parquet if --parquet provided")
    parser.add_argument("--output", required=True, help="Output .npy path for embeddings")
    parser.add_argument("--model", default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if not args.jsonl and not args.parquet:
        raise ValueError("Either --jsonl or --parquet must be specified")

    texts: List[str]
    if args.jsonl:
        import json

        with Path(args.jsonl).open() as f:
            texts = [json.loads(line)["contents"] for line in f]
    else:
        df = pd.read_parquet(Path(args.parquet), columns=[args.field])
        texts = df[args.field].astype(str).tolist()

    embeddings = encode_texts(
        texts, model_name=args.model, batch_size=args.batch_size, device=args.device
    )
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, embeddings)
    print(f"Saved {embeddings.shape} embeddings to {out_path}")


if __name__ == "__main__":
    main()
