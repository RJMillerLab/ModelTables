"""Build FAISS index for model card embeddings."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from modelsearch.common.faiss_utils import build_faiss_index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb", required=True, help="Path to embeddings .npy")
    parser.add_argument("--output", default="output/baseline_mc/faiss.index")
    parser.add_argument("--nlist", type=int, default=100)
    args = parser.parse_args()

    emb = np.load(args.emb)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    build_faiss_index(emb, out_path, nlist=args.nlist)


if __name__ == "__main__":
    main()

