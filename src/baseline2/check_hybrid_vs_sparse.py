#!/usr/bin/env python3
"""Check that hybrid (dense-reranked) results are a subset of sparse-101 candidates.

Usage
-----
python check_hybrid_vs_sparse.py \
  --hybrid   data/tmp/baseline3_hybrid.json \
  --sparse   data/tmp/search_sparse_101.json

The script normalises IDs (adds/removes the optional `.csv` suffix) before
comparison, because the sparse pipeline uses raw IDs while post-processed
results have `.csv` appended.

Output:
  • Summary statistics (fully covered / partially / missing) per query
  • Optional TSV listing queries where hybrid results are *not* fully covered.
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List, Set


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def norm_id(s: str) -> str:
    """Remove trailing `.csv` if present."""
    return s[:-4] if s.endswith(".csv") else s


def load_json(path: str) -> Dict[str, List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Verify hybrid results ⊆ sparse-101")
    ap.add_argument("--hybrid", required=True, help="baseline3_hybrid.json (post-processed)")
    ap.add_argument("--sparse", required=True, help="search_sparse_101.json (raw)")
    ap.add_argument("--report-tsv", default=None, help="Optional TSV file to list queries with missing docs")
    args = ap.parse_args()

    hybrid = load_json(args.hybrid)
    sparse = load_json(args.sparse)

    tot_q = len(hybrid)
    fully, partially, missing = 0, 0, 0
    rows_missing: List[str] = []

    for q, h_docs in hybrid.items():
        q_norm = norm_id(q)
        # sparse key may be stored with or without .csv; try both
        s_docs: List[str] = sparse.get(q_norm, sparse.get(q_norm + ".csv", []))
        s_set: Set[str] = {norm_id(d) for d in s_docs}
        # hybrid docs normalised
        h_norm = [norm_id(d) for d in h_docs]

        # compute missing docs
        miss = [d for d in h_norm if d not in s_set]
        if not miss:
            fully += 1
        elif len(miss) < len(h_norm):
            partially += 1
            rows_missing.append(f"{q}\tPARTIAL\t{','.join(miss)}")
        else:
            missing += 1
            rows_missing.append(f"{q}\tNONE\t-")

    # summary
    print("=== Coverage check"); print(f"Total queries   : {tot_q}")
    print(f"Fully covered   : {fully}  ({fully/tot_q:.2%})")
    print(f"Partially cover : {partially}  ({partially/tot_q:.2%})")
    print(f"No overlap      : {missing}  ({missing/tot_q:.2%})")

    if args.report_tsv and rows_missing:
        out = Path(args.report_tsv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            f.write("query_id\tstatus\tmissing_docs\n")
            for r in rows_missing:
                f.write(r + "\n")
        print(f"Details saved to {out}")


if __name__ == "__main__":
    main() 