#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert a sparse-search result stored as a single JSON object

    {
      "qid_1": ["doc1", "doc2", ...],
      "qid_2": ["docA", "docB", ...],
      ...
    }

into a TSV file with two columns  qid <TAB> docid .
The TSV can be fed to the dense-encoding / Faiss pipeline.

By default the first hit of every query is skipped (often the query
table pointing to itself).  If you want to keep all hits, change
`docs[1:]` to `docs`.

Usage
-----
    python filter_sparse_to_dense_dict.py \
           input_sparse.json \
           output_pairs.tsv
"""
import json
import sys
from pathlib import Path

if len(sys.argv) != 3:
    sys.exit(
        f"Usage: python {Path(__file__).name} "
        "input_sparse.json output.tsv\n"
        "Example:\n"
        "  python filter_sparse_to_dense_dict.py "
        "data/tmp/search_sparse_101.json "
        "data/tmp/sparse_top101.tsv"
    )

input_path, output_path = sys.argv[1], sys.argv[2]

# Load the whole dictionary
with open(input_path, "r", encoding="utf-8") as f_in:
    sparse_dict = json.load(f_in)

pair_cnt = 0
with open(output_path, "w", encoding="utf-8") as f_out:
    for qid, docs in sparse_dict.items():
        # Skip the first hit; use docs instead of docs[1:]
        # if you want to keep every document.
        for docid in docs[1:]:
            f_out.write(f"{qid}\t{docid}\n")
            pair_cnt += 1

print(f"Wrote {pair_cnt} qid-docid pairs to '{output_path}'")
