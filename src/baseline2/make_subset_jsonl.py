#!/usr/bin/env python3
# make_subset_jsonl.py
# Usage:
#   python make_subset_jsonl.py sparse_top101.tsv \
#          collection_text.jsonl subset.jsonl

import json, sys, pathlib

if len(sys.argv) != 4:
    sys.exit(f"Usage: python {pathlib.Path(__file__).name} "
             "sparse.tsv collection_text.jsonl subset.jsonl")

sparse_tsv, big_jsonl, subset_out = sys.argv[1:]

# 1. collect docids that appear in sparse_top101.tsv
docids = {line.strip().split('\t')[1] for line in open(sparse_tsv)}

# 2. stream through the full collection and keep only those docids
keep = 0
with open(subset_out, 'w', encoding='utf-8') as fout, \
     open(big_jsonl, 'r', encoding='utf-8') as fin:
    for line in fin:
        obj = json.loads(line)
        if obj['id'] in docids:
            fout.write(json.dumps({'id': obj['id'],
                                   'contents': obj['text']},
                                  ensure_ascii=False) + '\n')
            keep += 1

print(f"Wrote {keep} documents to {subset_out}")
