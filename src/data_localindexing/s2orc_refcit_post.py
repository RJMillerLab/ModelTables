"""
Convert full_hits.jsonl (local citation edges) to two parquet files
that mimic S2ORC API ref/cit cache outputs with STRICT 1-row-per-whitelist-ID rule.

Author: Zhengyuan Dong
Created: 2025-05-04
Last Edited: 2026-03-09
"""

import argparse
import json
import pandas as pd
from pathlib import Path
from collections import defaultdict
from src.utils import to_parquet


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert full_hits.jsonl to S2ORC-style ref/cit parquet files")
    parser.add_argument("--tag", default=None, help="Tag suffix (e.g. 251117). Outputs s2orc_*_{tag}.parquet")
    args = parser.parse_args()

    data_dir = Path("data/processed")
    suffix = f"_{args.tag}" if args.tag else ""

    input_jsonl = data_dir / f"full_hits{suffix}.jsonl"
    id_list_txt = data_dir / f"tmp_local_ids{suffix}.txt"
    citations_pq = data_dir / f"s2orc_citations_cache{suffix}.parquet"
    references_pq = data_dir / f"s2orc_references_cache{suffix}.parquet"

    with id_list_txt.open() as f:
        whitelist = [line.strip() for line in f if line.strip()]
    whitelist_set = set(whitelist)
    print(f"whitelist size = {len(whitelist)}")

    cit_bucket, ref_bucket = defaultdict(list), defaultdict(list)
    miss_count = 0
    with input_jsonl.open() as f:
        for line in f:
            rec = json.loads(line)
            try:
                citing = str(rec["citingcorpusid"])
                cited = str(rec["citedcorpusid"])
            except KeyError:
                miss_count += 1
                continue
            if citing in whitelist_set:
                ref_bucket[citing].append(rec)
            if cited in whitelist_set:
                cit_bucket[cited].append(rec)
    print(f"missed corpusId count = {miss_count}")

    def build_cache(bucket: dict):
        rows = []
        for cid in whitelist:
            rows.append(
                {
                    "corpusId": str(cid),
                    "original_response": json.dumps({"data": bucket[cid]}, ensure_ascii=False),
                }
            )
        return pd.DataFrame(rows)

    df_cit = build_cache(cit_bucket)
    df_ref = build_cache(ref_bucket)
    to_parquet(df_cit, citations_pq)
    to_parquet(df_ref, references_pq)
    print("citations rows =", len(df_cit), "->", citations_pq)
    print("references rows =", len(df_ref), "->", references_pq)
