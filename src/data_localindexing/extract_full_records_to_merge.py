"""
Convert full_hits.jsonl (local citation edges) to three parquet files
that mimic s2orc_API_query – with STRICT 1‑row‑per‑whitelist‑ID rule.

Author: Zhengyuan Dong 
Created: 2025‑05‑04
Last Edited: 2025‑05‑05
"""

import argparse
import json
import pandas as pd
from pathlib import Path
from collections import defaultdict
from src.utils import to_parquet


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert full_hits.jsonl to S2ORC-style parquet files")
    parser.add_argument("--tag", default=None, help="Tag suffix (e.g. 251117). Outputs s2orc_*_{tag}.parquet")
    args = parser.parse_args()
    # ---------- ❶ Path ----------
    DATA_DIR          = Path("data/processed")
    suffix = f"_{args.tag}" if args.tag else ""

    INPUT_JSONL       = DATA_DIR / f"full_hits{suffix}.jsonl"
    ID_LIST_TXT       = DATA_DIR / f"tmp_local_ids{suffix}.txt"        # ← whitelist
    CITATIONS_PQ      = DATA_DIR / f"s2orc_citations_cache{suffix}.parquet"
    REFERENCES_PQ     = DATA_DIR / f"s2orc_references_cache{suffix}.parquet"
    TITLES_CACHE_FILE = DATA_DIR / f"s2orc_titles2ids{suffix}.parquet"
    # ---------- ❷ Read whitelist ----------
    with ID_LIST_TXT.open() as f:
        WL = [line.strip() for line in f if line.strip()]
    WL_set = set(WL)
    print(f"➡  whitelist size = {len(WL)}")
    # ---------- ❸ Scan JSONL ----------
    cit_bucket, ref_bucket = defaultdict(list), defaultdict(list)
    miss_count = 0
    with INPUT_JSONL.open() as f:
        for line in f:
            rec = json.loads(line)
            try:
                citing = str(rec["citingcorpusid"])
                cited  = str(rec["citedcorpusid"])
            except KeyError as e:
                miss_count += 1
                continue
            if citing in WL_set:
                ref_bucket[citing].append(rec)
            if cited in WL_set:
                cit_bucket[cited].append(rec)
    print(f"❗ missed corpusId count = {miss_count}")
    # ---------- ❹ Convert to dataframe ----------
    def build_cache(bucket: dict, key_name: str):
        rows = []
        for cid in WL:
            lst = bucket[cid]
            rows.append({"corpusId": str(cid), "original_response": json.dumps({"data": lst}, ensure_ascii=False)})
        return pd.DataFrame(rows)
    df_cit = build_cache(cit_bucket, "citing_papers")
    df_ref = build_cache(ref_bucket, "cited_papers")
    to_parquet(df_cit, CITATIONS_PQ)
    to_parquet(df_ref, REFERENCES_PQ)
    print("✅  citations rows =", len(df_cit),  "→", CITATIONS_PQ)
    print("✅  references rows =", len(df_ref),  "→", REFERENCES_PQ)
