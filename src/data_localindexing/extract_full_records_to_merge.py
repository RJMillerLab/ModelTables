"""
Convert full_hits.jsonl (local citation edges) to three parquet files
that mimic s2orc_API_query – with STRICT 1‑row‑per‑whitelist‑ID rule.

Author: Zhengyuan Dong 
Created: 2025‑05‑04
Last Edited: 2025‑05‑05
"""

import json, pandas as pd
from pathlib import Path
from collections import defaultdict
from src.utils import to_parquet

# ---------- ❶ Path ----------
DATA_DIR          = Path("data/processed")
INPUT_JSONL       = DATA_DIR / "full_hits.jsonl"
ID_LIST_TXT       = DATA_DIR / "tmp_local_ids.txt"        # ← whitelist

CITATIONS_PQ      = DATA_DIR / "s2orc_citations_cache.parquet"
REFERENCES_PQ     = DATA_DIR / "s2orc_references_cache.parquet"
QUERY_RESULTS_PQ  = DATA_DIR / "s2orc_query_results.parquet"
TITLES_CACHE_FILE = DATA_DIR / "s2orc_titles2ids.parquet"

if __name__ == "__main__":
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
            rows.append({
                "corpusId": str(cid),
                "original_response": json.dumps({"data": lst}, ensure_ascii=False)
            })
        return pd.DataFrame(rows)
    df_cit = build_cache(cit_bucket, "citing_papers")
    df_ref = build_cache(ref_bucket, "cited_papers")
    to_parquet(df_cit, CITATIONS_PQ)
    to_parquet(df_ref, REFERENCES_PQ)
    print("✅  citations rows =", len(df_cit),  "→", CITATIONS_PQ)
    print("✅  references rows =", len(df_ref),  "→", REFERENCES_PQ)
    # ---------- ❺ stub: keep one empty row per whitelist id ----------
    # File paths
    from src.data_preprocess.s2orc_API_query import merge_all_results
    stub = merge_all_results(titles_cache=TITLES_CACHE_FILE,
                                    citations_cache=CITATIONS_PQ,
                                    references_cache=REFERENCES_PQ,
                                    output_file=QUERY_RESULTS_PQ,
                                    MERGE_KEY = "corpusId")
    print("✅  stub rows =", len(stub), "→", QUERY_RESULTS_PQ)
