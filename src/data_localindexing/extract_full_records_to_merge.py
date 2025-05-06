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

# ---------- ❶ Path ----------
DATA_DIR          = Path("data/processed")
INPUT_JSONL       = DATA_DIR / "full_hits.jsonl"
ID_LIST_TXT       = DATA_DIR / "tmp_local_ids.txt"        # ← whitelist

CITATIONS_PQ      = DATA_DIR / "s2orc_citations_cache.parquet"
REFERENCES_PQ     = DATA_DIR / "s2orc_references_cache.parquet"
QUERY_RESULTS_PQ  = DATA_DIR / "s2orc_query_results.parquet"

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
                citing = str(rec["citingcorpusid"])   ######## ← no get()
                cited  = str(rec["citedcorpusid"])    ######## ← no get()
            except KeyError as e:
                miss_count += 1
                continue
            if citing in WL_set:
                cit_bucket[citing].append(rec)
            if cited in WL_set:
                ref_bucket[cited].append(rec)
    print(f"❗ missed corpusId count = {miss_count}")
    # ---------- ❹ Convert to dataframe ----------
    def build_cache(bucket: dict, key_name: str):
        rows = []
        for cid in WL:
            lst = bucket[cid]  
            rows.append({
                "corpusId": str(cid),
                "original_response": json.dumps({"data": lst}, ensure_ascii=False),
                "parsed_response":  json.dumps({key_name: lst}, ensure_ascii=False)
            })
        return pd.DataFrame(rows)
    df_cit = build_cache(cit_bucket, "citing_papers")
    df_ref = build_cache(ref_bucket, "cited_papers")
    df_cit.to_parquet(CITATIONS_PQ , index=False)
    df_ref.to_parquet(REFERENCES_PQ, index=False)
    print("✅  citations rows =", len(df_cit),  "→", CITATIONS_PQ)
    print("✅  references rows =", len(df_ref),  "→", REFERENCES_PQ)
    # ---------- ❺ stub: keep one empty row per whitelist id ----------
    # File paths
    prefix = "" #"_429"
    TITLES_CACHE_FILE = f"data/processed/s2orc_titles2ids{prefix}.parquet"  ######## Cache file for titles mapping
    CITATIONS_CACHE_FILE = f"data/processed/s2orc_citations_cache{prefix}.parquet"  ######## Cache file for citations
    REFERENCES_CACHE_FILE = f"data/processed/s2orc_references_cache{prefix}.parquet"  ######## Cache file for references
    MERGED_RESULTS_FILE = f"data/processed/s2orc_query_results{prefix}.parquet"  ######## Output file for merged results
    from src.data_preprocess.s2orc_API_query import merge_all_results
    stub = merge_all_results(titles_cache=TITLES_CACHE_FILE,
                                    citations_cache=CITATIONS_CACHE_FILE,
                                    references_cache=REFERENCES_CACHE_FILE,
                                    output_file=MERGED_RESULTS_FILE,
                                    MERGE_KEY = "corpusId")
    print("✅  stub rows =", len(stub), "→", MERGED_RESULTS_FILE)
