#!/usr/bin/env python3
"""
Local exact title-to-id batch query.
Uses papers_index (from build_mini_s2orc_es) with EXACT phrase match only – no fuzzy.

Input:  s2orc_titles2ids{suffix}.parquet – extracts titles where query_status NOT in {success, 404}
Output: s2orc_titles2ids_local{suffix}.parquet – same schema: query_title, retrieved_title, paperId, corpusId, paper_identifier, query_status

Requires: ES running with papers_index (build_mini_s2orc_es --mode build).
"""

import argparse
import os
import re
import pandas as pd
from elasticsearch import Elasticsearch
from tqdm import tqdm
from src.utils import to_parquet

ES_HOST = os.getenv("ES_HOST", "http://localhost:9200")
ES_USER = os.getenv("ES_USER", "elastic")
ES_PASSWORD = os.getenv("ES_PASSWORD", "6KdUGb=SifNeWOy__lEz")

CACHE_COLUMNS = ["query_title", "retrieved_title", "paperId", "corpusId", "paper_identifier", "query_status"]
SKIP_RETRY_STATUSES = {"success", "404"}


def _normalize_title(s):
    """Same preprocessing as build_mini_s2orc_es."""
    if not s or not isinstance(s, str):
        return ""
    q = s.strip().lower()
    q = re.sub(r"[^a-z0-9\s\-]", "", q)
    q = re.sub(r"\s+", " ", q)
    return q


def exact_search_paper(es, index_name, query_title):
    """
    Exact phrase match on title.processed (slop=0). Only returns hit if normalized title equals.
    Returns (corpusid, retrieved_title) or (None, None).
    """
    q_str = _normalize_title(query_title)
    if not q_str:
        return None, None
    body = {
        "query": {
            "match_phrase": {
                "title.processed": {"query": q_str, "slop": 0}
            }
        },
        "size": 1
    }
    try:
        resp = es.search(index=index_name, body=body)
    except Exception as e:
        print(f"⚠️ ES error for '{query_title[:50]}...': {e}")
        return None, None
    hits = resp.get("hits", {}).get("hits", [])
    if not hits:
        return None, None
    src = hits[0]["_source"]
    retrieved_title = src.get("title", "").replace("\n", " ")
    corpusid = src.get("corpusid")
    # Exact: only accept if normalized forms match
    if _normalize_title(retrieved_title) != q_str:
        return None, None
    return corpusid, retrieved_title


def main():
    parser = argparse.ArgumentParser(description="Local exact title->id batch query")
    parser.add_argument("--tag", default=None, help="Tag suffix (e.g. 251117)")
    parser.add_argument("--index_name", default="papers_index", help="ES index (papers_index from build_mini_s2orc_es)")
    parser.add_argument("--input_file", default=None, help="Override s2orc_titles2ids path")
    parser.add_argument("--output_file", default=None, help="Override s2orc_titles2ids_local path")
    args = parser.parse_args()

    suffix = f"_{args.tag}" if args.tag else ""
    data_dir = "data/processed"
    input_path = args.input_file or os.path.join(data_dir, f"s2orc_titles2ids{suffix}.parquet")
    output_path = args.output_file or os.path.join(data_dir, f"s2orc_titles2ids_local{suffix}.parquet")

    if not os.path.exists(input_path):
        print(f"❌ Input not found: {input_path}")
        return

    df = pd.read_parquet(input_path)
    if "query_status" not in df.columns:
        df["query_status"] = "success"
    done_mask = df["query_status"].isin(SKIP_RETRY_STATUSES)
    done_titles = set(df.loc[done_mask, "query_title"].dropna().astype(str).tolist())

    # Titles to query: not success and not 404
    all_titles = set(df["query_title"].dropna().astype(str).tolist())
    titles_to_query = sorted(all_titles - done_titles)
    if not titles_to_query:
        print("✅ No titles to query (all done or 404). Exiting.")
        return

    print(f"🔍 {len(titles_to_query)} titles to query locally (exact match)")

    es = Elasticsearch(
        ES_HOST,
        basic_auth=(ES_USER, ES_PASSWORD),
        verify_certs=False,
    )
    if not es.indices.exists(index=args.index_name):
        print(f"❌ Index '{args.index_name}' does not exist. Run build_mini_s2orc_es --mode build first.")
        return

    rows = []
    for query_title in tqdm(titles_to_query, desc="Exact match"):
        corpusid, retrieved_title = exact_search_paper(es, args.index_name, query_title)
        if corpusid is not None:
            cid = str(corpusid)
            pid = f"CorpusID:{cid}"
            rows.append({
                "query_title": query_title,
                "retrieved_title": retrieved_title,
                "paperId": cid,
                "corpusId": cid,
                "paper_identifier": pid,
                "query_status": "success",
            })
        else:
            rows.append({
                "query_title": query_title,
                "retrieved_title": None,
                "paperId": None,
                "corpusId": None,
                "paper_identifier": None,
                "query_status": "no_results",
            })

    out_df = pd.DataFrame(rows, columns=CACHE_COLUMNS)
    to_parquet(out_df, output_path)
    n_ok = (out_df["query_status"] == "success").sum()
    print(f"💾 Saved {len(out_df)} rows to {output_path} ({n_ok} exact matches)")


if __name__ == "__main__":
    main()
