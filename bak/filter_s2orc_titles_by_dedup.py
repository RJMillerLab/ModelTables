# -*- coding: utf-8 -*-
"""
Create _4 from _3: filter by dedup_titles (normalize match), success first.
Separate from merge - run after merge produces _3.

Logic:
  - KEEP: (1) in dedup_titles, OR (2) has definitive query result (success or 404)
  - REMOVE: only rows that are NOT in dedup AND have no definitive result (429, timeout, etc.)
  - Match: normalize. Keep: full original query_title.
"""
import os
import json
import argparse
import pandas as pd
from src.utils import load_config, to_parquet
from src.data_preprocess.title_dedup_utils import normalize

# Statuses that count as "has query result" (definitive answer) — do NOT delete these
HAS_RESULT_STATUSES = {"success", "404"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="251117")
    parser.add_argument("--input", default=None, help="Default: s2orc_titles2ids_{tag}_3.parquet")
    parser.add_argument("--output", default=None, help="Default: s2orc_titles2ids_{tag}_4.parquet")
    args = parser.parse_args()

    config = load_config("config.yaml")
    base = os.path.join(config.get("base_path", "data"), "processed")
    tag = args.tag
    input_path = args.input or os.path.join(base, f"s2orc_titles2ids_{tag}_3.parquet")
    output_path = args.output or os.path.join(base, f"s2orc_titles2ids_{tag}_4.parquet")
    dedup_path = os.path.join(base, f"modelcard_dedup_titles_{tag}.json")

    df = pd.read_parquet(input_path)
    with open(dedup_path) as f:
        dedup_titles = json.load(f)
    dedup_norm = set(normalize(t) for t in dedup_titles if t)

    df["_norm"] = df["query_title"].astype(str).apply(normalize)
    in_dedup = df["_norm"].isin(dedup_norm)
    if "query_status" in df.columns:
        has_result = df["query_status"].astype(str).isin(HAS_RESULT_STATUSES)
    else:
        has_result = pd.Series(True, index=df.index)  # no status: conservatively keep all
    keep_mask = in_dedup | has_result
    removed = (~keep_mask).sum()
    if removed > 0:
        print(f"Removed {removed} rows (not in dedup and no success/404)")
    df = df.loc[keep_mask].drop(columns=["_norm"])

    status_order = {"success": 0, "404": 1, "429": 2, "no_results": 3, "timeout": 4, "request_error": 5, "exceeded_retries": 6}
    if "query_status" in df.columns:
        df["_order"] = df["query_status"].map(lambda s: status_order.get(str(s), 99))
        df = df.sort_values("_order").drop(columns=["_order"])

    to_parquet(df, output_path)
    print(f"Saved {len(df)} rows to {output_path}")


if __name__ == "__main__":
    main()
