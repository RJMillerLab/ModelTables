#!/usr/bin/env python
"""card_statistics.py – fast step-by-step stats using DuckDB.
Outputs step_by_step_filtering_statistics.pdf plus JSON counts to stdout.

Now supports a --tag argument to switch between snapshots:
- No tag (default): data/raw/train-*-of-00004.parquet
- --tag 251117    : data/raw_251117/train-*-of-00006.parquet

Dataset cards (datasetcard-train-*.parquet) are automatically excluded
by using the train-*.parquet pattern (which only matches model cards).
"""

import argparse
import json
import os

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


VALID_CARD_COND = "card IS NOT NULL AND card <> '' AND card <> 'Entry not found'"


def main() -> None:
    parser = argparse.ArgumentParser(description="Step-by-step statistics of model cards using DuckDB.")
    parser.add_argument("--tag", type=str, default=None, help="Snapshot tag (e.g., 251117).")
    args = parser.parse_args()
    suffix = f"_{args.tag}" if args.tag else ""

    # -------- paths / config --------
    # Untagged: historical default snapshot, 4 shards
    #   data/raw/train-*-of-00004.parquet
    # Tagged (e.g. 251117): new snapshot layout, 6 shards
    #   data/raw_<tag>/train-*-of-00006.parquet
    if args.tag:
        parquet_glob = os.path.join("data", f"raw{suffix}", "train-*-of-00006.parquet")
    else:
        parquet_glob = os.path.join("data", "raw", "train-*-of-00004.parquet")

    print(f"Parquet glob={parquet_glob}")

    # -------- duckdb connection --------
    con = duckdb.connect()

    def q(sql: str) -> int:
        """Run a scalar DuckDB query and return single int."""
        return con.execute(sql).fetchone()[0]

    base_clause = f"FROM read_parquet('{parquet_glob}') WHERE {VALID_CARD_COND}"

    # Sanity-check columns (zero-row SELECT, fast)
    cols = set(
        con.execute(f"SELECT * FROM read_parquet('{parquet_glob}') LIMIT 0").fetchdf().columns
    )
    print(f"Columns in raw parquet: {sorted(cols)}")

    # ----- counts dict -----
    counts: dict[str, int] = {}

    # 1) all repositories (all rows in raw snapshot)
    counts["All"] = q(f"SELECT COUNT(*) FROM read_parquet('{parquet_glob}')")

    # 2) non-empty model cards (after filtering out 'Entry not found')
    counts["Non-empty model cards"] = q(f"SELECT COUNT(*) {base_clause}")

    # 3) unique model cards – keep row with highest likes per card
    dedup_query = f"""
        SELECT * {base_clause}
        QUALIFY row_number() OVER (PARTITION BY card ORDER BY likes DESC) = 1
    """
    counts["Unique model cards"] = q(f"SELECT COUNT(*) FROM ({dedup_query})")

    # 4) downloads > 0
    counts["Downloads > 0"] = q(
        f"SELECT COUNT(*) FROM ({dedup_query}) WHERE downloads > 0"
    )

    # ---------- print JSON ----------
    print(json.dumps(counts, indent=2))

    # ---------- preview first two records ----------
    sample_df = con.execute(f"SELECT * {base_clause} LIMIT 2").fetchdf()
    print("\nPreview of first two rows (truncated):")
    with pd.option_context("display.max_columns", None, "display.max_colwidth", 120):
        print(sample_df)

    # ---------- plot ----------
    plt.figure(figsize=(12, 8))
    bars = plt.bar(
        counts.keys(),
        counts.values(),
        color=plt.cm.Blues(np.linspace(0.8, 0.4, len(counts))),
    )
    # Put numeric labels on top of each bar (with thousand separators)
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{int(height):,}",
            ha="center",
            va="bottom",
            fontsize=14,
        )
    plt.ylabel("Number of Model Repositories", fontsize=16)
    plt.xlabel("Filtering Steps", fontsize=16)
    plt.xticks(rotation=20, ha="right", fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()

    suffix = f"_{args.tag}" if args.tag else ""
    plt.savefig(f"data/analysis/step_by_step_filtering_statistics{suffix}.pdf", format="pdf", dpi=300, bbox_inches="tight")
    plt.savefig(f"data/analysis/step_by_step_filtering_statistics{suffix}.png", format="png", dpi=300, bbox_inches="tight")
    print(f"save fig to data/analysis/step_by_step_filtering_statistics{suffix}.pdf and data/analysis/step_by_step_filtering_statistics{suffix}.png")
    plt.close()


if __name__ == "__main__":
    main()

