"""
Benchmark pandas vs DuckDB SQL for the table‑source loading logic used in step3_gt.

Usage:
PYTHONPATH=. python -m src.data_gt.bench_step3_table_source_sql --tag 251117 --v2_mode
"""

import argparse
import os
import time
from typing import Dict

import duckdb
import pandas as pd

from src.utils import is_list_like, to_list_safe


def build_files(tag: str | None, v2_mode: bool) -> Dict[str, str]:
    suffix = f"_{tag}" if tag else ""
    v2_suffix = "_v2" if v2_mode else ""
    return {
        "step3_dedup": f"data/processed/modelcard_step3_dedup{v2_suffix}{suffix}.parquet",
        "valid_title": f"data/processed/all_title_list_valid{v2_suffix}{suffix}.parquet",
    }


def load_table_source_pandas(files: Dict[str, str]) -> pd.DataFrame:
    """Baseline: current pandas implementation from step3_gt.py."""
    valid_title_path = files["valid_title"]
    if not os.path.exists(valid_title_path):
        raise FileNotFoundError(f"valid_title parquet not found at {valid_title_path}")

    df_valid_title = pd.read_parquet(
        valid_title_path, columns=["modelId", "all_title_list_valid"]
    )

    df = pd.read_parquet(
        files["step3_dedup"],
        columns=[
            "modelId",
            "hugging_table_list_dedup",
            "github_table_list_dedup",
            "html_table_list_mapped_dedup",
            "llm_table_list_mapped_dedup",
        ],
    )

    df["all_table_list_dedup"] = df[
        [
            "hugging_table_list_dedup",
            "github_table_list_dedup",
            "html_table_list_mapped_dedup",
            "llm_table_list_mapped_dedup",
        ]
    ].apply(
        lambda row: [
            x
            for arr in row.tolist()
            if is_list_like(arr)
            for x in to_list_safe(arr)
        ],
        axis=1,
    )

    df_tables = pd.merge(
        df[["modelId", "all_table_list_dedup"]],
        df_valid_title,
        how="left",
        on="modelId",
    )

    mask = (
        df_tables["all_title_list_valid"].apply(
            lambda x: is_list_like(x) and len(to_list_safe(x)) > 0
        )
        & df_tables["all_table_list_dedup"].apply(
            lambda x: is_list_like(x) and len(to_list_safe(x)) > 0
        )
    )
    df_tables = df_tables.loc[
        mask, ["modelId", "all_table_list_dedup", "all_title_list_valid"]
    ]
    return df_tables.set_index("modelId", drop=False)


def load_table_source_duckdb(files: Dict[str, str]) -> pd.DataFrame:
    """
    DuckDB SQL version.

    Notes:
    - Assumes *_dedup columns are list-like (no nested lists), which is true for the
      output of the preprocessing pipeline. We use list_concat inside DuckDB instead
      of the Python-level flatten used in the pandas version.
    """
    valid_title_path = os.path.abspath(files["valid_title"])
    step3_path = os.path.abspath(files["step3_dedup"])

    if not os.path.exists(valid_title_path):
        raise FileNotFoundError(f"valid_title parquet not found at {valid_title_path}")
    if not os.path.exists(step3_path):
        raise FileNotFoundError(f"step3_dedup parquet not found at {step3_path}")

    con = duckdb.connect(":memory:")
    try:
        step3_path_sql = step3_path.replace("\\", "/")
        valid_title_path_sql = valid_title_path.replace("\\", "/")

        con.execute(
            """
            CREATE VIEW step3 AS
            SELECT
                modelId,
                list_concat(
                    coalesce(hugging_table_list_dedup, []),
                    coalesce(github_table_list_dedup, []),
                    coalesce(html_table_list_mapped_dedup, []),
                    coalesce(llm_table_list_mapped_dedup, [])
                ) AS all_table_list_dedup
            FROM read_parquet(?)
            """.replace(
                "read_parquet(?)", f"read_parquet('{step3_path_sql}')"
            )
        )

        con.execute(
            """
            CREATE VIEW titles AS
            SELECT
                modelId,
                all_title_list_valid
            FROM read_parquet(?)
            """.replace(
                "read_parquet(?)", f"read_parquet('{valid_title_path_sql}')"
            )
        )

        # Filter out rows with empty/NULL lists on either side.
        # DuckDB: list_length(list) returns length of a LIST column.
        df_tables = con.execute(
            """
            SELECT
                s.modelId,
                s.all_table_list_dedup,
                t.all_title_list_valid
            FROM step3 AS s
            JOIN titles AS t USING (modelId)
            WHERE
                all_title_list_valid IS NOT NULL
                AND array_length(all_title_list_valid, 1) > 0
                AND all_table_list_dedup IS NOT NULL
                AND array_length(all_table_list_dedup, 1) > 0
            """
        ).fetchdf()
    finally:
        con.close()

    return df_tables.set_index("modelId", drop=False)


def load_table_source_duckdb_rows(files: Dict[str, str]) -> list:
    """DuckDB load and return rows directly (no DataFrame). Same as step3_gt.load_table_source()."""
    df = load_table_source_duckdb(files)
    return [
        (list(p), list(c))
        for p, c in zip(df["all_title_list_valid"], df["all_table_list_dedup"])
    ]


def load_table_source_duckdb_rows_and_paper_rid_pairs(
    files: Dict[str, str],
) -> tuple:
    """DuckDB load returning (rows, paper_rid_pairs). Same SQL as step3_gt (base view + UNNEST)."""
    valid_title_path = os.path.abspath(files["valid_title"])
    step3_path = os.path.abspath(files["step3_dedup"])
    if not os.path.exists(valid_title_path):
        raise FileNotFoundError(f"valid_title parquet not found at {valid_title_path}")
    if not os.path.exists(step3_path):
        raise FileNotFoundError(f"step3_dedup parquet not found at {step3_path}")

    step3_path_sql = step3_path.replace("\\", "/")
    valid_title_path_sql = valid_title_path.replace("\\", "/")

    con = duckdb.connect(":memory:")
    try:
        con.execute(
            """
            CREATE VIEW step3 AS
            SELECT
                modelId,
                list_concat(
                    coalesce(hugging_table_list_dedup, []),
                    coalesce(github_table_list_dedup, []),
                    coalesce(html_table_list_mapped_dedup, []),
                    coalesce(llm_table_list_mapped_dedup, [])
                ) AS all_table_list_dedup
            FROM read_parquet(?)
            """.replace("read_parquet(?)", f"read_parquet('{step3_path_sql}')")
        )
        con.execute(
            """
            CREATE VIEW titles AS
            SELECT modelId, all_title_list_valid
            FROM read_parquet(?)
            """.replace("read_parquet(?)", f"read_parquet('{valid_title_path_sql}')")
        )
        con.execute(
            """
            CREATE VIEW base AS
            SELECT
                row_number() OVER (ORDER BY s.modelId) - 1 AS rid,
                s.all_table_list_dedup,
                t.all_title_list_valid
            FROM step3 AS s
            JOIN titles AS t USING (modelId)
            WHERE all_title_list_valid IS NOT NULL
              AND array_length(all_title_list_valid, 1) > 0
              AND all_table_list_dedup IS NOT NULL
              AND array_length(all_table_list_dedup, 1) > 0
            """
        )
        df = con.execute(
            "SELECT rid, all_table_list_dedup, all_title_list_valid FROM base ORDER BY rid"
        ).fetchdf()
        paper_rid_df = con.execute(
            """
            SELECT base.rid, u.title
            FROM base, UNNEST(base.all_title_list_valid) AS u(title)
            """
        ).fetchdf()
    finally:
        con.close()

    rows = [
        (list(p), list(c))
        for p, c in zip(df["all_title_list_valid"], df["all_table_list_dedup"])
    ]
    paper_rid_pairs = list(
        zip(paper_rid_df["rid"].tolist(), paper_rid_df["title"].tolist())
    )
    return rows, paper_rid_pairs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark pandas vs DuckDB SQL for step3_gt table-source loading."
    )
    parser.add_argument(
        "--tag",
        dest="tag",
        default=None,
        help="Tag suffix for versioning (e.g., 251117).",
    )
    parser.add_argument(
        "--v2_mode",
        dest="v2_mode",
        action="store_true",
        help="Use v2 mode for input parquet naming.",
    )
    args = parser.parse_args()

    files = build_files(args.tag, args.v2_mode)
    print("📁 Files in use:")
    for k, v in files.items():
        print(f"  {k}: {v}")

    # Warm-up I/O cache a bit with a very small read (DuckDB)
    print("\n🔄 Warming up DuckDB (small schema probe)...")
    _ = duckdb.connect(":memory:").execute(
        "DESCRIBE SELECT * FROM read_parquet(?) LIMIT 1", [os.path.abspath(files["step3_dedup"])]
    ).fetchall()

    print("\n⏱  Running pandas version...")
    t0 = time.perf_counter()
    df_pandas = load_table_source_pandas(files)
    t1 = time.perf_counter()
    print(f"  pandas: {len(df_pandas)} rows, elapsed {t1 - t0:.3f} s")

    print("\n⏱  Running DuckDB SQL version...")
    t2 = time.perf_counter()
    df_sql = load_table_source_duckdb(files)
    t3 = time.perf_counter()
    print(f"  duckdb: {len(df_sql)} rows, elapsed {t3 - t2:.3f} s")

    print("\n📊 Comparison:")
    print(f"  pandas rows: {len(df_pandas)}")
    print(f"  duckdb rows: {len(df_sql)}")
    print(f"  time pandas: {t1 - t0:.3f} s")
    print(f"  time duckdb: {t3 - t2:.3f} s")

    # Optional: quick sanity check on modelId overlap
    inter = set(df_pandas.index) & set(df_sql.index)
    print(f"  modelId intersection size: {len(inter)}")
    print(f"  modelId only in pandas: {len(set(df_pandas.index) - inter)}")
    print(f"  modelId only in duckdb: {len(set(df_sql.index) - inter)}")


if __name__ == "__main__":
    main()

