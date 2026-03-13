#!/usr/bin/env python3
"""
Compute per-table usage frequency from modelcard_step3_dedup parquet.

Definition:
- "occurrences": total number of times a table appears across all models and all sources
  (i.e., after UNNESTing hugging/github/html/llm table lists and counting rows).
- "n_models": number of distinct modelIds that reference the table at least once.

Outputs:
- Prints Top-K tables by occurrences (and by n_models optionally)
- Writes a parquet (and optionally csv) summary for further debugging.
"""

import argparse
import os
from typing import Optional

import duckdb
import pandas as pd
from src.utils import to_parquet

GENERIC_TABLE_PATTERNS = [
    "1910.09700_table",
    "204823751_table",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute per-table usage counts from step3_dedup parquet")
    parser.add_argument("--tag", default=None, help="Optional tag suffix (e.g., 251117).")
    parser.add_argument("--v2_mode", action="store_true", help="Use v2 mode filenames (adds _v2).")
    parser.add_argument("--top_k", type=int, default=100, help="Top-K tables to print.")
    parser.add_argument("--top_models", type=int, default=3, help="Max number of example modelIds to attach per table (for inspection).")
    parser.add_argument("--no_mask_generic", action="store_true", help="Do NOT filter out generic tables (default: filter them).")
    args = parser.parse_args()

    suffix = f"_{args.tag}" if args.tag else ""
    v2_suffix = "_v2" if args.v2_mode else ""
    step3_path = os.path.join("data", "processed", f"modelcard_step3_dedup{v2_suffix}{suffix}.parquet")

    mask_generic = not args.no_mask_generic

    con = duckdb.connect()

    where_generic = ""
    # filter out general tables
    if mask_generic:
        filters = " AND ".join([f"table_path NOT LIKE '%{p}%'" for p in GENERIC_TABLE_PATTERNS])
        where_generic = f"AND {filters}"

    # Unnest each list and union them. Count occurrences and distinct models per basename.
    query = f"""
    WITH base AS (
        SELECT
            modelId,
            hugging_table_list_dedup,
            github_table_list_dedup,
            html_table_list_mapped_dedup,
            llm_table_list_mapped_dedup
        FROM read_parquet('{step3_path}')
        WHERE modelId IS NOT NULL
    ),
    unnested AS (
        SELECT modelId, UNNEST(COALESCE(hugging_table_list_dedup, [])) AS table_path
        FROM base
        WHERE hugging_table_list_dedup IS NOT NULL AND array_length(hugging_table_list_dedup) > 0

        UNION ALL

        SELECT modelId, UNNEST(COALESCE(github_table_list_dedup, [])) AS table_path
        FROM base
        WHERE github_table_list_dedup IS NOT NULL AND array_length(github_table_list_dedup) > 0

        UNION ALL

        SELECT modelId, UNNEST(COALESCE(html_table_list_mapped_dedup, [])) AS table_path
        FROM base
        WHERE html_table_list_mapped_dedup IS NOT NULL AND array_length(html_table_list_mapped_dedup) > 0

        UNION ALL

        SELECT modelId, UNNEST(COALESCE(llm_table_list_mapped_dedup, [])) AS table_path
        FROM base
        WHERE llm_table_list_mapped_dedup IS NOT NULL AND array_length(llm_table_list_mapped_dedup) > 0
    ),
    cleaned AS (
        SELECT
            modelId,
            table_path,
            regexp_extract(table_path, '([^/\\\\]+)$', 1) AS table_basename
        FROM unnested
        WHERE table_path IS NOT NULL
          AND table_path <> ''
          {where_generic}
    )
    SELECT
        table_basename,
        COUNT(*) AS occurrences,
        COUNT(DISTINCT modelId) AS n_models
    FROM cleaned
    GROUP BY table_basename
    ORDER BY occurrences DESC, n_models DESC, table_basename ASC
    """

    df = con.execute(query).fetchdf()
    con.close()

    df["occurrences"] = df["occurrences"].astype("int64")
    df["n_models"] = df["n_models"].astype("int64")

    # Optional mask: keep only table_basenames listed in all_valid_title_valid*.txt
    if mask_generic:
        mask_file = f"data/analysis/all_valid_title_valid{v2_suffix}{suffix}.txt"
        if os.path.exists(mask_file):
            print(f"Applying mask from: {mask_file}")
            mask_basenames = set()
            with open(mask_file, "r") as f:
                for line in f:
                    p = line.strip()
                    if not p:
                        continue
                    mask_basenames.add(os.path.basename(p))
            before = len(df)
            df = df[df["table_basename"].isin(mask_basenames)].reset_index(drop=True)
            after = len(df)
            filtered_out = before - after
            missing_in_usage = len(mask_basenames - set(df["table_basename"]))
            print(f"Masked tables: kept {after:,} / {before:,} basenames present in usage table (filtered out {filtered_out:,}).")
            print(f"Mask coverage: {after:,} / {len(mask_basenames):,} basenames from mask appear in step3_dedup (missing {missing_in_usage:,}).")
        else:
            print(f"No mask file found at {mask_file}; using all tables in step3_dedup.")

    # Enrich with example modelIds using shared mapping helper
    print("\nBuilding model/title maps for enrichment ...")
    from src.data_analysis.report_generation import build_table_model_title_maps
    table_to_models, model_to_titles = build_table_model_title_maps(v2_suffix, suffix)

    def _get_models_for_basename(basename: str):
        models = sorted(table_to_models.get(basename, []))
        if not models:
            return []
        return models[: args.top_models]

    def format_example_models(basename: str) -> str:
        models = _get_models_for_basename(basename)
        return "; ".join(models)

    def format_example_titles(basename: str) -> str:
        models = _get_models_for_basename(basename)
        titles = []
        for mid in models:
            info = model_to_titles.get(mid, {})
            valid = info.get("valid") or []
            raw = info.get("raw") or []
            title = valid[0] if valid else (raw[0] if raw else "")
            titles.append(title)
        return " | ".join(titles)

    df["example_models"] = df["table_basename"].apply(format_example_models)
    df["example_titles"] = df["table_basename"].apply(format_example_titles)

    print(f"Input: {step3_path}")
    print(f"Unique tables (after mask): {len(df):,}")
    print(f"Total occurrences (after mask): {int(df['occurrences'].sum()):,}")

    top_k = min(args.top_k, len(df))
    print("\nTop tables by occurrences:")
    print(df.head(top_k).to_string(index=False))

    # Also print Top-K by n_models (helpful when duplicates inflate occurrences).
    print("\nTop tables by distinct modelId count:")
    # print all
    #print(df.sort_values(["n_models", "occurrences", "table_basename"], ascending=[False, False, True]).head(top_k).to_string(index=False))
    # print only table_basename and occurrences
    print(df[["table_basename", "occurrences"]].sort_values(["occurrences"], ascending=[False]).to_string(index=False))

    out_parquet = os.path.join("data", "analysis", f"table_usage_stats{v2_suffix}{suffix}.parquet")
    to_parquet(df, out_parquet)
    print(f"\nSaved full stats to {out_parquet}")

if __name__ == "__main__":
    main()

