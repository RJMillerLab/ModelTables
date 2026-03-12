#!/usr/bin/env python3
"""
Lookup titles (and modelIds) for given table CSV files.

Given one or more table basenames like:
  fcb942b12f93a1ddd81cef18cb62e53a_table_0.csv

this script:
  - Finds all modelIds whose *_table_list_dedup/_sym columns contain that table
  - Prints how many models reference it
  - Prints example modelIds and their titles
  - Prints the resolved CSV path on disk (via report_generation.get_file_path)

It reuses the same logic as report_generation.build_table_model_title_maps.
"""

import argparse
from typing import List

from src.data_analysis.report_generation import get_file_path, build_table_model_title_maps

def _load_table_list(args: argparse.Namespace) -> List[str]:
    tables: List[str] = []
    if args.table:
        tables.append(args.table)
    if args.tables_file:
        with open(args.tables_file, "r") as f:
            for line in f:
                name = line.strip()
                if name:
                    tables.append(name)
    # dedup while preserving order
    seen = set()
    unique_tables = []
    for t in tables:
        if t not in seen:
            seen.add(t)
            unique_tables.append(t)
    return unique_tables


def main() -> None:
    p = argparse.ArgumentParser(description="Lookup titles and modelIds for given table CSV basenames.")
    p.add_argument("--tag", type=str, default=None, help="Tag suffix for versioning (e.g., 251117).")
    p.add_argument("--v2_mode", action="store_true", help="Use v2 mode.")
    p.add_argument("--table", type=str, default=None, help="Table CSV basename.")
    p.add_argument("--tables-file", dest="tables_file", type=str, default=None, help="File with one table basename per line.")
    p.add_argument("--max_models", type=int, default=10, help="Max number of example modelIds/titles to print per table.")
    args = p.parse_args()
    tables = _load_table_list(args)
    if not tables:
        raise SystemExit("No tables provided. Use --table or --tables-file.")

    v2_suffix = "_v2" if args.v2_mode else ""
    suffix = f"_{args.tag}" if args.tag else ""

    print("Building table→models and model→titles maps ...")
    table_to_models, model_to_titles = build_table_model_title_maps(v2_suffix, suffix)

    for tbl in tables:
        print("\n" + "=" * 80)
        print(f"TABLE: {tbl}")

        # Try resolving CSV path
        try:
            csv_path = get_file_path(tbl, v2_suffix, suffix)
            print(f"CSV path: {csv_path}")
        except Exception as e:
            print(f"CSV path: <unknown> (get_file_path failed: {e})")

        models = sorted(table_to_models.get(tbl, []))
        n_models = len(models)
        print(f"Referenced by {n_models} model(s).")

        if n_models == 0:
            continue

        max_show = min(args.max_models, n_models)
        print(f"Showing first {max_show} model(s):")
        for mid in models[:max_show]:
            title_info = model_to_titles.get(mid, {})
            # Prefer valid titles if available, fallback to raw
            valid_titles = title_info.get("valid") or []
            raw_titles = title_info.get("raw") or []
            main_title = valid_titles[0] if valid_titles else (raw_titles[0] if raw_titles else "<no title>")
            print(f"  - modelId: {mid}")
            print(f"    title : {main_title}")


if __name__ == "__main__":
    main()

