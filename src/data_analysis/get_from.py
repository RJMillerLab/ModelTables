#!/usr/bin/env python
"""
Author: Zhengyuan Dong
Date: 2025-09-28
Last Edited: 2025-09-29

Description: Unified CLI to get <target> from <source> using hardcoded parquet mappings.

Usage:
    python -m src.data_analysis.get_from --target html_table_list_mapped_dedup --source modelId --value google-bert/bert-base-uncased
    python -m src.data_analysis.get_from --target readme_path --source csv_paths --value "64dc62e53f_table2.csv" 
    python -m src.data_analysis.get_from --target modelId --source hugging_table_list --value data/processed/deduped_hugging_csvs/021f09961f_table1.csv
"""
import argparse
import os
from typing import List, Dict, Any, Tuple, Optional

import pandas as pd
import numpy as np
import duckdb


# No hardcoded parquet mappings or routes; everything is inferred from logs/parquet_schema.log


def list_routes() -> List[str]:
    return [
        "Use --target <attr> --source <attr> --value <val>",
    ]


# Removed specialized resolvers; only generic attribute-based querying is supported


def load_parquet_schema_sorted(log_path: str = "logs/parquet_schema.log") -> List[Tuple[str, List[str]]]:
    """Load schema log and return a list of (parquet_path, sorted_columns), sorted by path."""
    mapping = parse_parquet_schema_log(log_path)
    items: List[Tuple[str, List[str]]] = []
    for path, cols in mapping.items():
        items.append((path, sorted(cols)))
    items.sort(key=lambda x: x[0])
    return items


def _collect_column_frequencies(mapping: Dict[str, List[str]]) -> Dict[str, int]:
    freq: Dict[str, int] = {}
    for _, cols in mapping.items():
        for c in cols:
            freq[c] = freq.get(c, 0) + 1
    return freq


def _prioritize_candidates(token: str, candidates: List[str]) -> List[str]:
    token_l = token.lower()
    # Simple preference rules
    preferences: List[str] = []
    if token_l in {"csvs", "csv", "table", "tables"}:
        preferences = [
            "hugging_table_list_sym", "github_table_list_sym", "html_table_list_sym", "llm_table_list_sym",
            "hugging_table_list", "github_table_list", "html_table_list_mapped", "llm_table_list_mapped",
            "hugging_table_list_dedup", "github_table_list_dedup", "html_table_list_mapped_dedup", "llm_table_list_mapped_dedup",
            "csv_path",
        ]
    elif token_l in {"readme", "readme_path", "readmepath"}:
        preferences = ["readme_path"]
    elif token_l in {"model", "modelid", "model_id"}:
        preferences = ["modelId"]
    # Rank candidates by preference order first
    pref_rank = {name: i for i, name in enumerate(preferences)}
    def sort_key(name: str) -> Tuple[int, int, str]:
        return (pref_rank.get(name, 10_000), len(name), name)
    return sorted(candidates, key=sort_key)


def guess_attribute_name(token: str, mapping: Dict[str, List[str]], debug: bool = False) -> Tuple[Optional[str], List[str]]:
    all_cols = set()
    for _, cols in mapping.items():
        for c in cols:
            all_cols.add(c)
    # exact match
    if token in all_cols:
        if debug:
            print(f"resolved attribute '{token}' by exact match")
        return token, [token]
    # case-insensitive exact
    token_l = token.lower()
    ci_matches = [c for c in all_cols if c.lower() == token_l]
    if len(ci_matches) == 1:
        if debug:
            print(f"resolved attribute '{token}' -> '{ci_matches[0]}' by case-insensitive match")
        return ci_matches[0], ci_matches
    # substring match
    sub_matches = [c for c in all_cols if token_l in c.lower()]
    if debug:
        print(f"substring matches for '{token}': {sub_matches[:10]}")
    if len(sub_matches) == 1:
        return sub_matches[0], sub_matches
    if sub_matches:
        ordered = _prioritize_candidates(token, sub_matches)
        if debug:
            print(f"picked '{ordered[0]}' from candidates")
        return ordered[0], ordered
    # nothing found
    return None, []


def parse_parquet_schema_log(log_path: str = "logs/parquet_schema.log", debug: bool = False) -> Dict[str, List[str]]:
    """Parse parquet schema log and return mapping parquet_path -> columns list."""
    if not os.path.exists(log_path):
        return {}
    mapping: Dict[str, List[str]] = {}
    current_file: Optional[str] = None
    try:
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.rstrip('\n')
                if line.startswith('FILE: '):
                    parts = line.split('FILE: ', 1)[-1].split(' | ', 1)
                    if parts:
                        current_file = parts[0].strip()
                        mapping[current_file] = []
                elif current_file and ':' in line:
                    col = line.strip().split(':', 1)[0].strip()
                    if col and col not in mapping[current_file]:
                        mapping[current_file].append(col)
    except Exception:
        return {}
    if debug:
        print(f"parsed schema: {len(mapping)} parquet files")
        preview = list(mapping.items())[:8]
        for p, cols in preview:
            print(f"  {p} -> {len(cols)} cols (sample: {cols[:6]})")
    return mapping


def find_parquets_with_attrs(attr_a: str, attr_b: str, mapping: Dict[str, List[str]], debug: bool = False) -> List[str]:
    both: List[str] = []
    for path, cols in mapping.items():
        if attr_a in cols and attr_b in cols and os.path.exists(path):
            both.append(path)
    if debug:
        print(f"parquets containing both '{attr_a}' and '{attr_b}': {len(both)}")
        for p in both[:10]:
            print(f"  candidate: {p}")
    return both


def find_suggested_parquets(attr_a: str, attr_b: str, mapping: Dict[str, List[str]], debug: bool = False) -> Tuple[List[str], List[str]]:
    has_a: List[str] = []
    has_b: List[str] = []
    for path, cols in mapping.items():
        if os.path.exists(path):
            if attr_a in cols:
                has_a.append(path)
            if attr_b in cols:
                has_b.append(path)
    if debug:
        print(f"parquets with source '{attr_a}': {len(has_a)} | with target '{attr_b}': {len(has_b)}")
    return has_a, has_b


def _flatten_cell(value: Any) -> List[Any]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, list):
        out: List[Any] = []
        for v in value:
            if isinstance(v, (list, np.ndarray)):
                out.extend(_flatten_cell(v))
            else:
                out.append(v)
        return out
    if isinstance(value, np.ndarray):
        return _flatten_cell(value.tolist())
    return [value]


def generic_get_attr_from_attr(target_attr: str, source_attr: str, value: Any, log_path: str = "logs/parquet_schema.log", debug: bool = False) -> List[Any]:
    mapping = parse_parquet_schema_log(log_path, debug=debug)
    if not mapping:
        print(f"Parquet schema log not found or empty: {log_path}")
        return []
    candidates = find_parquets_with_attrs(source_attr, target_attr, mapping, debug=debug)
    if not candidates:
        print("No parquet found containing both attributes.")
        has_s, has_t = find_suggested_parquets(source_attr, target_attr, mapping, debug=debug)
        if has_s and has_t:
            print("Suggestion: combine these parquets manually:")
            print(f"  source_attr in: {has_s[:5]}")
            print(f"  target_attr in: {has_t[:5]}")
        return []

    results: List[Any] = []
    for pq in candidates:
        print(f"we load from parquet {pq}")
        # Prefer DuckDB SQL path for speed and list-aware filtering
        try:
            con = duckdb.connect(database=':memory:')
            # Build predicate depending on value type (string vs other)
            if isinstance(value, str):
                predicate = (
                    f"CASE WHEN typeof({source_attr})='LIST' "
                    f"THEN list_contains({source_attr}, ?) OR "
                    f"     list_any(list_transform({source_attr}, x -> lower(cast(x AS VARCHAR)) LIKE '%' || lower(?) )) "
                    f"ELSE lower(cast({source_attr} AS VARCHAR)) = lower(?) OR "
                    f"     lower(cast({source_attr} AS VARCHAR)) LIKE '%' || lower(?) END"
                )
                params = [value, value, value, value]
            else:
                predicate = (
                    f"CASE WHEN typeof({source_attr})='LIST' "
                    f"THEN list_contains({source_attr}, ?) "
                    f"ELSE {source_attr} = ? END"
                )
                params = [value, value]

            query = (
                f"SELECT {source_attr} AS s, {target_attr} AS t "
                f"FROM read_parquet('{pq}') WHERE {predicate}"
            )
            if debug:
                print(f"  SQL: {query}")
            res = con.execute(query, params).fetchall()
            if not res:
                if debug:
                    print("  matched rows: 0 (duckdb)")
                    try:
                        sample = con.execute(
                            f"SELECT {source_attr} AS s, {target_attr} AS t FROM read_parquet('{pq}') LIMIT 3"
                        ).fetchall()
                        print("  sample rows:")
                        for s, t in sample:
                            print(f"    source={s} | target={t}")
                    except Exception:
                        pass
                con.close()
                # Try next parquet
                continue
            if debug:
                print(f"  matched rows: {len(res)} (duckdb)")
            for _, t in res:
                results.extend(_flatten_cell(t))
            con.close()
            continue
        except Exception as e:
            if debug:
                print(f"  duckdb failed, falling back to pandas. Reason: {e}")
            # Fall back to pandas approach
            try:
                df = pd.read_parquet(pq, columns=[source_attr, target_attr])
            except Exception:
                continue
            if source_attr not in df.columns or target_attr not in df.columns:
                continue
            if debug:
                dtypes = df[[source_attr, target_attr]].dtypes.to_dict()
                print(f"  columns loaded: {list(dtypes.keys())} dtypes: {dtypes}")
            def _cell_contains_value(cell: Any, needle: Any) -> bool:
                if cell is None:
                    return False
                if isinstance(cell, (list, np.ndarray)):
                    try:
                        iterable = cell.tolist() if isinstance(cell, np.ndarray) else cell
                    except Exception:
                        iterable = list(cell)
                    for elem in iterable:
                        if isinstance(needle, str):
                            if isinstance(elem, str):
                                el = elem.lower()
                                nl = needle.lower()
                                if (el == nl) or el.endswith("/" + nl) or el.endswith(nl):
                                    return True
                        else:
                            if elem == needle:
                                return True
                    return False
                if isinstance(needle, str):
                    try:
                        s = str(cell)
                        sl = s.lower()
                        nl = needle.lower()
                        return bool((sl == nl) or sl.endswith("/" + nl) or sl.endswith(nl))
                    except Exception:
                        return False
                try:
                    return bool(cell == needle)
                except Exception:
                    return False
            mask = df[source_attr].apply(lambda cell: _cell_contains_value(cell, value))
            matched = df[mask]
            if matched.empty:
                if debug:
                    print("  matched rows: 0 (pandas)")
                    try:
                        sample = df[[source_attr, target_attr]].head(3)
                        print("  sample rows:")
                        for _, row in sample.iterrows():
                            print(f"    source={row[source_attr]} | target={row[target_attr]}")
                    except Exception:
                        pass
                continue
            if debug:
                print(f"  matched rows: {len(matched)} (pandas)")
            for cell in matched[target_attr].tolist():
                results.extend(_flatten_cell(cell))
    if debug:
        print(f"total results after flatten: {len(results)}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Get <target> from <source> using parquet mappings")
    parser.add_argument("--target", type=str, help="Target attribute name (e.g., hugging_table_list, readme_path)")
    parser.add_argument("--source", type=str, help="Source attribute name (e.g., modelId, csv_path, csv_paths)")
    parser.add_argument("--value", type=str, help="The value to match on for the source attribute")
    parser.add_argument("--list", action="store_true", help="List supported routes and exit")
    parser.add_argument("--schema-log", type=str, default="logs/parquet_schema.log", help="Path to parquet schema log")
    parser.add_argument("--debug", action="store_true", help="Print step-by-step parsing and matching info")
    parser.add_argument("--verbose", action="store_true", help="Verbously print parsed parquet overview")
    parser.add_argument("--full", action="store_true", help="Do not truncate long string results when printing")
    parser.add_argument("--max-results", type=int, default=200, help="Maximum number of results to print")
    args = parser.parse_args()

    if args.list:
        for line in list_routes():
            print(line)
        return

    # Print-only parse overview
    mapping = parse_parquet_schema_log(args.schema_log, debug=args.debug)
    if not mapping:
        print(f"Parquet schema log not found or empty: {args.schema_log}")
        return
    if args.verbose:
        items = sorted(mapping.items(), key=lambda x: x[0])
        for path, cols in items:
            preview = cols if isinstance(cols, list) else []
            print(f"PARQUET: {path}")
            print(f"  num_cols={len(cols)} preview={preview}")

    # Dynamic resolution from --target/--source
    mapping = parse_parquet_schema_log(args.schema_log, debug=args.debug)
    if not mapping:
        print(f"Parquet schema log not found or empty: {args.schema_log}")
        return
    resolved_target, target_cands = guess_attribute_name(args.target or "", mapping, debug=args.debug)
    resolved_source, source_cands = guess_attribute_name(args.source or "", mapping, debug=args.debug)
    if args.debug:
        print(f"resolved target attr: {resolved_target} (candidates: {target_cands[:6]})")
        print(f"resolved source attr: {resolved_source} (candidates: {source_cands[:6]})")
    if not resolved_target or not resolved_source:
        print("Unable to resolve attributes from target/source. Provide clearer names.")
        return
    # Print exact candidate parquets that contain both attributes
    cands = find_parquets_with_attrs(resolved_source, resolved_target, mapping, debug=False)
    if cands:
        print("parquets_with_both:")
        for p in cands:
            print(f"  {p}")
    else:
        print("No parquet found containing both attributes.")
        has_s, has_t = find_suggested_parquets(resolved_source, resolved_target, mapping, debug=False)
        if has_s:
            print("  source_attr in:")
            for p in has_s[:10]:
                print(f"    {p}")
        if has_t:
            print("  target_attr in:")
            for p in has_t[:10]:
                print(f"    {p}")
    results_any = generic_get_attr_from_attr(
        resolved_target,
        resolved_source,
        args.value,
        log_path=args.schema_log,
        debug=args.debug,
    )
    if not results_any:
        print("No results found.")
        return
    # Deduplicate while preserving order
    seen = set()
    unique_results: List[Any] = []
    for r in results_any:
        key = r if isinstance(r, (int, float, str)) else str(r)
        if key in seen:
            continue
        seen.add(key)
        unique_results.append(r)
    print(f"results: {len(unique_results)} (showing up to {args.max_results})")
    printed = 0
    for r in unique_results:
        if printed >= max(0, args.max_results):
            break
        if isinstance(r, str) and not args.full and len(r) > 300:
            print(r[:300] + "... [truncated]")
        else:
            print(r)
        printed += 1


if __name__ == "__main__":
    main()


