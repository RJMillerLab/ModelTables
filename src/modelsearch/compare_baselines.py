"""
Author: Zhengyuan Dong
Created: 2025-10-02
Last Edited: 2025-10-02
Description: Compare two baselines for a given modelId (model card):
  1) Dense baseline (neighbors from baseline1 search over model cards)
  2) Table-search-derived baseline (use precomputed table search results, map CSVs -> modelIds)

Usage:
  python src/modelsearch/compare_baselines.py \
    --model_id Salesforce/codet5-base \
    --relationship_parquet data/processed/modelcard_step3_dedup.parquet \
    --starmie_json results/table_search.json \
    --dense_neighbors output/modelsearch/modelsearch_neighbors.json \
    --output_md output/compare_Salesforce_codet5-base.md
"""

import os
import json
import argparse
from typing import Dict, List, Set, Tuple, Any

import duckdb
import pandas as pd
import numpy as np


def _flatten_cell(value: Any) -> List[Any]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
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


def _read_relationships(parquet_path: str) -> pd.DataFrame:
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Relationship parquet not found: {parquet_path}")
    # Load minimally needed columns; try a few options
    df = pd.read_parquet(parquet_path)
    cols = df.columns.tolist()
    if "modelId" not in cols:
        raise ValueError("relationship_parquet must contain column 'modelId'")

    # Collect all potential CSV/list columns: prefer known keys, else any containing 'csv' or 'table_list'
    preferred = [
        "hugging_table_list_dedup",
        "github_table_list_dedup",
        "html_table_list_mapped_dedup",
        "llm_table_list_mapped_dedup",
        "hugging_table_list",
        "github_table_list",
        "html_table_list_mapped",
        "llm_table_list_mapped",
        "csvs",
        "csv_paths",
        "csv_path",
    ]
    list_cols = [c for c in preferred if c in cols]
    if not list_cols:
        list_cols = [c for c in cols if ("csv" in c.lower() or "table_list" in c.lower()) and c != "modelId"]
    if not list_cols:
        raise ValueError("No CSV/list-like columns found in relationship parquet.")

    # Build long form: modelId, csv_basename (explode and normalize)
    records: List[Tuple[str, str]] = []
    for _, row in df.iterrows():
        mid = row["modelId"]
        for c in list_cols:
            vals = _flatten_cell(row.get(c))
            for v in vals:
                try:
                    vstr = str(v)
                except Exception:
                    continue
                base = os.path.basename(vstr)
                if base:
                    records.append((mid, base))
    rel = pd.DataFrame(records, columns=["modelId", "csv_basename"]).drop_duplicates()
    return rel


def _read_starmie_results(path: str) -> Dict[str, List[str]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"starmie_json not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    mapping: Dict[str, List[str]] = {}
    if isinstance(data, dict):
        # Direct mapping: query_csv -> [retrieved csvs]
        for k, v in data.items():
            q = os.path.basename(str(k))
            lst = v if isinstance(v, list) else []
            mapping[q] = [os.path.basename(str(x)) for x in lst]
    elif isinstance(data, list):
        # List of records
        for rec in data:
            if not isinstance(rec, dict):
                continue
            q = os.path.basename(str(rec.get("query_csv", "")))
            if not q:
                continue
            retrieved = rec.get("retrieved") or rec.get("retrieved_csvs") or []
            if not isinstance(retrieved, list):
                retrieved = []
            mapping[q] = [os.path.basename(str(x)) for x in retrieved]
    else:
        raise ValueError("Unsupported starmie_json format")
    return mapping


def _read_dense_neighbors(path: str) -> Dict[str, List[str]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"dense_neighbors not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("dense_neighbors JSON must be a dict: modelId -> [neighbors]")
    return {str(k): [str(x) for x in v] for k, v in data.items()}


def _link(model_id: str) -> str:
    return f"https://huggingface.co/{model_id}"


def compare(model_id: str,
            relationship_parquet: str,
            starmie_json: str,
            dense_neighbors: str) -> Tuple[List[str], List[str], List[str]]:
    rel = _read_relationships(relationship_parquet)
    starmie = _read_starmie_results(starmie_json)
    dense = _read_dense_neighbors(dense_neighbors)

    # CSVs of the query model
    q_csvs = rel.loc[rel["modelId"] == model_id, "csv_basename"].dropna().unique().tolist()

    # Retrieved CSVs via table search across all query CSVs
    retrieved_csvs: Set[str] = set()
    for q in q_csvs:
        retrieved_csvs.update(starmie.get(q, []))

    # Map retrieved CSVs -> modelIds using relationship parquet (smart membership over exploded list columns)
    derived_modelids = rel.loc[rel["csv_basename"].isin(list(retrieved_csvs)), "modelId"].dropna().unique().tolist()

    # Dense neighbors for the modelId
    dense_cands = dense.get(model_id, [])

    # Keep top-N reasonable length for display
    dense_top = dense_cands[:50]
    derived_top = derived_modelids[:50]
    inter = sorted(list(set(dense_top) & set(derived_top)))
    return dense_top, derived_top, inter


def write_markdown(model_id: str,
                   dense_top: List[str],
                   derived_top: List[str],
                   inter: List[str],
                   out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# Baseline Comparison for {model_id}\n\n")
        f.write(f"![Pipeline](docs/pipeline.png)\n\n")
        f.write(f"Experiment target: [{model_id}]({_link(model_id)})\n\n")

        f.write("## Baseline (Dense model-card neighbors)\n\n")
        for m in dense_top:
            f.write(f"- [{m}]({_link(m)})\n")
        f.write("\n")

        f.write("## Ours (Table-search-derived via CSV mapping)\n\n")
        for m in derived_top:
            f.write(f"- [{m}]({_link(m)})\n")
        f.write("\n")

        f.write("## Intersection\n\n")
        for m in inter:
            f.write(f"- [{m}]({_link(m)})\n")
        f.write("\n")

        f.write("---\n")
        f.write(f"Dense count: {len(dense_top)} | Table-derived count: {len(derived_top)} | Intersection: {len(inter)}\n")


def main():
    parser = argparse.ArgumentParser(description="Compare dense vs table-search baselines for a modelId")
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--relationship_parquet", required=True,
                        help="Parquet mapping modelId to csvs (expects columns: modelId and csv_path or csvs)")
    # Support alternative flag names used in docs: --table_search_result, --modelsearch_base_result
    parser.add_argument("--starmie_json", required=False,
                        help="JSON with precomputed table search results")
    parser.add_argument("--table_search_result", required=False,
                        help="Alias for --starmie_json")
    parser.add_argument("--dense_neighbors", required=False, default=None,
                        help="JSON mapping modelId to neighbor modelIds (dense baseline)")
    parser.add_argument("--modelsearch_base_result", required=False,
                        help="Alias for --dense_neighbors")
    parser.add_argument("--output_md", required=True,
                        help="Output Markdown path for comparison")

    args = parser.parse_args()

    # Resolve aliases and defaults
    starmie_path = args.starmie_json or args.table_search_result
    if not starmie_path:
        raise ValueError("Please provide --starmie_json or --table_search_result")
    dense_path = args.dense_neighbors or args.modelsearch_base_result or "output/modelsearch/modelsearch_neighbors.json"

    dense_top, derived_top, inter = compare(
        model_id=args.model_id,
        relationship_parquet=args.relationship_parquet,
        starmie_json=starmie_path,
        dense_neighbors=dense_path,
    )
    write_markdown(args.model_id, dense_top, derived_top, inter, args.output_md)
    print(f"✅ Wrote comparison to {args.output_md}")


if __name__ == "__main__":
    main()


