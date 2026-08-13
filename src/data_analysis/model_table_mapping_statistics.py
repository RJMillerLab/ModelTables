#!/usr/bin/env python3
"""Plot model--table mapping and deduplicated-table reuse distributions.

The released corpus uses three table sources: Hugging Face model cards,
GitHub READMEs, and arXiv HTML.  This script applies the valid-table mask,
counts unique tables per model and distinct models per deduplicated table, and
produces a compact two-panel figure plus reusable statistics files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BLUE = "#2F6F9F"
ORANGE = "#D9822B"
INK = "#25313C"
GRID = "#D9DEE3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True, help="Processing tag, e.g. 251117")
    parser.add_argument("--v2_mode", action="store_true", help="Use v2 processed files")
    parser.add_argument("--output-dir", default="data/analysis")
    return parser.parse_args()


def percentile_summary(series: pd.Series) -> dict[str, float | int]:
    return {
        "count": int(series.size),
        "mean": float(series.mean()),
        "median": float(series.median()),
        "p90": float(series.quantile(0.90)),
        "p95": float(series.quantile(0.95)),
        "p99": float(series.quantile(0.99)),
        "max": int(series.max()),
    }


def empirical_ccdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return unique observed values and P(X >= x), without binning."""
    x, counts = np.unique(values, return_counts=True)
    tail_counts = np.cumsum(counts[::-1])[::-1]
    return x, tail_counts / tail_counts[0] * 100.0


def main() -> None:
    args = parse_args()
    suffix = f"_{args.tag}"
    v2_suffix = "_v2" if args.v2_mode else ""
    step3 = Path(f"data/processed/modelcard_step3_dedup{v2_suffix}{suffix}.parquet")
    valid_list = Path(f"data/analysis/all_valid_title_valid{v2_suffix}{suffix}.txt")
    for path in (step3, valid_list):
        if not path.exists():
            raise FileNotFoundError(path)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / f"model_table_mapping_statistics{v2_suffix}{suffix}"

    con = duckdb.connect()
    step3_sql = str(step3.resolve()).replace("'", "''")
    valid_sql = str(valid_list.resolve()).replace("'", "''")
    pairs = con.execute(
        f"""
        WITH valid AS (
            SELECT trim(column0) AS table_path
            FROM read_csv('{valid_sql}', header=false, columns={{'column0': 'VARCHAR'}})
            WHERE trim(column0) <> ''
        ), source_pairs AS (
            SELECT modelId, unnest(coalesce(hugging_table_list_dedup, [])) AS table_path
            FROM read_parquet('{step3_sql}')
            UNION ALL
            SELECT modelId, unnest(coalesce(github_table_list_dedup, [])) AS table_path
            FROM read_parquet('{step3_sql}')
            UNION ALL
            SELECT modelId, unnest(coalesce(html_table_list_mapped_dedup, [])) AS table_path
            FROM read_parquet('{step3_sql}')
        )
        SELECT DISTINCT source_pairs.modelId, source_pairs.table_path
        FROM source_pairs
        INNER JOIN valid USING (table_path)
        WHERE source_pairs.modelId IS NOT NULL AND source_pairs.table_path IS NOT NULL
        """
    ).fetchdf()
    con.close()
    if pairs.empty:
        raise ValueError("No model--table pairs remain after applying the valid-table mask")

    tables_per_model = (
        pairs.groupby("modelId", as_index=False)["table_path"]
        .nunique()
        .rename(columns={"table_path": "n_tables"})
        .sort_values(["n_tables", "modelId"], ascending=[False, True])
    )
    models_per_table = (
        pairs.groupby("table_path", as_index=False)["modelId"]
        .nunique()
        .rename(columns={"modelId": "n_models"})
        .sort_values(["n_models", "table_path"], ascending=[False, True])
    )
    models_per_table.insert(0, "rank", np.arange(1, len(models_per_table) + 1))

    tables_per_model.to_parquet(prefix.with_name(prefix.name + "_tables_per_model.parquet"), index=False)
    models_per_table.to_parquet(prefix.with_name(prefix.name + "_models_per_table.parquet"), index=False)

    summary = {
        "processing_tag": args.tag,
        "table_sources": ["ModelCard", "GitHub", "arXiv"],
        "valid_model_table_pairs": int(len(pairs)),
        "models_with_valid_tables": int(tables_per_model.shape[0]),
        "unique_valid_tables": int(models_per_table.shape[0]),
        "tables_per_model": percentile_summary(tables_per_model["n_tables"]),
        "models_per_table": percentile_summary(models_per_table["n_models"]),
        "tables_reused_by_multiple_models": int((models_per_table["n_models"] > 1).sum()),
        "share_tables_reused_by_multiple_models": float((models_per_table["n_models"] > 1).mean()),
    }
    prefix.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.labelcolor": INK, "text.color": INK,
    })
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.15, 2.75))

    model_values = tables_per_model["n_tables"].to_numpy()
    model_x, model_ccdf = empirical_ccdf(model_values)
    left.plot(model_x, model_ccdf, color=BLUE, linewidth=2.0)
    left.set_xscale("log")
    left.set_yscale("log")
    left.set_xlabel("Unique tables per model")
    left.set_ylabel("Models (%)")
    left.set_title("(a) Tables associated with each model", loc="left", fontsize=9.5)

    reuse_values = models_per_table["n_models"].to_numpy()
    reuse_x, reuse_ccdf = empirical_ccdf(reuse_values)
    right.plot(reuse_x, reuse_ccdf, color=ORANGE, linewidth=2.0)
    right.set_xscale("log")
    right.set_yscale("log")
    right.set_xlabel("Distinct models per deduplicated table")
    right.set_ylabel("Tables (%)")
    right.set_title("(b) Table reuse across models", loc="left", fontsize=9.5)

    for axis in (left, right):
        axis.grid(True, axis="y", color=GRID, linewidth=0.6)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.20, top=0.86, wspace=0.36)
    fig.savefig(prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(json.dumps(summary, indent=2))
    print(f"Saved {prefix}.json/.pdf/.png and distribution parquets")


if __name__ == "__main__":
    main()
