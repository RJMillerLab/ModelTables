#!/usr/bin/env python3
"""Generate model-card and table coverage statistics from a local snapshot.

The main corpus funnel deliberately stops at models linked to at least one
table from the three released sources (ModelCard, GitHub, and arXiv).  Paper
resolution and BibTeX filtering belong to downstream evaluation and are not
used to define corpus coverage.

Example:
    python -m src.data_analysis.model_coverage_statistics \
        --tag 251117 --v2_mode
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter


INK = "#25313c"
BLUE = "#2f6f9f"
LIGHT_BLUE = "#83b6d8"
ORANGE = "#d9822b"
GRID = "#d9dee3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True, help="Processing tag, e.g. 251117")
    parser.add_argument(
        "--snapshot-date",
        default="2025-09-20",
        help="Public Hugging Face dataset snapshot date (default: 2025-09-20)",
    )
    parser.add_argument("--v2_mode", action="store_true", help="Use v2 processed tables")
    parser.add_argument("--output-dir", default="data/analysis")
    return parser.parse_args()


def format_count(value: int) -> str:
    return f"{value / 1_000:.0f}K" if value >= 100_000 else f"{value:,}"


def main() -> None:
    args = parse_args()
    suffix = f"_{args.tag}"
    v2_suffix = "_v2" if args.v2_mode else ""
    step1 = Path(f"data/processed/modelcard_step1{suffix}.parquet")
    step3 = Path(f"data/processed/modelcard_step3_dedup{v2_suffix}{suffix}.parquet")
    for path in (step1, step3):
        if not path.exists():
            raise FileNotFoundError(path)

    con = duckdb.connect()
    step1_sql = str(step1.resolve()).replace("'", "''")
    step3_sql = str(step3.resolve()).replace("'", "''")
    row = con.execute(
        f"""
        WITH cards AS (
            SELECT * FROM read_parquet('{step1_sql}')
        ), table_models AS (
            SELECT DISTINCT modelId
            FROM read_parquet('{step3_sql}')
            WHERE (hugging_table_list_dedup IS NOT NULL AND len(hugging_table_list_dedup) > 0)
               OR (github_table_list_dedup IS NOT NULL AND len(github_table_list_dedup) > 0)
               OR (html_table_list_mapped_dedup IS NOT NULL AND len(html_table_list_mapped_dedup) > 0)
        )
        SELECT
            count(*) AS snapshot_records,
            count(*) FILTER (
                WHERE card_readme IS NOT NULL
                  AND length(trim(card_readme)) > 0
                  AND card_readme <> 'Entry not found'
            ) AS nonempty_readme,
            count(*) FILTER (WHERE modelId IN (SELECT modelId FROM table_models)) AS linked_table,
            count(*) FILTER (WHERE card_tags IS NOT NULL AND length(trim(card_tags)) > 0) AS metadata_tags,
            count(*) FILTER (WHERE pipeline_tag IS NOT NULL AND length(trim(pipeline_tag)) > 0) AS pipeline_tags,
            count(*) FILTER (WHERE all_links IS NOT NULL AND length(trim(all_links)) > 0) AS external_links,
            count(*) FILTER (WHERE github_link IS NOT NULL AND len(github_link) > 0) AS github_links,
            count(*) FILTER (WHERE pdf_link IS NOT NULL AND len(pdf_link) > 0) AS paper_links,
            count(*) FILTER (
                WHERE parsed_bibtex_tuple_list IS NOT NULL
                  AND len(parsed_bibtex_tuple_list) > 0
            ) AS parsed_bibtex
        FROM cards
        """
    ).fetchone()
    keys = [
        "snapshot_records", "nonempty_readme", "linked_table", "metadata_tags",
        "pipeline_tags", "external_links", "github_links", "paper_links", "parsed_bibtex",
    ]
    counts = dict(zip(keys, map(int, row)))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / f"model_coverage_statistics{v2_suffix}{suffix}"
    payload = {
        "processing_tag": args.tag,
        "snapshot_date": args.snapshot_date,
        "scope_note": "Snapshot contains Hugging Face model-card records, not cardless Hub repositories.",
        "table_sources": ["ModelCard", "GitHub", "arXiv"],
        "counts": counts,
    }
    prefix.with_suffix(".json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with prefix.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "count", "share_of_snapshot"])
        for key, value in counts.items():
            writer.writerow([key, value, f"{value / counts['snapshot_records']:.6f}"])

    funnel = [
        ("Crawled model-card records", counts["snapshot_records"]),
        ("Non-empty READMEs", counts["nonempty_readme"]),
        ("Models linked to $\\geq$1 table", counts["linked_table"]),
    ]
    coverage = [
        ("Metadata tags", counts["metadata_tags"]),
        ("External links", counts["external_links"]),
        ("Pipeline tags", counts["pipeline_tags"]),
        ("GitHub links", counts["github_links"]),
        ("Paper links", counts["paper_links"]),
        ("Parsed BibTeX", counts["parsed_bibtex"]),
    ]

    # One shared axis makes all quantities directly comparable. Use the same
    # dark-to-light Blues palette as the original coverage figures.
    items = funnel + coverage
    names, values = zip(*items)
    positions = [8.0, 7.2, 6.4, 5.1, 4.3, 3.5, 2.7, 1.9, 1.1]
    colors = plt.cm.Blues(np.linspace(0.86, 0.34, len(items)))

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 10, "axes.labelcolor": INK, "text.color": INK})
    fig, ax = plt.subplots(figsize=(7.15, 3.05))
    bars = ax.barh(positions, values, color=colors, edgecolor="none", linewidth=0)
    ax.set_yticks(positions, names)
    ax.tick_params(axis="y", labelsize=11, length=0, pad=8)
    ax.set_xlim(0, counts["snapshot_records"] * 1.13)
    ax.set_xlabel("Model-card records")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1_000:.0f}K"))
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, values):
        ax.text(value + counts["snapshot_records"] * 0.015, bar.get_y() + bar.get_height() / 2,
                format_count(value), ha="left", va="center")
    # Visually separate the three corpus-stage quantities from auxiliary fields.
    ax.axhline(5.88, color=GRID, linewidth=0.8)
    # Center both group headings on the same vertical axis as the x-axis label.
    ax.text(0.5, 8.48, "Corpus stages", transform=ax.get_yaxis_transform(),
            ha="center", fontsize=10.5, color=INK)
    ax.text(0.5, 5.52, "Metadata and links", transform=ax.get_yaxis_transform(),
            ha="center", fontsize=10.5, color=INK)
    ax.set_ylim(0.65, 8.7)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="x", colors=INK, length=0)
    fig.tight_layout()
    fig.savefig(prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(json.dumps(payload, indent=2))
    print(f"Saved {prefix}.json/.csv/.pdf/.png")


if __name__ == "__main__":
    main()
