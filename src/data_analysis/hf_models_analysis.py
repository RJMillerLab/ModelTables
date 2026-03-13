#!/usr/bin/env python
"""hf_models_analysis.py – statistics from local raw parquet + modelcard_step3_dedup*.parquet.

Outputs:
- data/analysis/hf_models_analysis{v2_suffix}{suffix}.pdf/png
- data/analysis/hf_cross_analysis{v2_suffix}{suffix}.pdf/png
and prints JSON counts plus validation info to stdout.

COMPLETE VALIDATION: No sampling, checks all models with tables to ensure
they have valid modelcards.
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import duckdb

# Reuse the same valid‑card condition as in model_snapshot_overlap.py
VALID_CARD_COND = "card IS NOT NULL AND card <> '' AND card <> 'Entry not found'"

def main(suffix, v2_suffix) -> None:
    step3_dedup_path = os.path.join("data", "processed", f"modelcard_step3_dedup{v2_suffix}{suffix}.parquet")

    # -------- paths / config for raw modelcards --------
    # Align with card_statistics: pull "All Models" / "Models w/ Cards" from local raw parquet.
    #
    # Untagged: data/raw/train-*-of-00004.parquet
    # Tagged:   data/raw_<tag>/train-*-of-00006.parquet
    if suffix:
        raw_glob = os.path.join("data", f"raw{suffix}", "train-*-of-00006.parquet")
    else:
        raw_glob = os.path.join("data", "raw", "train-*-of-00004.parquet")

    print(f"Using raw_glob={raw_glob}")
    print(f"Using STEP3_DEDUP_PATH={step3_dedup_path}")

    # -------- duckdb connections --------
    con_main = duckdb.connect()  # in-memory, for raw_glob queries
    con_parquet = duckdb.connect()

    def q_main(sql: str) -> int:
        """Run a scalar query on the main connection (raw_glob)."""
        return con_main.execute(sql).fetchone()[0]

    def q_parquet(sql: str) -> int:
        """Run a scalar query on the parquet connection."""
        return con_parquet.execute(sql).fetchone()[0]

    # ----- counts dict -----
    counts: dict[str, int] = {}

    # 1) All models in HF (from local raw parquet)
    print("Counting all models (from raw parquet)...")
    counts["All Models"] = q_main(f"SELECT COUNT(*) FROM read_parquet('{raw_glob}')")

    # 2) Models with cards (non-empty, not 'Entry not found')
    print("Counting models with valid cards (from raw parquet)...")
    counts["Models w/ Cards"] = q_main(
        f"""
        SELECT COUNT(*) FROM read_parquet('{raw_glob}')
        WHERE card IS NOT NULL AND card <> '' AND card <> 'Entry not found'
        """
    )

    # 3) Models with tables (ANY source), **among models that already have valid cards**
    #    This keeps the bar chart as a true step‑by‑step filter:
    #    All Models → Models w/ Cards → Models w/ Any Table (with cards) → Models w/ Hugging Tables (with cards).
    print("Counting models with tables from any source (restricted to models with valid cards)...")
    counts["Models w/ Any Table"] = q_parquet(
        f"""
        SELECT COUNT(DISTINCT r.modelId)
        FROM read_parquet('{raw_glob}') AS r
        JOIN read_parquet('{step3_dedup_path}') AS s
          ON r.modelId = s.modelId
        WHERE {VALID_CARD_COND}
          AND (
              hugging_table_list_dedup IS NOT NULL AND array_length(hugging_table_list_dedup) > 0
           OR html_table_list_mapped_dedup IS NOT NULL AND array_length(html_table_list_mapped_dedup) > 0
           OR llm_table_list_mapped_dedup IS NOT NULL AND array_length(llm_table_list_mapped_dedup) > 0
           OR github_table_list_dedup IS NOT NULL AND array_length(github_table_list_dedup) > 0
          )
        """
    )

    # 4) Models with HuggingFace tables specifically, **also restricted to models with valid cards**
    print("Counting models with HuggingFace tables (restricted to models with valid cards)...")
    counts["Models w/ Hugging Tables"] = q_parquet(
        f"""
        SELECT COUNT(DISTINCT r.modelId)
        FROM read_parquet('{raw_glob}') AS r
        JOIN read_parquet('{step3_dedup_path}') AS s
          ON r.modelId = s.modelId
        WHERE {VALID_CARD_COND}
          AND hugging_table_list_dedup IS NOT NULL
          AND array_length(hugging_table_list_dedup) > 0
        """
    )

    # ---------- COMPLETE VALIDATION: Double Check all models with tables ----------
    print("\n=== COMPLETE VALIDATION ===")
    print("Validating that ALL models with tables have valid modelcards...")

    validation_query = f"""
        WITH models_with_tables AS (
            SELECT DISTINCT modelId
            FROM read_parquet('{step3_dedup_path}')
            WHERE (hugging_table_list_dedup IS NOT NULL AND array_length(hugging_table_list_dedup) > 0)
               OR (html_table_list_mapped_dedup IS NOT NULL AND array_length(html_table_list_mapped_dedup) > 0)
               OR (llm_table_list_mapped_dedup IS NOT NULL AND array_length(llm_table_list_mapped_dedup) > 0)
               OR (github_table_list_dedup IS NOT NULL AND array_length(github_table_list_dedup) > 0)
        ),
        raw_with_flag AS (
            SELECT
                modelId,
                CASE 
                    WHEN card IS NULL OR card = '' OR card = 'Entry not found' 
                    THEN 'NO_CARD' 
                    ELSE 'HAS_CARD' 
                END as card_status,
                card as card_content
            FROM read_parquet('{raw_glob}')
        )
        SELECT 
            mwt.modelId,
            rw.card_status,
            rw.card_content
        FROM models_with_tables mwt
        LEFT JOIN raw_with_flag rw ON mwt.modelId = rw.modelId
        ORDER BY mwt.modelId
    """

    validation_results = con_main.execute(validation_query).fetchdf()

    models_with_tables_total = len(validation_results)
    models_with_tables_and_cards = len(
        validation_results[validation_results["card_status"] == "HAS_CARD"]
    )
    models_with_tables_no_cards = len(
        validation_results[validation_results["card_status"] == "NO_CARD"]
    )

    print(f"Total models with tables: {models_with_tables_total:,}")
    print(f"Models with tables AND valid cards: {models_with_tables_and_cards:,}")
    print(f"Models with tables BUT NO valid cards: {models_with_tables_no_cards:,}")

    if models_with_tables_no_cards > 0:
        print(
            f"\n⚠️  WARNING: Found {models_with_tables_no_cards} models with tables but NO valid modelcards!"
        )
        print("Examples of models with tables but no cards:")
        examples_no_cards = validation_results[
            validation_results["card_status"] == "NO_CARD"
        ].head(10)
        for _, row in examples_no_cards.iterrows():
            print(f"  - {row['modelId']}: {row['card_status']}")
    else:
        print("✅ SUCCESS: All models with tables have valid modelcards!")

    # ---------- CROSS ANALYSIS ----------
    print("\n=== CROSS ANALYSIS ===")

    models_with_cards = counts["Models w/ Cards"]
    models_without_cards = counts["All Models"] - models_with_cards

    print("\nModelCards Analysis:")
    print(f"Models WITH modelcards: {models_with_cards:,}")
    print(f"Models WITHOUT modelcards: {models_without_cards:,}")
    print(f"Difference: {models_with_cards - models_without_cards:,}")

    models_with_tables = counts["Models w/ Any Table"]
    models_without_tables = counts["All Models"] - models_with_tables

    print("\nTables Analysis:")
    print(f"Models WITH tables: {models_with_tables:,}")
    print(f"Models WITHOUT tables: {models_without_tables:,}")
    print(f"Difference: {models_with_tables - models_without_tables:,}")

    models_with_both = models_with_tables_and_cards  # From validation above
    models_with_cards_no_tables = models_with_cards - models_with_both
    models_with_tables_no_cards = models_with_tables - models_with_both

    print("\nCross Analysis (ModelCards vs Tables):")
    print(f"Models with BOTH cards and tables: {models_with_both:,}")
    print(f"Models with cards but NO tables: {models_with_cards_no_tables:,}")
    print(f"Models with tables but NO cards: {models_with_tables_no_cards:,}")
    neither = (
        counts["All Models"]
        - models_with_cards
        - models_with_tables
        + models_with_both
    )
    print(f"Models with NEITHER: {neither:,}")

    # ---------- MAIN STATISTICS JSON ----------
    print("\n=== MAIN STATISTICS ===")
    print("HuggingFace Models Analysis:")
    print(json.dumps(counts, indent=2))

    # ---------- detailed breakdown ----------
    print("\nDetailed breakdown:")
    print(f"Total models: {counts['All Models']:,}")
    print(f"Models with cards: {counts['Models w/ Cards']:,}")
    print(f"Models with tables: {counts['Models w/ Any Table']:,}")
    print(f"Models with HuggingFace tables: {counts['Models w/ Hugging Tables']:,}")

    total_models = counts["All Models"]
    print("\nPercentages:")
    print(f"Models with cards: {counts['Models w/ Cards']/total_models*100:.1f}%")
    print(f"Models with tables: {counts['Models w/ Any Table']/total_models*100:.1f}%")
    print(
        f"Models with HuggingFace tables: {counts['Models w/ Hugging Tables']/total_models*100:.1f}%"
    )

    # ---------- table source distribution ----------
    print("\nTable source distribution:")
    table_sources = con_parquet.execute(
        f"""
        SELECT 
            COUNT(*) as total_models_with_tables,
            COUNT(CASE WHEN hugging_table_list_dedup IS NOT NULL AND array_length(hugging_table_list_dedup) > 0 THEN 1 END) as hugging_source,
            COUNT(CASE WHEN html_table_list_mapped_dedup IS NOT NULL AND array_length(html_table_list_mapped_dedup) > 0 THEN 1 END) as html_source,
            COUNT(CASE WHEN llm_table_list_mapped_dedup IS NOT NULL AND array_length(llm_table_list_mapped_dedup) > 0 THEN 1 END) as llm_source,
            COUNT(CASE WHEN github_table_list_dedup IS NOT NULL AND array_length(github_table_list_dedup) > 0 THEN 1 END) as github_source
        FROM read_parquet('{step3_dedup_path}')
        WHERE (hugging_table_list_dedup IS NOT NULL AND array_length(hugging_table_list_dedup) > 0)
           OR (html_table_list_mapped_dedup IS NOT NULL AND array_length(html_table_list_mapped_dedup) > 0)
           OR (llm_table_list_mapped_dedup IS NOT NULL AND array_length(llm_table_list_mapped_dedup) > 0)
           OR (github_table_list_dedup IS NOT NULL AND array_length(github_table_list_dedup) > 0)
        """
    ).fetchdf()

    print(
        f"Total models with tables: {table_sources.iloc[0]['total_models_with_tables']:,}"
    )
    print(f"From HuggingFace cards: {table_sources.iloc[0]['hugging_source']:,}")
    print(f"From HTML/PDF: {table_sources.iloc[0]['html_source']:,}")
    print(f"From LLM processing: {table_sources.iloc[0]['llm_source']:,}")
    print(f"From GitHub READMEs: {table_sources.iloc[0]['github_source']:,}")

    # ---------- plot main statistics (4 columns) ----------
    plt.figure(figsize=(14, 8))
    bars = plt.bar(
        counts.keys(), counts.values(), color=plt.cm.Blues(np.linspace(0.8, 0.4, len(counts)))
    )

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{int(height):,}",
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight="bold",
        )

    plt.ylabel("Number of Models", fontsize=16)
    plt.xlabel("Step by Step Filtering", fontsize=16)
    plt.xticks(rotation=0, ha="center", fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    os.makedirs("data/analysis", exist_ok=True)
    out_suffix = f"{v2_suffix}{suffix}"
    plt.savefig(
        f"data/analysis/hf_models_analysis{out_suffix}.pdf",
        format="pdf",
        dpi=300,
        bbox_inches="tight",
    )
    plt.savefig(
        f"data/analysis/hf_models_analysis{out_suffix}.png",
        format="png",
        dpi=300,
        bbox_inches="tight",
    )
    print(
        f"\nsave fig to data/analysis/hf_models_analysis{out_suffix}.pdf and data/analysis/hf_models_analysis{out_suffix}.png"
    )
    plt.close()

    # ---------- plot cross analysis ----------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Subplot 1: ModelCards vs No ModelCards
    labels1 = ["With ModelCards", "Without ModelCards"]
    values1 = [models_with_cards, models_without_cards]
    colors1 = ["#2ca02c", "#d62728"]
    bars1 = ax1.bar(labels1, values1, color=colors1)
    ax1.set_title("Models with vs without ModelCards", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Number of Models", fontsize=12)

    for bar in bars1:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{int(height):,}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # Subplot 2: Tables vs No Tables
    labels2 = ["With Tables", "Without Tables"]
    values2 = [models_with_tables, models_without_tables]
    colors2 = ["#1f77b4", "#ff7f0e"]
    bars2 = ax2.bar(labels2, values2, color=colors2)
    ax2.set_title("Models with vs without Tables", fontsize=14, fontweight="bold")
    ax2.set_ylabel("Number of Models", fontsize=12)

    for bar in bars2:
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{int(height):,}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(
        f"data/analysis/hf_cross_analysis{out_suffix}.pdf",
        format="pdf",
        dpi=300,
        bbox_inches="tight",
    )
    plt.savefig(
        f"data/analysis/hf_cross_analysis{out_suffix}.png",
        format="png",
        dpi=300,
        bbox_inches="tight",
    )
    print(
        f"save cross analysis fig to data/analysis/hf_cross_analysis{out_suffix}.pdf and data/analysis/hf_cross_analysis{out_suffix}.png"
    )
    plt.close()

    # Close connections
    con_main.close()
    con_parquet.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze HuggingFace models and tables coverage.")
    parser.add_argument("--tag", type=str, default=None, help="Tag suffix for versioning (e.g., 251117).")
    parser.add_argument("--v2_mode", action="store_true", help="Use v2 mode.")
    args = parser.parse_args()

    suffix = f"_{args.tag}" if args.tag else ""
    v2_suffix = "_v2" if args.v2_mode else ""
    main(suffix, v2_suffix)

