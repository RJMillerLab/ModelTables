"""Plot the original dual-axis corpus comparison with validated averages.

``# Tables`` and ``# Cols`` are totals in the input parquet, while
``Avg # Rows`` is already an average. Therefore average columns are computed as
``# Cols / # Tables`` before plotting. The left axis is corpus size; the right
axis is average table shape. Both use log scales.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BENCHMARK_ORDER = [
    "SANTOS Small", "TUS Small", "TUS Large", "SANTOS Large", "WDC",
    "GitTable", "WikiTables", "UGEN-V1", "UGEN-V2",
    "scilake-hugging", "scilake-github", "scilake-html",
]

DISPLAY_NAMES = {
    "SANTOS Small": "SANTOS-S", "TUS Small": "TUS-S",
    "TUS Large": "TUS-L", "SANTOS Large": "SANTOS-L", "WDC": "WDC",
    "GitTable": "GitTables", "WikiTables": "WikiTables",
    "UGEN-V1": "UGEN-V1", "UGEN-V2": "UGEN-V2",
    "scilake-hugging": "Our-ModelCard", "scilake-github": "Our-GitHub",
    "scilake-html": "Our-arXiv",
}

TABLES_BLUE = "#274C77"
SHAPE_ORANGE = "#C65D3A"


def format_table_count(value: float) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 100_000:
        return f"{value / 1_000:.0f}K"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:.0f}"


def load_metrics(input_path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(input_path).copy()
    required = {"Benchmark", "# Tables", "# Cols", "Avg # Rows"}
    missing_columns = required.difference(frame.columns)
    if missing_columns:
        raise ValueError(f"Missing columns: {sorted(missing_columns)}")
    frame["Avg # Cols"] = frame["# Cols"] / frame["# Tables"]
    indexed = frame.set_index("Benchmark")
    missing_rows = [name for name in BENCHMARK_ORDER if name not in indexed.index]
    if missing_rows:
        raise ValueError(f"Missing benchmark rows: {missing_rows}")
    selected = indexed.loc[BENCHMARK_ORDER].copy()
    metrics = selected[["# Tables", "Avg # Rows", "Avg # Cols"]].to_numpy(dtype=float)
    if not np.isfinite(metrics).all() or (metrics <= 0).any():
        raise ValueError("Corpus metrics must be finite and positive")
    return selected


def plot(input_path: Path, output_prefix: Path, annotate_ours: bool = True) -> None:
    selected = load_metrics(input_path)
    x = np.arange(len(selected))

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.5,
        "axes.labelsize": 9, "xtick.labelsize": 7.3,
        "ytick.labelsize": 8, "legend.fontsize": 7.5,
    })
    fig, tables_ax = plt.subplots(figsize=(7.15, 3.45))
    shape_ax = tables_ax.twinx()

    tables_ax.axvspan(-0.5, 8.5, color="#F7F4F1", zorder=0)
    tables_ax.axvspan(8.5, 11.5, color="#EEF3F7", zorder=0)
    tables_ax.axvline(8.5, color="#8493A1", linewidth=0.8, linestyle=(0, (3, 2)), zorder=1)

    tables_line = tables_ax.plot(
        x, selected["# Tables"], color=TABLES_BLUE, marker="o",
        markersize=4.2, markeredgecolor="white", markeredgewidth=0.6,
        linewidth=1.6, label="Tables", zorder=4,
    )[0]
    rows_line = shape_ax.plot(
        x, selected["Avg # Rows"], color=SHAPE_ORANGE, marker="s",
        markersize=3.8, markerfacecolor="white", markeredgewidth=1.0,
        linewidth=1.4, linestyle="--", label="Avg. rows", zorder=3,
    )[0]
    cols_line = shape_ax.plot(
        x, selected["Avg # Cols"], color=SHAPE_ORANGE, marker="^",
        markersize=4.2, markeredgecolor=SHAPE_ORANGE, markeredgewidth=0.7,
        linewidth=1.4, linestyle=":", label="Avg. cols", zorder=3,
    )[0]

    tables_ax.set_yscale("log")
    shape_ax.set_yscale("log")
    tables_ax.set_ylim(3e2, 1.2e8)
    shape_ax.set_ylim(3.5, 1.5e4)
    tables_ax.set_ylabel("# Tables (log)", color=TABLES_BLUE)
    shape_ax.set_ylabel("Avg. rows / cols (log)", color=SHAPE_ORANGE)
    tables_ax.tick_params(axis="y", colors=TABLES_BLUE)
    shape_ax.tick_params(axis="y", colors=SHAPE_ORANGE)
    tables_ax.spines["left"].set_color(TABLES_BLUE)
    shape_ax.spines["right"].set_color(SHAPE_ORANGE)
    tables_ax.set_xticks(x, [DISPLAY_NAMES[name] for name in BENCHMARK_ORDER], rotation=48, ha="right")
    tables_ax.set_xlim(-0.5, len(x) - 0.5)
    tables_ax.grid(axis="y", which="major", color="#C8D0D7", linewidth=0.6, alpha=0.75)
    tables_ax.set_axisbelow(True)
    tables_ax.text(4.0, 0.985, "Prior corpora", transform=tables_ax.get_xaxis_transform(),
                   ha="center", va="top", color="#675D58", fontsize=8)
    tables_ax.text(10.0, 0.985, "ModelTables", transform=tables_ax.get_xaxis_transform(),
                   ha="center", va="top", color=TABLES_BLUE, fontsize=8)
    tables_ax.legend(
        [tables_line, rows_line, cols_line], ["Tables", "Avg. rows", "Avg. cols"],
        loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3, frameon=False,
        handlelength=1.8, columnspacing=0.8, handletextpad=0.4,
    )

    if annotate_ours:
        for index in range(9, 12):
            tables_ax.annotate(
                format_table_count(float(selected["# Tables"].iloc[index])),
                (index, float(selected["# Tables"].iloc[index])), xytext=(0, 5),
                textcoords="offset points", ha="center", va="bottom",
                color=tables_line.get_color(), fontsize=6.4,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 0.35},
            )
            shape_ax.annotate(
                f"{selected['Avg # Rows'].iloc[index]:.1f}",
                (index, float(selected["Avg # Rows"].iloc[index])), xytext=(5, 0),
                textcoords="offset points", ha="left", va="center",
                color=rows_line.get_color(), fontsize=6.4,
            )
            shape_ax.annotate(
                f"{selected['Avg # Cols'].iloc[index]:.1f}",
                (index, float(selected["Avg # Cols"].iloc[index])), xytext=(-5, 0),
                textcoords="offset points", ha="right", va="center",
                color=SHAPE_ORANGE, fontsize=6.4,
            )

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.10, right=0.90, top=0.88, bottom=0.25)
    fig.savefig(output_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/analysis/benchmark_results_v2_251117.parquet")
    parser.add_argument("--output-prefix", default="data/analysis/corpus_statistics_compact")
    args = parser.parse_args()
    selected = load_metrics(Path(args.input))
    print(selected[["# Tables", "Avg # Rows", "Avg # Cols"]].to_string())
    plot(Path(args.input), Path(args.output_prefix))
    print(f"saved={args.output_prefix}.png/.pdf")


if __name__ == "__main__":
    main()
