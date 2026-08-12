"""Plot a compact dual-axis comparison of corpus statistics."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


BENCHMARK_ORDER = [
    "SANTOS Small",
    "TUS Small",
    "TUS Large",
    "SANTOS Large",
    "WDC",
    "GitTable",
    "WikiTables",
    "UGEN-V1",
    "UGEN-V2",
    "scilake-hugging",
    "scilake-github",
    "scilake-html",
]

DISPLAY_NAMES = {
    "SANTOS Small": "SANTOS-S",
    "TUS Small": "TUS-S",
    "TUS Large": "TUS-L",
    "SANTOS Large": "SANTOS-L",
    "WDC": "WDC",
    "GitTable": "GitTables",
    "WikiTables": "WikiTables",
    "UGEN-V1": "UGEN-V1",
    "UGEN-V2": "UGEN-V2",
    "scilake-hugging": "Our-ModelCard",
    "scilake-github": "Our-GitHub",
    "scilake-html": "Our-arXiv",
}


def format_table_count(value: float) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 100_000:
        return f"{value / 1_000:.0f}K"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:.0f}"


def plot(input_path: Path, output_prefix: Path, annotate_ours: bool = False) -> None:
    frame = pd.read_parquet(input_path).copy()
    frame["Avg # Cols"] = frame["# Cols"] / frame["# Tables"]
    indexed = frame.set_index("Benchmark")
    missing = [name for name in BENCHMARK_ORDER if name not in indexed.index]
    if missing:
        raise ValueError(f"Missing benchmark rows: {missing}")
    selected = indexed.loc[BENCHMARK_ORDER]
    x = list(range(len(selected)))

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "xtick.labelsize": 7.3,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.5,
        }
    )
    fig, left = plt.subplots(figsize=(3.35, 3.05))
    right = left.twinx()

    left.axvspan(-0.5, 8.5, color="#F7F4F1", zorder=0)
    left.axvspan(8.5, 11.5, color="#EEF3F7", zorder=0)
    left.axvline(8.5, color="#8493A1", linewidth=0.8, linestyle=(0, (3, 2)), zorder=1)

    tables_line = left.plot(
        x,
        selected["# Tables"],
        color="#274C77",
        marker="o",
        markersize=4.2,
        markeredgecolor="white",
        markeredgewidth=0.6,
        linewidth=1.6,
        label="Tables",
        zorder=4,
    )[0]
    rows_line = right.plot(
        x,
        selected["Avg # Rows"],
        color="#557A95",
        marker="s",
        markersize=3.8,
        markerfacecolor="white",
        markeredgewidth=1.0,
        linewidth=1.4,
        linestyle="--",
        label="Avg. rows",
        zorder=3,
    )[0]
    cols_line = right.plot(
        x,
        selected["Avg # Cols"],
        color="#8FB3C5",
        marker="^",
        markersize=4.2,
        markeredgecolor="#557A95",
        markeredgewidth=0.7,
        linewidth=1.4,
        linestyle=":",
        label="Avg. cols",
        zorder=3,
    )[0]

    left.set_yscale("log")
    right.set_yscale("log")
    left.set_ylim(3e2, 1.2e8)
    right.set_ylim(3.5, 1.5e4)
    left.set_ylabel("# Tables (log)")
    right.set_ylabel("Avg. rows / cols (log)")
    left.set_xticks(x, [DISPLAY_NAMES[name] for name in BENCHMARK_ORDER], rotation=90, ha="center", va="top")
    left.set_xlim(-0.5, len(x) - 0.5)
    left.grid(axis="y", which="major", color="#C8D0D7", linewidth=0.6, alpha=0.75)
    left.set_axisbelow(True)
    left.spines["top"].set_visible(False)
    right.spines["top"].set_visible(False)

    left.text(4.0, 1.015, "Prior corpora", transform=left.get_xaxis_transform(), ha="center", va="bottom", color="#675D58", fontsize=7.5)
    left.text(10.0, 1.015, "Our benchmark", transform=left.get_xaxis_transform(), ha="center", va="bottom", color="#274C77", fontsize=7.5)
    left.legend(
        [tables_line, rows_line, cols_line],
        ["Tables", "Avg. rows", "Avg. cols"],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.105),
        ncol=3,
        frameon=False,
        handlelength=1.8,
        columnspacing=0.8,
        handletextpad=0.4,
    )

    if annotate_ours:
        for index in range(9, 12):
            left.annotate(
                format_table_count(float(selected["# Tables"].iloc[index])),
                (index, float(selected["# Tables"].iloc[index])),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                va="bottom",
                color=tables_line.get_color(),
                fontsize=5.8,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 0.35},
            )
            right.annotate(
                f"{float(selected['Avg # Rows'].iloc[index]):.1f}",
                (index, float(selected["Avg # Rows"].iloc[index])),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                color=rows_line.get_color(),
                fontsize=5.8,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 0.35},
            )
            right.annotate(
                f"{float(selected['Avg # Cols'].iloc[index]):.1f}",
                (index, float(selected["Avg # Cols"].iloc[index])),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                color="#557A95",
                fontsize=5.8,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 0.35},
            )

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.18, right=0.82, top=0.78, bottom=0.34)
    fig.savefig(output_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/analysis/benchmark_results_v2_251117.parquet")
    parser.add_argument("--output-prefix", default="data/analysis/benchmark_metrics_single_column_v2_251117")
    args = parser.parse_args()
    output_prefix = Path(args.output_prefix)
    plot(Path(args.input), output_prefix, annotate_ours=False)
    annotated_prefix = output_prefix.with_name(f"{output_prefix.name}_annotated")
    plot(Path(args.input), annotated_prefix, annotate_ours=True)
    print(f"saved={output_prefix}.png/.pdf")
    print(f"saved={annotated_prefix}.png/.pdf")


if __name__ == "__main__":
    main()
