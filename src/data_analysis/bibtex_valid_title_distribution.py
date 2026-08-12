"""Plot per-model-card distributions for parsed BibTeX entries and valid titles.

Usage:
python -m src.data_analysis.bibtex_valid_title_distribution --tag 251117 --v2_mode
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


BIN_SPECS = (
    ("1", 1, 1),
    ("2", 2, 2),
    ("3", 3, 3),
    ("4", 4, 4),
    ("5", 5, 5),
    ("6-10", 6, 10),
    ("11-20", 11, 20),
    ("21+", 21, None),
)
COLORS = ("#dceeff", "#b9dcf7", "#90c6ed", "#62a9df", "#3e89c7", "#226da9", "#165481", "#103b61")


def binned_counts(series: pd.Series) -> list[int]:
    """Return counts in compact exact-count and long-tail bins."""
    return [
        int((series.ge(lower) if upper is None else series.between(lower, upper)).sum())
        for _, lower, upper in BIN_SPECS
    ]


def draw_panel(ax: plt.Axes, counts: list[int], total: int, title: str, xlabel: str) -> None:
    labels = [label for label, _, _ in BIN_SPECS]
    bars = ax.bar(labels, counts, color=COLORS, edgecolor="#12344d", linewidth=0.55)
    ymax = max(counts) if counts else 0
    ax.set_ylim(0, ymax * 1.16 if ymax else 1)
    ax.set_title(title, fontsize=11.5, color="#17212b", pad=10)
    ax.set_xlabel(xlabel, fontsize=9.5)
    ax.grid(axis="y", color="#dce3e8", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#8292a0")
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            count + ymax * 0.022,
            f"{count:,}\n{count / total:.1%}",
            ha="center",
            va="bottom",
            fontsize=7.5,
            color="#17212b",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default=None, help="Data tag, e.g. 251117.")
    parser.add_argument("--v2_mode", action="store_true", help="Use the v2 valid-title parquet name.")
    parser.add_argument("--output-prefix", default=None, help="Output prefix without .png/.pdf.")
    args = parser.parse_args()

    suffix = f"_{args.tag}" if args.tag else ""
    v2_suffix = "_v2" if args.v2_mode else ""
    step1_path = Path("data/processed") / f"modelcard_step1{suffix}.parquet"
    valid_title_path = Path("data/processed") / f"all_title_list_valid{v2_suffix}{suffix}.parquet"
    output_prefix = Path(args.output_prefix or f"data/analysis/bibtex_valid_title_count_distribution{v2_suffix}{suffix}")
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    bibtex_counts = pd.read_parquet(step1_path, columns=["successful_parse_count"])["successful_parse_count"]
    bibtex_counts = bibtex_counts.fillna(0).round().astype(int)
    bibtex_counts = bibtex_counts[bibtex_counts > 0]

    valid_title_counts = pd.read_parquet(valid_title_path, columns=["all_title_list_valid"])["all_title_list_valid"]
    valid_title_counts = valid_title_counts.map(lambda titles: len(titles) if titles is not None else 0)
    valid_title_counts = valid_title_counts[valid_title_counts > 0]

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 3.8), constrained_layout=True)
    draw_panel(
        axes[0],
        binned_counts(bibtex_counts),
        len(bibtex_counts),
        f"Parsed BibTeX entries (n = {len(bibtex_counts):,})",
        "Successfully parsed entries per model card",
    )
    draw_panel(
        axes[1],
        binned_counts(valid_title_counts),
        len(valid_title_counts),
        f"Valid paper titles (n = {len(valid_title_counts):,})",
        "Resolved valid titles per model card",
    )
    axes[0].set_ylabel("Model cards", fontsize=10)
    fig.savefig(f"{output_prefix}.png", dpi=220, bbox_inches="tight")
    fig.savefig(f"{output_prefix}.pdf", bbox_inches="tight")
    print(f"Saved {output_prefix}.png and {output_prefix}.pdf")


if __name__ == "__main__":
    main()
