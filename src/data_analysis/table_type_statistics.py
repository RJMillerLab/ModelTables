"""Generate a compact paper figure from table-type labels."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


LABEL_ORDER = ["performance_metric", "performance_context", "hf_training_log", "non_performance"]
LABEL_NAMES = {
    "performance_metric": "Performance: metric",
    "performance_context": "Performance: context",
    "hf_training_log": "Non-performance: training log",
    "non_performance": "Non-performance: other",
}
COLORS = {
    "performance_metric": "#0B4F8A",
    "performance_context": "#2878B5",
    "hf_training_log": "#76ADD3",
    "non_performance": "#BDD7E9",
}
RESOURCE_ORDER = ["huggingface", "github", "paper_html"]
RESOURCE_NAMES = {"huggingface": "Hugging Face", "github": "GitHub", "paper_html": "arXiv HTML"}


def read_labels(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def plot(rows: list[dict[str, str]], output_prefix: Path) -> None:
    label_counts = Counter(row["label"] for row in rows)
    resource_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        resource_counts[row["resource"]][row["label"]] += 1
    labels = [label for label in LABEL_ORDER if label_counts[label]]
    plt.rcParams.update({"font.size": 10, "axes.labelsize": 10, "axes.titlesize": 11})
    resources = [resource for resource in RESOURCE_ORDER if resource_counts[resource]]
    groups = [("all", "All sources"), *[(resource, RESOURCE_NAMES[resource]) for resource in resources]]
    totals = {"all": len(rows), **{resource: sum(resource_counts[resource].values()) for resource in resources}}
    bottoms = [0.0] * len(groups)
    fig, bar_ax = plt.subplots(figsize=(7.0, 2.25))
    for label in labels:
        values = [
            100 * (label_counts[label] if group == "all" else resource_counts[group][label]) / totals[group]
            for group, _ in groups
        ]
        bar_ax.bar(range(len(groups)), values, bottom=bottoms, color=COLORS[label], edgecolor="white", linewidth=0.7, label=LABEL_NAMES[label])
        bottoms = [bottom + value for bottom, value in zip(bottoms, values)]
    bar_ax.set_ylim(0, 100)
    bar_ax.set_yticks([0, 25, 50, 75, 100])
    bar_ax.set_ylabel("Tables (%)")
    bar_ax.set_xticks(range(len(groups)), [name for _, name in groups])
    bar_ax.set_title("Table types by source")
    bar_ax.spines[["top", "right"]].set_visible(False)
    bar_ax.grid(axis="y", color="#D8E2EA", linewidth=0.6)
    bar_ax.set_axisbelow(True)

    handles, legend_labels = bar_ax.get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=4, frameon=False, columnspacing=0.9, handlelength=1.1, fontsize=8, bbox_to_anchor=(0.5, 0.01))
    fig.subplots_adjust(left=0.09, right=0.99, top=0.84, bottom=0.30)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot table-type label statistics for the paper.")
    parser.add_argument("--labels", default="data/table_type/table_type_labels_v2_251117.tsv")
    parser.add_argument("--output-prefix", default="data/table_type/table_type_statistics_v2_251117")
    args = parser.parse_args()
    rows = read_labels(Path(args.labels))
    plot(rows, Path(args.output_prefix))
    print(f"tables={len(rows):,}")
    print(f"saved={Path(args.output_prefix)}.png/.pdf")


if __name__ == "__main__":
    main()
