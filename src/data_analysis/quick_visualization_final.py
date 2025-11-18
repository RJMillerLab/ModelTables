#!/usr/bin/env python3
"""
Quick Visualization Script (Final Minimal Dot Version)
Author: Zhengyuan Dong
Modified: 2025-10-21
Description:
Paper-ready 2-row visualization — only key dots (max, total) are annotated,
no lines, all labels stay within plot frame, and layout matches qc_stats_fig style.
"""

import matplotlib.pyplot as plt
import numpy as np
import json
import os

# === Font settings (consistent with qc_stats_fig, no bold) ===
plt.rcParams.update({
    'font.size': 26,
    'axes.titlesize': 32,
    'axes.labelsize': 30,
    'xtick.labelsize': 19,
    'ytick.labelsize': 24,
    'legend.fontsize': 22,
    'figure.titlesize': 32
})


def create_combined_visualization():
    """Create final minimalist dot-based visualization."""
    print("Creating final minimalist paper visualization...")

    # --- Load data ---
    with open('step_by_step_data.json', 'r') as f:
        step_data = json.load(f)
    with open('table_frequency_data.json', 'r') as f:
        freq_data = json.load(f)

    # --- Figure setup ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), constrained_layout=True)

    # ==============================================================
    # (1) Step-by-step filtering visualization
    # ==============================================================
    categories = [
        "All Models",
        "Non-empty\nModel Cards",
        "Card Contains\nTables",
        "Card Contains Tables,\nw/ Valid Paper",
        "Card Contains Tables,\nw/ Enhanced Tables,\nw/ Valid Papers"
    ]
    counts = step_data['counts'][:-1]
    colors = step_data['colors'][:-1]

    x_positions = np.arange(len(categories)) * 1.25
    bars = ax1.bar(x_positions, counts, color=colors, alpha=0.85, width=0.8)

    ax1.set_title("Model Filtering Process", pad=25)
    ax1.set_ylabel("Model Count") # (Log Scale)
    ax1.set_xlabel("Filtering Conditions on Model")
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(categories, rotation=0, ha='center')
    ax1.set_yscale('log')
    ax1.margins(x=0.05)

    # Add numeric labels
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height * 1.15,
                 f"{count:,}", ha="center", va="bottom", fontsize=16)

    # ==============================================================
    # (2) Table Frequency Distribution Visualization
    # ==============================================================
    all_frequencies = freq_data['all_frequencies']
    avg_frequency = freq_data['avg_frequency']
    total_tables = freq_data['total_tables']
    max_frequency = max(all_frequencies)
    min_frequency = min(all_frequencies)

    n_points = len(all_frequencies)
    x_positions = np.arange(n_points)
    ax2.scatter(x_positions, all_frequencies,
                color="#ffb74d", s=25, alpha=0.75, edgecolors="none")

    # === Axis setup ===
    ax2.set_yscale("log")
    ax2.set_xlabel("Table Ranking")
    ax2.set_ylabel("Table Frequency") # (Log Scale)
    ax2.set_title("Table Frequency Distribution", pad=25)
    ax2.set_xlim(-n_points * 0.02, n_points * 1.02)
    ax2.set_ylim(min_frequency * 0.5, max_frequency * 1.6)
    ax2.grid(True, alpha=0.3)
    ax2.set_facecolor("#f8f9fa")

    # === Average frequency line ===
    ax2.axhline(y=avg_frequency, color="red", linestyle="--", linewidth=2, alpha=0.8, label="Average")
    ax2.text(n_points * 0.55, avg_frequency * 1.8,
             f"Average: {avg_frequency:.1f}",
             fontsize=16, color="red",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))

    # === Blue dot (Max count) ===
    ax2.scatter(0, max_frequency, s=100, color="blue", zorder=5, label="Max Table Frequency")
    ax2.text(n_points * 0.02, max_frequency * 1,
             f"Max Table Frequency: {int(max_frequency):,}",
             fontsize=16, color="blue",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))

    # === Green dot (Total table count) ===
    ax2.scatter(n_points - 1, all_frequencies[-1], s=100, color="green", zorder=5, label="Unique Table Count")
    ax2.text(n_points * 0.82, all_frequencies[-1] * 1.4,
             f"Unique Table Count: {int(total_tables):,}",
             fontsize=16, color="green",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))

    # === Legend inside frame ===
    #ax2.legend(loc="upper right", frameon=True, facecolor="white",
    #           edgecolor="#dddddd", fancybox=True, framealpha=0.9)

    # ==============================================================
    # Save Figure
    # ==============================================================
    os.makedirs("data/analysis", exist_ok=True)
    plt.savefig("data/analysis/quick_visualization_combined.pdf", dpi=300, bbox_inches="tight")
    plt.savefig("data/analysis/quick_visualization_combined.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("✅ Minimalist visualization saved:")
    print("• data/analysis/quick_visualization_combined.pdf")
    print("• data/analysis/quick_visualization_combined.png")


def main():
    print("=== Quick Visualization Generation ===")

    if not os.path.exists("step_by_step_data.json"):
        print("Error: step_by_step_data.json not found!")
        return
    if not os.path.exists("table_frequency_data.json"):
        print("Error: table_frequency_data.json not found!")
        return

    create_combined_visualization()
    print("\n=== Visualization Complete ===")


if __name__ == "__main__":
    main()
