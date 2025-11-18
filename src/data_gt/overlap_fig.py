import os
import argparse
import gzip
import pickle
import numpy as np
import matplotlib.pyplot as plt
import warnings
import pandas as pd
from src.utils import load_config

# === Functions ===
def load_and_summarize(path):
    """Load matrix data and summarize counts for each key."""
    with gzip.open(path, 'rb') as f:
        data = pickle.load(f)

    records = []
    for key, mat in data.items():
        orig = getattr(mat, 'data', np.array([], dtype=float))
        orig_count = len(orig)
        filt = orig[(orig > 0) & np.isfinite(orig)]
        filt_count = len(filt)
        records.append({
            'key': key,
            'original_count': orig_count,
            'filtered_count': filt_count
        })

    df = pd.DataFrame.from_records(records)
    #df = df.sort_values(by='filtered_count', ascending=False)
    return df

def load_overlap_data(path, keys):
    """Load matrices and extract non-zero, finite overlap score arrays."""
    with gzip.open(path, 'rb') as f:
        data = pickle.load(f)
    distributions = []
    for k in keys:
        if k not in data:
            print(f"Warning: key '{k}' not found in data. Skipping.")
            distributions.append(np.array([], dtype=float))
            continue
        orig = data[k].data
        print(f"[DEBUG] '{k}' original count = {len(orig)}")  # Debug original count
        # Filter out zeros, NaN, and infinite values
        arr = orig[orig > 0]
        arr = arr[np.isfinite(arr)]
        print(f"[DEBUG] '{k}' filtered positive finite count = {len(arr)}")  # Debug filtered count
        distributions.append(arr)
    return distributions

def plot_violin_by_mode(distributions, metrics, modes, colors, save_path):
    """Grouped violin plots (one subplot = one mode)."""
    warnings.filterwarnings("ignore", category=RuntimeWarning,
                            message="invalid value encountered")
    n_metrics, n_modes = len(metrics), len(modes)

    fig, axes = plt.subplots(
        1, n_modes, figsize=(12, 4.5),
        sharey=True, gridspec_kw={"wspace": 0.25}
    )

    width   = 0.8 / n_metrics
    offsets = np.linspace(-0.4 + width/2, 0.4 - width/2, n_metrics)

    for j, mode in enumerate(modes):
        ax = axes[j]

        # cosmetic
        ax.set_xticks([])
        ax.set_xlabel(mode.capitalize(), fontsize=18)
        if j == 0:
            ax.set_ylabel("Overlap Score (log scale)", fontsize=20)
        ax.set_yscale("log")
        ax.set_xlim(-0.55, 0.55)
        ax.set_axisbelow(True)
        ax.grid(axis="y", ls="--", alpha=0.3)
        for spine in ax.spines.values():
            spine.set_visible(False)

        # draw violins
        for i, metric in enumerate(metrics):
            idx  = j * n_metrics + i
            pos  = offsets[i]
            parts = ax.violinplot(
                distributions[idx],
                positions=[pos],
                widths=width,
                showmeans=True,
                showextrema=False
            )
            for pc in parts["bodies"]:
                pc.set_facecolor(colors[i])
                pc.set_edgecolor("none")
                pc.set_alpha(0.85)
            if 'cmeans' in parts:
                parts['cmeans'].set_color('#8b2e2e')
                parts['cmeans'].set_linewidth(1.5)

    # Legend
    handles = [plt.Line2D([0], [0], color=colors[i], lw=6)
               for i in range(n_metrics)]
    fig.legend(handles, metrics, loc="lower center",
               bbox_to_anchor=(0.5, -0.05), ncol=n_metrics, frameon=False, fontsize=14)

    plt.suptitle(
        "Violin plot of paper-reference overlap score distributions\n"
        "by metric and reference importance", fontsize=20
    )
    plt.tight_layout(rect=[0, 0.05, 1, 0.92])
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

# === Main execution ===
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot violin figures of overlap rates")
    parser.add_argument('--tag', dest='tag', default=None,
                        help='Tag suffix for versioning (e.g., 251117). Enables versioning mode.')
    parser.add_argument('--input', dest='input', default=None,
                        help='Path to modelcard_citation_all_matrices pkl.gz (default: auto-detect from tag)')
    parser.add_argument('--output', dest='output', default=None,
                        help='Path to output PDF (default: overlap_violin_by_mode_<tag>.pdf)')
    args = parser.parse_args()
    
    config = load_config('config.yaml')
    base_path = config.get('base_path', 'data')
    processed_base_path = os.path.join(base_path, 'processed')
    tag = args.tag
    suffix = f"_{tag}" if tag else ""
    
    # Determine input/output paths based on tag
    uploaded_path = args.input or os.path.join(processed_base_path, f"modelcard_citation_all_matrices{suffix}.pkl.gz")
    output_path = args.output or (f"overlap_violin_by_mode{suffix}.pdf" if tag else "overlap_violin_by_mode.pdf")
    
    print("📁 Paths in use:")
    print(f"   Input matrices:      {uploaded_path}")
    print(f"   Output PDF:          {output_path}")
    summary_df = load_and_summarize(uploaded_path)
    print(summary_df.to_string(index=False))

    keys = [
        'max_pr', 'jaccard', 'dice',
        'max_pr_influential', 'jaccard_influential', 'dice_influential',
        'max_pr_methodology_or_result', 'jaccard_methodology_or_result', 'dice_methodology_or_result',
        'max_pr_methodology_or_result_influential', 'jaccard_methodology_or_result_influential', 'dice_methodology_or_result_influential'
    ]
    metrics = ['max_pr', 'jaccard', 'dice']
    modes = ['overall', 'influential', 'intent', 'influential+\nintent']

    # Colors for each mode
    #colors = ['C0', 'C1', 'C2', 'C3']
    palette_baseline = ["#8b2e2e", "#b74a3c", "#d96e44", "#f29e4c", "#FFBE5F"]
    palette_resource = ["#486f90", "#4e8094", "#50a89d", "#a5d2bc"]
    colors = ["#b74a3c", "#d96e44", "#f29e4c"]
    # Load distributions
    dists = load_overlap_data(uploaded_path, keys)

    plot_violin_by_mode(dists, metrics, modes, colors, output_path)
    print(f'Plots saved: {output_path}')
