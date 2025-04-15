# -*- coding: utf-8 -*-
"""
Author: Zhengyuan Dong
Created: 2025-04-14
Last Modified: 2025-04-14
Description: This script is used to get the arXiv ID for the title extracted from the PDF file.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import load_npz, csr_matrix, csgraph
import pandas as pd
import os

def compute_graph_metrics(adj_matrix: csr_matrix, name: str, directed: bool = False) -> dict:
    """
    Computes a few basic graph metrics from a sparse adjacency matrix.
    Removes self-loops so they don't count as edges or contribute to degree.
    """
    adj_matrix = adj_matrix.copy()
    adj_matrix.setdiag(0)  # remove self-loops

    n_nodes = adj_matrix.shape[0]
    nnz = adj_matrix.nnz

    if directed:
        n_edges = nnz
    else:
        # For an undirected graph, each edge shows up twice in nnz
        n_edges = nnz // 2

    if n_nodes > 0:
        if directed:
            avg_degree = n_edges / n_nodes
        else:
            avg_degree = (2 * n_edges) / n_nodes
    else:
        avg_degree = 0

    if not directed:
        n_components, labels = csgraph.connected_components(
            adj_matrix, directed=False, return_labels=True
        )
    else:
        # Example: treat as weakly connected if directed
        n_components, labels = csgraph.connected_components(
            adj_matrix, directed=True, connection='weak', return_labels=True
        )
    comp_sizes = np.bincount(labels)
    largest_cc_size = comp_sizes.max() if len(comp_sizes) else 0

    return {
        "name": name,
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "avg_degree": avg_degree,
        "n_components": n_components,
        "largest_cc_size": largest_cc_size,
    }

def main():
    base_dir = "data/gt"
    suffix = "__overlap_rate"  # or "__direct_label", etc.

    # Adjacency files (adjust as needed)
    adjacency_files = {
        "paper_level": os.path.join(base_dir, f"paper_level_adjacency{suffix}.npz"),
        "model_level": os.path.join(base_dir, f"model_level_adjacency{suffix}.npz"),
        "csv_level":   os.path.join(base_dir, f"csv_level_adjacency{suffix}.npz"),
        "csv_symlink": os.path.join(base_dir, f"csv_symlink_adjacency{suffix}.npz"),
        "csv_real":    os.path.join(base_dir, f"csv_real_adjacency{suffix}.npz"),
    }

    # Decide if adjacency is directed or undirected:
    is_directed = False

    # 1) Load adjacency, compute metrics
    results = []
    adjacency_data = {}
    for name, path in adjacency_files.items():
        mat = load_npz(path)
        adjacency_data[name] = mat
        metrics = compute_graph_metrics(mat, name, directed=is_directed)
        results.append(metrics)

    # Put results into DataFrame for convenience
    df = pd.DataFrame(results).set_index("name")
    print("Summary of Graph Metrics:")
    print(df)

    # 2) Create one big figure with subplots
    #    We'll do 4 rows x 3 columns = 12 subplots total.
    #    That gives us enough room for:
    #      * 5 bar charts (first 2 rows)
    #      * 5 histograms (next 2 rows)
    #    and still have 2 subplots left blank or used for anything else you like.
    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(18, 24))
    axes = axes.ravel()  # flatten into a 1D list of Axes for easy indexing

    # 3) Bar charts in first 5 subplots (use however arrangement you like)
    #    We'll do:
    #      subplot[0] = #Nodes
    #      subplot[1] = #Edges
    #      subplot[2] = Average Degree
    #      subplot[3] = #Components
    #      subplot[4] = Largest CC size
    #    subplots[5..] can be used for histograms or remain empty

    metric_names = [
        ("n_nodes",        "Number of Nodes"),
        ("n_edges",        "Number of Edges"),
        ("avg_degree",     "Average Degree"),
        ("n_components",   "Number of Components"),
        ("largest_cc_size","Largest CC Size"),
    ]

    # For each metric, we make a bar chart
    for i, (col, title_str) in enumerate(metric_names):
        ax = axes[i]
        ax.bar(df.index, df[col])
        ax.set_title(f"{title_str} by Adjacency Type")
        ax.set_xlabel("Adjacency Type")
        ax.set_ylabel(title_str)
        # We can rotate x-axis labels if needed:
        ax.tick_params(axis='x', rotation=15)

    # 4) Degree distribution histograms in subplots[5..9]
    #    We'll go in alphabetical order, or the same order as in adjacency_files
    #    so that each adjacency gets one histogram.

    hist_index = 5
    for name in adjacency_files.keys():
        ax = axes[hist_index]
        mat = adjacency_data[name].copy()
        # remove self-loops for histogram
        mat.setdiag(0)

        degrees = np.array(mat.getnnz(axis=1))
        ax.hist(degrees, bins=50)
        ax.set_title(f"Degree Dist: {name}")
        ax.set_xlabel("Degree")
        ax.set_ylabel("Node Count")

        hist_index += 1
        if hist_index >= len(axes):
            print("Warning: ran out of subplot slots!")
            break

    # 5) If any subplots remain unused, you could hide them:
    for j in range(hist_index, len(axes)):
        fig.delaxes(axes[j])  # remove the unused subplot

    # 6) Adjust layout nicely
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
