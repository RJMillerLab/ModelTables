"""
Author: Zhengyuan Dong
Created: 2025-04-07
Last Modified: 2025-04-07
Description: Plot benchmark results for number of tables, columns, and average rows per table.
"""

from src.data_analysis.qc_stats import plot_metric
import pandas as pd
import numpy as np
import os

if __name__ == "__main__":
    OUTPUT_DIR = "data/analysis"
    results_path = os.path.join(OUTPUT_DIR, "benchmark_results.parquet")
    results_df = pd.read_parquet(results_path)
    # remove rows with WDC
    results_df = results_df[~results_df["Benchmark"].str.contains("WDC")]
    #results_df.to_parquet(results_path, index=False)
    print(results_df)
    plot_metric(results_df, "# Tables", "benchmark_tables.pdf")
    plot_metric(results_df, "# Cols", "benchmark_cols.pdf")
    plot_metric(results_df, "Avg # Rows", "benchmark_avg_rows.pdf")
    print(f"Saved figure to {OUTPUT_DIR}/benchmark_tables.pdf")
    print(f"Saved figure to {OUTPUT_DIR}/benchmark_cols.pdf")
    print(f"Saved figure to {OUTPUT_DIR}/benchmark_avg_rows.pdf")