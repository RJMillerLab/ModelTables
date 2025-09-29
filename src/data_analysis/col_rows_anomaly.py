#!/usr/bin/env python
"""Compare two folders of CSVs: histogram of per-table columns and rows (by basename).

Usage example:
    python src/data_analysis/top_col_tables.py \
             --v1-dir data/processed/deduped_hugging_csvs \
             --v2-dir data/processed/deduped_hugging_csvs_v2 \
             --recursive \

Notes:
    - Columns = number of fields in the header row (CSV-parsed)
    - Rows = number of lines in the file (including header)
    - Files are matched by basename between folders
    - The script saves a single overlay PNG with two subplots (columns and rows)
"""
import argparse
import os
import csv
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


def count_rows_fast(csv_path, chunk_size=8 * 1024 * 1024):
    """Count rows quickly by counting newlines in binary chunks.

    - Counts b"\n" occurrences across the file
    - If file is non-empty and does not end with a newline, adds 1
    """
    try:
        file_size = os.path.getsize(csv_path)
        if file_size == 0:
            return 0
        newline_count = 0
        last_byte_newline = False
        with open(csv_path, 'rb') as f:
            while True:
                data = f.read(chunk_size)
                if not data:
                    break
                newline_count += data.count(b'\n')
                last_byte_newline = data.endswith(b'\n')
        # If file doesn't end with a newline, there's one more line
        return newline_count if last_byte_newline else newline_count + 1
    except Exception:
        return 0


def count_columns_from_header_fast(csv_path, max_scan_bytes=8 * 1024 * 1024):
    """Read up to the first newline only and parse that header row with csv.reader.

    This avoids scanning the entire file for malformed quoting elsewhere.
    """
    try:
        header_bytes = bytearray()
        with open(csv_path, 'rb') as f:
            while True:
                # Read moderately sized chunks to find first newline quickly
                chunk = f.read(64 * 1024)
                if not chunk:
                    break
                nl_pos = chunk.find(b'\n')
                if nl_pos != -1:
                    header_bytes.extend(chunk[:nl_pos])
                    break
                header_bytes.extend(chunk)
                if len(header_bytes) >= max_scan_bytes:
                    break
        if not header_bytes:
            return 0
        header_str = header_bytes.decode('utf-8', errors='ignore')
        row = next(csv.reader([header_str]), None)
        return len(row) if row is not None else 0
    except Exception:
        return 0


def count_columns_from_header_ultra_simple(csv_path, max_scan_bytes=8 * 1024 * 1024):
    # Deprecated path kept for compatibility; not used in simplified mode.
    return count_columns_from_header_fast(csv_path, max_scan_bytes=max_scan_bytes)


def list_csv_files(paths, recursive=False):
    files = []
    for path in paths:
        if os.path.isdir(path):
            if recursive:
                for root, _, fnames in os.walk(path):
                    for fname in fnames:
                        if fname.lower().endswith('.csv'):
                            files.append(os.path.join(root, fname))
            else:
                for fname in os.listdir(path):
                    if fname.lower().endswith('.csv'):
                        files.append(os.path.join(path, fname))
    return files


def index_csv_basenames(directory, recursive=False):
    basenames = set()
    if not os.path.isdir(directory):
        return basenames
    if recursive:
        for root, _, fnames in os.walk(directory):
            for fname in fnames:
                if fname.lower().endswith('.csv'):
                    basenames.add(fname)
    else:
        for fname in os.listdir(directory):
            if fname.lower().endswith('.csv'):
                basenames.add(fname)
    return basenames


def count_columns_from_header(csv_path):
    """Count columns by reading only the first row with CSV parsing (handles quoted commas)."""
    try:
        with open(csv_path, 'r', newline='', encoding='utf-8', errors='ignore') as f:
            reader = csv.reader(f)
            first_row = next(reader, None)
            if first_row is None:
                return 0
            return len(first_row)
    except Exception:
        return 0


def count_rows_including_header(csv_path):
    """Count rows efficiently, including the header row."""
    try:
        count = 0
        with open(csv_path, 'rb') as f:
            for _ in f:
                count += 1
        return count
    except Exception:
        return 0

### updated ###
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1-dir", default="data/processed/deduped_hugging_csvs", help="Directory containing version 1 CSVs")
    ap.add_argument("--v2-dir", default="data/processed/deduped_hugging_csvs_v2", help="Directory containing version 2 CSVs")
    ap.add_argument("--recursive", action="store_true", help="Recurse into subdirectories of the provided directories")
    ap.add_argument("--out", default="tmp/v1v2_overlay.tsv", help="Optional output TSV with basename and v1/v2 columns/rows")
    ap.add_argument("--png-out", default="tmp/v1v2_overlay.png", help="Path to save the overlay histogram PNG")
    ap.add_argument("--bins", type=int, default=50, help="Number of bins for histograms")
    ap.add_argument("--workers", type=int, default=min(32, (os.cpu_count() or 4) * 2), help="Max worker threads for parallel file processing")
    ap.add_argument("--top", type=int, default=0, help="Print top-N examples where rows >> columns (by ratio)")
    ap.add_argument("--csv-out", default="tmp/v1v2_overlay.csv", help="Optional CSV file with per-basename stats and ratios")
    ap.add_argument("--anomalies-csv-out", default="tmp/v1v2_overlay_anomalies.csv", help="Optional CSV file to save anomaly cases only")
    ap.add_argument("--anomaly-ratio", type=float, default=100.0, help="Rows/cols ratio threshold to flag anomalies")
    ap.add_argument("--anomaly-min-rows", type=int, default=1000, help="Minimum rows to consider when flagging anomalies")
    args = ap.parse_args()

    # Compare v1 vs v2 with shared x-limits, always counting columns and rows
    v1_files = list_csv_files([args.v1_dir], recursive=args.recursive)
    if not v1_files:
        print("No CSV files found in v1_dir.")
        return
    v2_basenames = index_csv_basenames(args.v2_dir, recursive=args.recursive)
    v1_files = [p for p in v1_files if os.path.basename(p) in v2_basenames]
    if not v1_files:
        print("No overlapping CSV basenames between v1_dir and v2_dir.")
        return

    # Build v2 file map by basename
    v2_file_map = {}
    if args.recursive:
        for root, _, fnames in os.walk(args.v2_dir):
            for fname in fnames:
                if fname.lower().endswith('.csv'):
                    v2_file_map[fname] = os.path.join(root, fname)
    else:
        for fname in os.listdir(args.v2_dir):
            if fname.lower().endswith('.csv'):
                v2_file_map[fname] = os.path.join(args.v2_dir, fname)

    # Gather stats for v1 and v2
    v1_orig, v1_trans = [], []
    v2_orig, v2_trans = [], []
    col_counter = count_columns_from_header_fast

    # Process pairs in parallel
    pairs = [(v1_path, v2_file_map.get(os.path.basename(v1_path))) for v1_path in v1_files]
    pairs = [(p1, p2) for (p1, p2) in pairs if p2]
    base_to_paths = {os.path.basename(p1): (p1, p2) for (p1, p2) in pairs}
    results_rows = []  # (basename, v1_cols, v1_rows, v2_cols, v2_rows)
    def _proc_pair(v1_path, v2_path):
        v1_o = col_counter(v1_path)
        v2_o = col_counter(v2_path)
        v1_t = count_rows_fast(v1_path)
        v2_t = count_rows_fast(v2_path)
        return v1_o, v1_t, v2_o, v2_t, os.path.basename(v1_path)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_proc_pair, p1, p2): (p1, p2) for (p1, p2) in pairs}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing CSV pairs"):
            try:
                v1_o, v1_t, v2_o, v2_t, base = fut.result()
                v1_orig.append(v1_o)
                v1_trans.append(v1_t)
                v2_orig.append(v2_o)
                v2_trans.append(v2_t)
                results_rows.append((base, v1_o, v1_t, v2_o, v2_t))
            except Exception:
                pass

    # Compute x-limits separately for columns and rows for better resolution
    cols_combined = v1_orig + v2_orig
    rows_combined = v1_trans + v2_trans
    if not cols_combined and not rows_combined:
        print("No data to plot.")
        return
    col_x_min = min(cols_combined) if cols_combined else 0
    col_x_max = max(cols_combined) if cols_combined else 1
    row_x_min = min(rows_combined) if rows_combined else 0
    row_x_max = max(rows_combined) if rows_combined else 1

    # Plot overlay figure (two subplots): columns and rows
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    axc, axr = axes
    axc.hist(v1_orig, bins=args.bins, alpha=0.6, label='V1 columns', color='#1f77b4', edgecolor='black')
    axc.hist(v2_orig, bins=args.bins, alpha=0.6, label='V2 columns', color='#2ca02c', edgecolor='black')
    axc.set_xlabel("Count")
    axc.set_ylabel("Frequency")
    axc.set_title("Columns: V1 vs V2")
    axc.legend()

    axr.hist(v1_trans, bins=args.bins, alpha=0.6, label='V1 rows', color='#ff7f0e', edgecolor='black')
    axr.hist(v2_trans, bins=args.bins, alpha=0.6, label='V2 rows', color='#9467bd', edgecolor='black')
    axr.set_xlabel("Count")
    axr.set_title("Rows: V1 vs V2")
    axr.legend()

    axc.set_yscale('log')
    axr.set_yscale('log')

    fig.suptitle("Overlay: Original Columns and Transposed Rows")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    os.makedirs(os.path.dirname(args.png_out), exist_ok=True)
    fig.savefig(args.png_out, dpi=200)
    plt.close(fig)

    # Combined overlay histograms for V1 and V2
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    if v1_orig or v1_trans:
        ax1.hist(v1_orig, bins=args.bins, alpha=0.6, label='V1 columns', color='#1f77b4', edgecolor='black')
        ax1.hist(v1_trans, bins=args.bins, alpha=0.5, label='V1 rows', color='#ff7f0e', edgecolor='black')
        ax1.set_xlabel("Count")
        ax1.set_ylabel("Frequency")
        ax1.set_title("V1: Columns vs Rows")
        ax1.set_yscale('log')
        ax1.legend()
    if v2_orig or v2_trans:
        ax2.hist(v2_orig, bins=args.bins, alpha=0.6, label='V2 columns', color='#2ca02c', edgecolor='black')
        ax2.hist(v2_trans, bins=args.bins, alpha=0.5, label='V2 rows', color='#9467bd', edgecolor='black')
        ax2.set_xlabel("Count")
        ax2.set_ylabel("Frequency")
        ax2.set_title("V2: Columns vs Rows")
        ax2.set_yscale('log')
        ax2.legend()
    fig.tight_layout()
    combined_hist_path = "tmp/combined_hist.png"
    os.makedirs(os.path.dirname(combined_hist_path), exist_ok=True)
    fig.savefig(combined_hist_path, dpi=200)
    plt.close(fig)

    # Combined scatter plots: side-by-side V1 and V2 in one figure
    if results_rows:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        v1_x_rows, v1_y_cols = [], []
        v2_x_rows, v2_y_cols = [], []
        for base, v1_o, v1_t, v2_o, v2_t in results_rows:
            if v1_o is not None and v1_t is not None:
                v1_x_rows.append(v1_t)
                v1_y_cols.append(v1_o)
            if v2_o is not None and v2_t is not None:
                v2_x_rows.append(v2_t)
                v2_y_cols.append(v2_o)
        ax1.scatter(v1_x_rows, v1_y_cols, s=8, alpha=0.5, edgecolors='none', c='#1f77b4')
        ax1.set_xlabel("Rows (count)")
        ax1.set_ylabel("Columns (count)")
        ax1.set_title("V1: Rows vs Columns")
        ax1.grid(True, linestyle='--', alpha=0.3)
        ax2.scatter(v2_x_rows, v2_y_cols, s=8, alpha=0.5, edgecolors='none', c='#2ca02c')
        ax2.set_xlabel("Rows (count)")
        ax2.set_ylabel("Columns (count)")
        ax2.set_title("V2: Rows vs Columns")
        ax2.grid(True, linestyle='--', alpha=0.3)
        fig.suptitle("Combined Scatter Plots: V1 and V2")
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        combined_scatter_path = "tmp/combined_scatter.png"
        os.makedirs(os.path.dirname(combined_scatter_path), exist_ok=True)
        fig.savefig(combined_scatter_path, dpi=200)
        plt.close(fig)

    # Optional TSV output with per-basename counts
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, 'w', encoding='utf-8') as f:
            f.write("basename\tv1_cols\tv1_rows\tv2_cols\tv2_rows\n")
            for base, v1_o, v1_t, v2_o, v2_t in sorted(results_rows):
                f.write(f"{base}\t{v1_o}\t{v1_t}\t{v2_o}\t{v2_t}\n")

    # Optional CSV output with paths and ratios
    if args.csv_out:
        os.makedirs(os.path.dirname(args.csv_out), exist_ok=True)
        with open(args.csv_out, 'w', newline='', encoding='utf-8') as fcsv:
            writer = csv.writer(fcsv)
            writer.writerow([
                'basename',
                'v1_path', 'v1_cols', 'v1_rows', 'v1_rows_per_col',
                'v2_path', 'v2_cols', 'v2_rows', 'v2_rows_per_col'
            ])
            for base, v1_o, v1_t, v2_o, v2_t in sorted(results_rows):
                v1_path, v2_path = base_to_paths.get(base, (None, None))
                v1_rpc = (v1_t / v1_o) if v1_o else None
                v2_rpc = (v2_t / v2_o) if v2_o else None
                writer.writerow([
                    base,
                    v1_path, v1_o, v1_t, f"{v1_rpc:.6f}" if v1_rpc is not None else '',
                    v2_path, v2_o, v2_t, f"{v2_rpc:.6f}" if v2_rpc is not None else ''
                ])

    # Optional anomalies CSV filtered by thresholds
    if args.anomalies_csv_out:
        os.makedirs(os.path.dirname(args.anomalies_csv_out), exist_ok=True)
        with open(args.anomalies_csv_out, 'w', newline='', encoding='utf-8') as fcsv:
            writer = csv.writer(fcsv)
            writer.writerow(['version', 'basename', 'path', 'cols', 'rows', 'rows_per_col'])
            for base, v1_o, v1_t, v2_o, v2_t in results_rows:
                v1_path, v2_path = base_to_paths.get(base, (None, None))
                if v1_o and v1_t and v1_t >= args.anomaly_min_rows and (v1_t / v1_o) >= args.anomaly_ratio:
                    writer.writerow(['v1', base, v1_path, v1_o, v1_t, f"{(v1_t / v1_o):.6f}"])
                if v2_o and v2_t and v2_t >= args.anomaly_min_rows and (v2_t / v2_o) >= args.anomaly_ratio:
                    writer.writerow(['v2', base, v2_path, v2_o, v2_t, f"{(v2_t / v2_o):.6f}"])

    # Print top-N examples where rows are much larger than columns (per version)
    if args.top and args.top > 0:
        ratio_examples = []
        for base, v1_o, v1_t, v2_o, v2_t in results_rows:
            v1_path, v2_path = base_to_paths.get(base, (None, None))
            if v1_o and v1_o > 0 and v1_t is not None:
                ratio_examples.append((v1_t / max(1, v1_o), 'v1', base, v1_o, v1_t, v1_path))
            if v2_o and v2_o > 0 and v2_t is not None:
                ratio_examples.append((v2_t / max(1, v2_o), 'v2', base, v2_o, v2_t, v2_path))
        ratio_examples.sort(key=lambda x: x[0], reverse=True)
        print("Top examples (rows/cols ratio):")
        print("ratio\tversion\tbasename\tcols\trows\tpath")
        for ratio, ver, base, cols, rows, path in ratio_examples[:args.top]:
            print(f"{ratio:.2f}\t{ver}\t{base}\t{cols}\t{rows}\t{path}")

if __name__ == "__main__":
    main()
