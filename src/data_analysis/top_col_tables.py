#!/usr/bin/env python
"""List top-N tables with most columns from one or more DuckDB databases or folders containing them.
Usage:
    python top_col_tables.py data/modellake.db data/modellake_tr.db --top 500 --out tmp/top_tables.txt
The script prints "db_filename:table_name\t#cols" lines and also saves to --out if provided.
It also draws a histogram of the column counts with the same color setting.
"""
import argparse, os, duckdb, matplotlib.pyplot as plt

def get_tables_from_db(db_path):
    conn = duckdb.connect(db_path)
    # Get list of tables from current schema
    tables_info = conn.execute("PRAGMA show_tables;").fetchall()
    tables = []
    for row in tables_info:
        table_name = row[0]
        cols = conn.execute(f"PRAGMA table_info('{table_name}');").fetchall()
        col_count = len(cols)
        tables.append((col_count, f"{os.path.basename(db_path)}:{table_name}"))
    conn.close()
    return tables

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="DuckDB database file path(s) or folder path(s) containing .db files")
    ap.add_argument("--top", type=int, default=500)
    ap.add_argument("--out", default=None, help="Output txt path")
    args = ap.parse_args()

    all_tables = []
    # Process each provided path. If folder, add all .db files.
    for path in args.paths:
        if os.path.isdir(path):
            for fname in os.listdir(path):
                if fname.endswith(".db"):
                    db_file = os.path.join(path, fname)
                    all_tables.extend(get_tables_from_db(db_file))
        else:
            all_tables.extend(get_tables_from_db(path))

    # Sort tables based on column count descending
    all_tables.sort(key=lambda x: x[0], reverse=True)
    top_list = all_tables[:args.top]

    for cnt, name in top_list:
        print(f"{name}\t{cnt}")

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            for cnt, name in top_list:
                f.write(f"{name}\t{cnt}\n")
        print("Saved to", args.out)

    # Draw histogram of column counts for all tables with same color setting
    counts = [cnt for cnt, _ in all_tables]
    plt.hist(counts, bins=30, color='#1f77b4', edgecolor='black')
    plt.xlabel("Number of Columns")
    plt.ylabel("Frequency")
    plt.title("Histogram of Table Column Counts")
    plt.show()

if __name__ == "__main__":
    main()
