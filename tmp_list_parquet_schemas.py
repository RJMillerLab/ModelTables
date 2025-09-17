from pathlib import Path
import pyarrow.parquet as pq
import sys

def human(n: float) -> str:
    units = ['B','KB','MB','GB','TB']
    idx = 0
    n = float(n)
    while n >= 1024 and idx < 4:
        n /= 1024
        idx += 1
    return f"{n:.1f}{units[idx]}"

base = Path('data/processed')
files = sorted(base.rglob('*.parquet'), key=lambda p: p.stat().st_size, reverse=True)
for p in files:
    try:
        pf = pq.ParquetFile(str(p))
        schema = pf.schema_arrow
        nrows = pf.metadata.num_rows
        size = p.stat().st_size
        cols = [f"{f.name}:{f.type}" for f in schema]
        print(f"FILE: {p} | SIZE: {human(size)} | ROWS: {nrows} | COLS: {len(cols)}")
        for c in cols:
            print(f"  {c}")
    except Exception as e:
        print(f"FILE: {p} | ERROR: {e}")
