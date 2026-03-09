#!/usr/bin/env python3
# Extract full records from step*_file.ndjson matching the given citation IDs.
# Usage (tag-only interface, consistent with other S2ORC scripts):
#     python s2orc_refcit_local.py --tag 251117 --src_dir /u501/z6dong/shared_data/se_citations_250218
# This will read IDs from hit_ids_<tag>.txt and write to full_hits_<tag>.jsonl.

import argparse, glob, os, sys, re

def iter_ids(ids_path: str):
    """Read one ID per line, return as a set[str] (string comparison is faster)."""
    with open(ids_path) as f:
        return {ln.strip() for ln in f if ln.strip()}

def pick_lines(step_file: str, wanted: set[str], out_fh):
    """Sequentially scan step_file and write lines matching citationid to out_fh."""
    # Precompiled regex to capture the first group of digits, efficient enough
    id_re = re.compile(r'"citationid"\s*:\s*(\d+)')
    with open(step_file, "r", encoding="utf-8") as fh:
        for ln in fh:
            m = id_re.search(ln)
            if not m:  # Each line is expected to have this field; this check adds robustness
                continue
            if m.group(1) in wanted:
                out_fh.write(ln)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--tag",
        required=True,
        help="Tag suffix (e.g. 251117). Controls both input IDs and output JSONL: "
             "hit_ids_<tag>.txt → full_hits_<tag>.jsonl",
    )
    ap.add_argument(
        "--src_dir",
        required=True,
        help="Directory containing step*_file.ndjson",
    )
    args = ap.parse_args()

    suffix = f"_{args.tag}"
    ids_path = f"hit_ids{suffix}.txt"
    out_path = f"full_hits{suffix}.jsonl"

    wanted = iter_ids(ids_path)
    if not wanted:
        print("⚠️  IDs file is empty. Exiting."); sys.exit(0)

    step_files = sorted(glob.glob(os.path.join(args.src_dir, "step*_file")))
    if not step_files:
        print("❌ No step*_file found."); sys.exit(1)

    print(f"🗃  Will scan {len(step_files)} files, extracting {len(wanted)} IDs")
    with open(out_path, "w", encoding="utf-8") as out_fh:
        for fp in step_files:
            pick_lines(fp, wanted, out_fh)
    print(f"✅  Done. Lines written to → {out_path}")

if __name__ == "__main__":
    main()

