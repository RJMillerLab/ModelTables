# -*- coding: utf-8 -*-
"""
Extract s2orc title→paperId results from API query log files.
Merges multiple logs, deduplicates by query_title (prefer success > 404 > others).
Output: s2orc_titles2ids_{tag}_5.parquet (or custom output path).

Usage:
    PYTHONPATH=. python bak/s2orc_log_parser.py --tag 251117 --logdir logs
    PYTHONPATH=. python bak/s2orc_log_parser.py --tag 251117 --logs logs/s2orc_API_query_0421.log logs/s2orc_API_query_0504.log ...
"""
import re
import argparse
import glob
import os
import pandas as pd
from pathlib import Path


def parse_success_line(line):
    """Parse: ✅ For 'query_title': paperId=xxx, corpusId=xxx, retrieved_title='...'
    Uses re.search (not re.match) because tqdm may concatenate onto the same line."""
    m = re.search(r"✅ For '(.+?)': paperId=([^,\s]+), corpusId=([^,\s]+), retrieved_title='(.+)'", line)
    if not m:
        return None
    query_title = m.group(1).strip()
    paper_id = m.group(2)
    corpus_id = m.group(3)
    retrieved_title = m.group(4).rstrip("'").strip()  # trailing ' from pattern
    return {
        "query_title": query_title,
        "retrieved_title": retrieved_title,
        "paperId": paper_id,
        "corpusId": corpus_id,
        "paper_identifier": f"CorpusID:{corpus_id}" if corpus_id else paper_id,
        "query_status": "success",
    }


def parse_error_line(line):
    """Parse: ❌ HTTP error 404 while searching for: query_title"""
    m = re.search(r"❌ HTTP error (\d+) while searching for: (.+)", line)
    if not m:
        return None
    status_code = m.group(1)
    query_title = m.group(2).strip()
    if "Processing Titles" in query_title:  # tqdm concatenation
        query_title = query_title.split("Processing Titles")[0].strip()
    status = "404" if status_code == "404" else f"http_{status_code}"
    return {
        "query_title": query_title,
        "retrieved_title": None,
        "paperId": None,
        "corpusId": None,
        "paper_identifier": None,
        "query_status": status,
    }


def parse_log_file(path):
    """Parse a single log file, yield records."""
    records = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\r\n")
            if "✅ For '" in line and "paperId=" in line:
                rec = parse_success_line(line)
                if rec:
                    records.append(rec)
            elif "❌ HTTP error" in line and "while searching for:" in line:
                rec = parse_error_line(line)
                if rec:
                    records.append(rec)
    return records


def merge_records(records_list):
    """Deduplicate by query_title: prefer success > 404 > others."""
    status_rank = {"success": 0, "404": 1, "429": 2, "http_500": 3, "timeout": 4, "no_results": 5}
    by_title = {}
    for rec in records_list:
        q = rec["query_title"]
        curr = by_title.get(q)
        rank_curr = status_rank.get(curr["query_status"], 99) if curr else 99
        rank_new = status_rank.get(rec["query_status"], 99)
        if curr is None or rank_new < rank_curr:
            by_title[q] = rec
    return list(by_title.values())


def main():
    parser = argparse.ArgumentParser(description="Extract s2orc results from API query logs")
    parser.add_argument("--tag", default="251117", help="Tag for output file")
    parser.add_argument("--logdir", default="logs", help="Directory containing log files")
    parser.add_argument("--logs", nargs="*", help="Explicit list of log files (overrides --logdir)")
    parser.add_argument("--pattern", default="s2orc_API_query*.log", help="Glob pattern in logdir")
    parser.add_argument("--output", default=None, help="Output parquet path (default: data/processed/s2orc_titles2ids_{tag}_5.parquet)")
    parser.add_argument("--exclude", nargs="*", default=["backup", "bakup", "bak"], help="Exclude log paths containing these strings")
    parser.add_argument("--extra_logdirs", nargs="*", help="Extra log dirs to scan (e.g. ~/Repo/ModelTables/logs)")
    args = parser.parse_args()

    if args.logs:
        log_paths = [p for p in args.logs if os.path.exists(p)]
    else:
        log_paths = sorted(glob.glob(os.path.join(args.logdir, args.pattern)))
        if args.extra_logdirs:
            for d in args.extra_logdirs:
                d = os.path.expanduser(d)
                log_paths.extend(glob.glob(os.path.join(d, args.pattern)))
        log_paths = sorted(set(p for p in log_paths if os.path.exists(p)))
        log_paths = [p for p in log_paths if not any(ex in p.lower() for ex in args.exclude)]

    if not log_paths:
        print("No log files found.")
        return

    print(f"Parsing {len(log_paths)} log files...")
    all_records = []
    for path in log_paths:
        recs = parse_log_file(path)
        all_records.extend(recs)
        if recs:
            print(f"  {path}: {len(recs)} records")

    if not all_records:
        print("No records extracted.")
        return

    merged = merge_records(all_records)
    print(f"Total before dedup: {len(all_records)}, after dedup: {len(merged)}")

    df = pd.DataFrame(merged)
    output = args.output or os.path.join("data", "processed", f"s2orc_titles2ids_{args.tag}_5.parquet")
    os.makedirs(os.path.dirname(output), exist_ok=True)
    df.to_parquet(output, index=False)
    print(f"Saved {len(df)} rows to {output}")
    if "query_status" in df.columns:
        print("Status breakdown:", df["query_status"].value_counts().to_dict())


if __name__ == "__main__":
    main()
