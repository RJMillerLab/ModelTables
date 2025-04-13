"""
Author: Zhengyuan Dong
Created: 2025-04-12
Last Modified: 2025-04-12
Description: This script extracts error titles from a log file and saves them in either TXT or JSON format.
Usage:
    python -m src.data_preprocess.s2orc_log_429 --logfile logs/s2orc_API_query.log --outformat json --outfile data/processed/modelcard_dedup_titles_429.json --error 429
"""

import json
import argparse

def extract_error_titles(log_file, error_filter=None):
    error_titles = []
    with open(log_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    # Modify logic: iterate through each line and parse lines starting with "❌ HTTP error"   ########
    for line in lines:  ######## Change: use a for loop instead of an index-based while loop
        line = line.strip()   ########
        if line.startswith("❌ HTTP error"):   ########
            # If error_filter is specified, only process the line if it contains the error_filter       
            if error_filter and error_filter not in line:  ########
                continue   ########
            # The error line format is similar to:
            # "❌ HTTP error 404 while searching for: internlm-xcomposer/examples/screenshot-to-webpage.html at main · internlm/internlm-xcomposer · github"
            # Split the line by "while searching for:" to extract the title portion
            parts = line.split("while searching for:")   ########
            if len(parts) >= 2:   ########
                title = parts[1].strip()   ########
                error_titles.append(title)   ########
    return error_titles

def save_as_txt(titles, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for title in titles:
            f.write(title + "\n")

def save_as_json(titles, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(titles, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract error titles from a log file")
    parser.add_argument("--logfile", help="Path to the log file")
    parser.add_argument("--outformat", choices=["txt", "json"], help="Output format: txt or json")
    parser.add_argument("--outfile", help="Path to the output file")
    parser.add_argument("--error", type=str, default="", help="Error filter string, e.g. '429' or '404'")
    args = parser.parse_args()

    titles = extract_error_titles(args.logfile, error_filter=args.error if args.error else None)
    if args.outformat == "txt":
        save_as_txt(titles, args.outfile)
    elif args.outformat == "json":
        save_as_json(titles, args.outfile)
