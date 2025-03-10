"""
Author: Zhengyuan Dong
Created: 2025-03-09
Last Modified: 2025-03-09

python build_mini_citation.py build --directory ./
python build_mini_citation.py query --citationid 169 --directory ./
"""
import os
import json
import glob
import sqlite3
import argparse
from tqdm import tqdm

DB_FILE = "citation_memory.db"

def create_memory_db():
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute('''
        CREATE TABLE IF NOT EXISTS citations (
            citationid INTEGER PRIMARY KEY,
            citingcorpusid INTEGER,
            citedcorpusid INTEGER
        )
    ''')
    conn.commit()
    conn.close()

def load_json_to_db(directory):
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    all_files = glob.glob(os.path.join(directory, 'step*_file'))
    for file in tqdm(all_files, desc="Loading JSON into SQLite", position=0):
        records = []
        with open(file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Processing lines...", position=1):  ######## 设置 position=1
                try:
                    citation = json.loads(line.strip())
                    records.append((
                        citation.get("citationid"),
                        citation.get("citingcorpusid"),
                        citation.get("citedcorpusid")
                    ))
                except json.JSONDecodeError:
                    print('skipping line...')
                    continue
        cur.executemany(
            "INSERT OR IGNORE INTO citations (citationid, citingcorpusid, citedcorpusid) VALUES (?, ?, ?)",
            records
        )
        conn.commit()
    conn.close()

def query_citation_by_id(citationid):
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("SELECT citationid, citingcorpusid, citedcorpusid FROM citations WHERE citationid = ?", (citationid,))
    result = cur.fetchone()
    conn.close()
    
    if result:
        citation_data = {
            "citationid": result[0],
            "citingcorpusid": result[1],
            "citedcorpusid": result[2]
        }
        print(json.dumps(citation_data, indent=2, ensure_ascii=False))
    else:
        print(f"❌ Not found citationid: {citationid}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    load_parser = subparsers.add_parser("build")
    load_parser.add_argument("--directory", type=str, required=True, help="")

    query_parser = subparsers.add_parser("query")
    query_parser.add_argument("--citationid", type=int, required=True, help="citationid")

    args = parser.parse_args()

    if args.mode == "build":
        create_memory_db()
        load_json_to_db(args.directory)
    elif args.mode == "query":
        query_citation_by_id(args.citationid)

