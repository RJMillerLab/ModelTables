#!/usr/bin/env python3
"""
Author: Zhengyuan Dong
Created: 2025-03-09
Last Modified: 2025-03-14

Usage:
    Build the graph database:
        python build_mini_citation_kuzu.py --mode build --directory ./

Description:
    This script processes all files in the specified directory whose names match "step*_file".
    Each file is assumed to be NDJSON (one JSON object per line) or a JSON array in a file.

    Each line/object has the following fields:
        - citationid (UINT32)
        - citingcorpusid (UINT32)
        - citedcorpusid  (UINT32)
        - isinfluential (BOOL)
        - contexts (JSON array)
        - intents  (JSON array, e.g. list of lists)

    We create a minimal node table "Corpus" that only stores the id,
    and a relationship table "Cites" (from Corpus to Corpus) that stores the citation edge.

"""

import os
import glob
import argparse
import subprocess
from tqdm import tqdm
import kuzu

DB_PATH = "./demo_graph_db"

def debug_file(file_path):
    try:
        file_result = subprocess.run(["file", file_path], capture_output=True, text=True)
        print(f"DEBUG: File type of {file_path}:\n{file_result.stdout.strip()}")
        head_result = subprocess.run(["head", "-n", "3", file_path], capture_output=True, text=True)
        print(f"DEBUG: First 3 lines of {file_path}:\n{head_result.stdout.strip()}")
    except Exception as e:
        print(f"DEBUG: Error running debug commands on {file_path}: {e}")

def prepare_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == '[':
            return file_path
        else:
            lines = [line.strip() for line in f if line.strip()]
            json_array_str = "[" + ",".join(lines) + "]"
            tmp_file = file_path + ".array.json"
            with open(tmp_file, 'w', encoding='utf-8') as out_f:
                out_f.write(json_array_str)
            print(f"DEBUG: Converted NDJSON to JSON array in {tmp_file}")
            return tmp_file

def create_graph_db():
    print('Creating Kùzu DB...')
    db = kuzu.Database(DB_PATH)
    conn = kuzu.Connection(db)

    conn.execute("INSTALL json;")
    conn.execute("LOAD EXTENSION json;")

    conn.execute("""
        CREATE NODE TABLE IF NOT EXISTS Corpus(
            id UINT32,
            PRIMARY KEY (id)
        );
    """)

    conn.execute("""
        CREATE REL TABLE IF NOT EXISTS Cites(
            FROM Corpus TO Corpus,
            citationid UINT32,
            isinfluential BOOL,
            contexts JSON,
            intents JSON
        );
    """)

    conn.close()
    db.close()
    print('Database and schema creation finished.\n')

def load_json_to_graph(directory):
    print('Loading JSON->Graph...')
    db = kuzu.Database(DB_PATH)
    conn = kuzu.Connection(db)
    conn.execute("LOAD EXTENSION json;")

    all_files = glob.glob(os.path.join(directory, "step*_file"))

    for file in tqdm(all_files, desc="Processing JSON files", position=0):
        print(f'\nProcessing file: {file}')
        prepared_file = prepare_file(file)

        conn.execute(f"""
            COPY Corpus FROM (
                LOAD FROM '{prepared_file}' (file_format='JSON')
                RETURN DISTINCT to_uint32(citingcorpusid) AS id
                UNION
                LOAD FROM '{prepared_file}' (file_format='JSON')
                RETURN DISTINCT to_uint32(citedcorpusid) AS id
            );
        """)

        conn.execute(f"""
            COPY Cites FROM (
                LOAD FROM '{prepared_file}' (file_format='JSON')
                RETURN to_uint32(citingcorpusid) AS from,
                       to_uint32(citedcorpusid)  AS to,
                       to_uint32(citationid)     AS citationid,
                       isinfluential,
                       contexts,
                       intents
            );
        """)

    conn.close()
    db.close()
    print('\nAll files processed. Finished load_json_to_graph().')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kùzu JSON-based citation graph loader (minimal nodes, focus on cites).")
    parser.add_argument("--directory", type=str, required=True,
                        help="Directory containing files matching 'step*_file' (NDJSON or JSON).")
    parser.add_argument("--mode", choices=["build", "query"], default="build",
                        help="build: load data into Kùzu; query: query a citation by id")
    parser.add_argument("--citationid", type=int, help="Citation ID to query (for query mode)")
    args = parser.parse_args()

    if args.mode == "build":
        create_graph_db()
        load_json_to_graph(args.directory)

    elif args.mode == "query":
        db = kuzu.Database(DB_PATH)
        conn = kuzu.Connection(db)
        query = f"""
            MATCH (a:Corpus)-[r:Cites {{citationid: {args.citationid}}}]->(b:Corpus)
            RETURN a.id AS citingcorpusid, b.id AS citedcorpusid, r.citationid AS citationid,
                   r.isinfluential AS isinfluential, r.contexts AS contexts, r.intents AS intents;
        """
        response = conn.execute(query)
        results = []
        while response.has_next():
            results.append(response.get_next())
        if results:
            print("Query result:", results[0])
        else:
            print(f"❌ Not found citationid: {args.citationid}")
        response.close()
        conn.close()
        db.close()
