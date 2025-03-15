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
        - citationid (INT64)  ######## changed to INT64 ########
        - citingcorpusid (INT64) ######## changed to INT64 ########
        - citedcorpusid  (INT64) ######## changed to INT64 ########
        - isinfluential (BOOL)
        - contexts (JSON array)
        - intents  (JSON array, e.g. list of lists)

    We create a minimal node table "Corpus" that only stores the id,
    and a relationship table "Cites" (from Corpus to Corpus) that stores the citation edge.
    This approach minimizes the storage on nodes while focusing on the citation edges.

    Requirements:
        - Kùzu is correctly installed and can be imported via `import kuzu` ########
        - The kuzu Python module is installed using pip ########
"""

import os
import glob
import argparse
import subprocess  # For optional debugging
from tqdm import tqdm
import kuzu

DB_PATH = "./demo_graph_db"  # Database storage path ########

def debug_file(file_path):
    """
    Use Linux commands to check the file type and print the first three lines for debugging.
    Optional, call only when necessary. ########
    """
    try:
        file_result = subprocess.run(["file", file_path], capture_output=True, text=True)
        print(f"DEBUG: File type of {file_path}:\n{file_result.stdout.strip()}")
        head_result = subprocess.run(["head", "-n", "3", file_path], capture_output=True, text=True)
        print(f"DEBUG: First 3 lines of {file_path}:\n{head_result.stdout.strip()}")
    except Exception as e:
        print(f"DEBUG: Error running debug commands on {file_path}: {e}")

def prepare_file(file_path):
    """
    If the file is in NDJSON format (one JSON object per line), convert it to a JSON array and return the new file path;
    if the file is already in JSON array format, return the original path. ########
    """
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

    # Install and load JSON extension ########
    conn.execute("INSTALL json;")
    conn.execute("LOAD EXTENSION json;")

    # Create node table: Corpus, storing only id, changed to INT64 ########
    conn.execute("""
        CREATE NODE TABLE IF NOT EXISTS Corpus(
            id INT64,
            PRIMARY KEY (id)
        );
    """)

    # Create relationship table: Cites, storing citation edge info, citationid changed to INT64 ########
    conn.execute("""
        CREATE REL TABLE IF NOT EXISTS Cites(
            FROM Corpus TO Corpus,
            citationid INT64,
            isinfluential BOOL,
            contexts JSON,
            intents JSON
        );
    """)

    conn.close()
    db.close()
    print('Database and schema creation finished.\n')

def load_json_to_graph(directory):
    """
    Read all data files in the specified directory matching "step*_file".
    First, load the deduplicated node ids from all files into the Corpus table,
    then load the edge data one file at a time into the Cites table. ########
    """
    print('Loading JSON->Graph...')
    db = kuzu.Database(DB_PATH)
    conn = kuzu.Connection(db)
    conn.execute("LOAD EXTENSION json;")

    all_files = glob.glob(os.path.join(directory, "step*_file"))
    
    # Construct a union query for nodes from all files to ensure global deduplication ########
    union_queries = []
    for file in all_files:
        union_queries.append(
            f"LOAD FROM '{file}' (file_format='JSON') WHERE citingcorpusid IS NOT NULL RETURN DISTINCT to_int64(citingcorpusid) AS id" ########
        )
        union_queries.append(
            f"LOAD FROM '{file}' (file_format='JSON') WHERE citedcorpusid IS NOT NULL RETURN DISTINCT to_int64(citedcorpusid) AS id" ########
        )
    union_query_str = " UNION ".join(union_queries)  ########
    node_load_query = f"COPY Corpus FROM ({union_query_str});"  ########
    
    print("Loading nodes into Corpus...")
    conn.execute(node_load_query)

    # Load edge data into Cites table from each file individually ########
    for file in all_files:
        print(f'\nProcessing file: {file}')
        print('Loading edges to Cites table...')
        edge_query = f"""
            COPY Cites FROM (
                LOAD FROM '{file}' (file_format='JSON')
                WHERE citingcorpusid IS NOT NULL AND citedcorpusid IS NOT NULL
                RETURN to_int64(citingcorpusid) AS from,
                       to_int64(citedcorpusid)  AS to,
                       to_int64(citationid)     AS citationid,
                       isinfluential,
                       json_parse(contexts) AS contexts,
                       json_parse(intents)  AS intents
            );
        """
        conn.execute(edge_query)
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
