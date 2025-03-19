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
        - intents (JSON array, e.g. list of lists)

    We create a minimal node table "Corpus" that only stores the id,
    and a relationship table "Cites" (from Corpus to Corpus) that stores the citation edge.
    This approach minimizes the storage on nodes while focusing on the citation edges.

    Requirements:
        - Kùzu is correctly installed and can be imported via `import kuzu`
        - The kuzu Python module is installed using pip
"""

import os
import glob
import argparse
import subprocess  # For optional debugging
import json  ######## checkpoint: used for saving progress
from tqdm import tqdm
import kuzu

TEMP_BATCH_FILE = "temp_batch.json"
DB_PATH = "./demo_graph_db"  # Database storage path ########
CHECKPOINT_FILE = "checkpoint.json"  ######## checkpoint: define checkpoint file

def load_checkpoint():
    """Load checkpoint file, return empty dict if not exists. ######## checkpoint"""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        # Initialize structure
        return {"nodes": {}, "edges": {}}

def save_checkpoint(cp):
    """Save checkpoint to file. ######## checkpoint"""
    with open(CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
        json.dump(cp, f, indent=2)

def debug_file(file_path):
    """
    Use Linux commands to check the file type and print the first three lines for debugging.
    Optional, call only when necessary.
    """
    try:
        file_result = subprocess.run(["file", file_path], capture_output=True, text=True)
        print(f"DEBUG: File type of {file_path}:\n{file_result.stdout.strip()}")
        head_result = subprocess.run(["head", "-n", "3", file_path], capture_output=True, text=True)
        print(f"DEBUG: First 3 lines of {file_path}:\n{head_result.stdout.strip()}")
    except Exception as e:
        print(f"DEBUG: Error running debug commands on {file_path}: {e}")

######## Changed: Removed temporary file generation; process NDJSON in memory ########
def process_ndjson_in_batches_memory(file_path, batch_size=10000):
    """
    Read NDJSON line by line in batches. Yield each batch as a list of dicts.
    """
    batch = []
    batch_index = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    record = json.loads(line)
                    batch.append(record)
                except Exception as e:
                    print("⚠️ Skipping invalid JSON line:", e)
                if len(batch) >= batch_size:
                    yield batch_index, batch
                    batch = []
                    batch_index += 1
        if batch:
            yield batch_index, batch

def write_temp_json(batch_records):
    """Write current batch to temp_batch.json (overwrites previous)"""
    with open(TEMP_BATCH_FILE, "w", encoding="utf-8") as fout:
        for rec in batch_records:
            fout.write(json.dumps(rec) + "\n")

def create_graph_db():
    print('Creating Kùzu DB...')
    db = kuzu.Database(DB_PATH)
    conn = kuzu.Connection(db)

    # Install and load JSON extension
    conn.execute("INSTALL json;")
    conn.execute("LOAD EXTENSION json;")

    # Create node table: Corpus, storing only id, changed to INT64
    result = conn.execute("""
        CREATE NODE TABLE IF NOT EXISTS Corpus(
            id INT64,
            PRIMARY KEY (id)
        );
    """)
    result.close()

    # Create relationship table: Cites, storing citation edge info, citationid changed to INT64
    result = conn.execute("""
        CREATE REL TABLE IF NOT EXISTS Cites(
            FROM Corpus TO Corpus,
            citationid INT64,
            isinfluential BOOL,
            contexts JSON,
            intents JSON
        );
    """)
    result.close()
    conn.close()
    db.close()
    print('Database and schema creation finished.\n')

def load_json_to_graph(directory):
    print('Loading JSON->Graph...')
    checkpoint = load_checkpoint()
    if "nodes" not in checkpoint:
        checkpoint["nodes"] = {}
    if "edges" not in checkpoint:
        checkpoint["edges"] = {}

    db = kuzu.Database(DB_PATH)
    conn = kuzu.Connection(db)
    conn.execute("LOAD EXTENSION json;")

    all_files = glob.glob(os.path.join(directory, "step*_file"))

    # Process nodes batch-wise
    for file in tqdm(all_files, desc="Processing node files"):
        file_cp = checkpoint["nodes"].get(file, -1)
        for batch_index, batch_records in tqdm(process_ndjson_in_batches_memory(file, batch_size=10000),
                                               desc=f"File {os.path.basename(file)} (nodes)", leave=False):
            if batch_index <= file_cp:
                continue

            # Extract unique node IDs from batch
            node_ids = set()
            for rec in batch_records:
                if "citingcorpusid" in rec and rec["citingcorpusid"] is not None:
                    node_ids.add(int(rec["citingcorpusid"]))
                if "citedcorpusid" in rec and rec["citedcorpusid"] is not None:
                    node_ids.add(int(rec["citedcorpusid"]))
            node_batch = [{"id": nid} for nid in node_ids]

            # Write to temp file
            write_temp_json(node_batch)
            query = f"COPY Corpus FROM '{TEMP_BATCH_FILE}' (file_format='JSON');"
            try:
                conn.execute(query)
                print(f"[Batch {batch_index}] Nodes loaded: {len(node_batch)}")
            except Exception as e:
                print(f"[Batch {batch_index}] ⚠️ Node load error:", e)

            checkpoint["nodes"][file] = batch_index
            save_checkpoint(checkpoint)

    # Process edges batch-wise
    for file in tqdm(all_files, desc="Processing edge files"):
        file_cp = checkpoint["edges"].get(file, -1)
        for batch_index, batch_records in tqdm(process_ndjson_in_batches_memory(file, batch_size=10000),
                                               desc=f"File {os.path.basename(file)} (edges)", leave=False):
            if batch_index <= file_cp:
                continue

            # Write edge records directly to temp
            write_temp_json(batch_records)
            query = (
                "COPY Cites FROM ("
                f"LOAD FROM '{TEMP_BATCH_FILE}' (file_format='JSON') "
                "RETURN to_int64(citingcorpusid) AS from, "
                "to_int64(citedcorpusid) AS to, "
                "to_int64(citationid) AS citationid, "
                "isinfluential, "
                "CAST(contexts AS JSON) AS contexts, "
                "CAST(intents AS JSON) AS intents"
                ");"
            )
            try:
                conn.execute(query)
                print(f"[Batch {batch_index}] Edges loaded: {len(batch_records)}")
            except Exception as e:
                print(f"[Batch {batch_index}] ⚠️ Edge load error:", e)

            checkpoint["edges"][file] = batch_index
            save_checkpoint(checkpoint)

    conn.close()
    db.close()
    print("✅ All files processed. Finished load_json_to_graph().")

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
