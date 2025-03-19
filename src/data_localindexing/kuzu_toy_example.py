#!/usr/bin/env python3
"""
Toy example: Build a graph from edge NDJSON file (with deduplication and debug info)

Steps:
1. Create a node table `Corpus` (primary key `id`) and a relationship table `Cites`.
2. Generate an edge data file (`step1_file`), each line represents an edge with fields:
   citationid, citingcorpusid, citedcorpusid, isinfluential, contexts, intents.
3. Extract unique citing and cited node IDs from edge data and write to temporary file `temp_nodes.json`.
4. Load node file into `Corpus` table via COPY FROM.
5. Load edge data into `Cites` table directly from `step1_file` using COPY FROM with mapping.

Note: Ensure Kùzu and its JSON extension are correctly installed and DB_PATH is valid.
"""

import os
import json
import kuzu

DB_PATH = "./demo_graph_db"
EDGE_FILE = "step1_file"              # Edge data file (NDJSON format)
TEMP_NODES_FILE = "temp_nodes.json"   # Temporary file for unique nodes

def create_schema():
    """Create node table `Corpus` and relationship table `Cites`"""
    db = kuzu.Database(DB_PATH)
    conn = kuzu.Connection(db)
    conn.execute("INSTALL json;")
    conn.execute("LOAD EXTENSION json;")
    conn.execute("""
        CREATE NODE TABLE IF NOT EXISTS Corpus(
            id INT64,
            PRIMARY KEY (id)
        );
    """)
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
    print("Schema created.")

def create_edge_file():
    """Generate a sample edge data file (NDJSON), each line is one edge"""
    with open(EDGE_FILE, "w", encoding="utf-8") as f:
        # Sample edges: 456 -> 789, 457 -> 790, and a duplicate of the first
        f.write('{"citationid":123,"citingcorpusid":456,"citedcorpusid":789,"isinfluential":false,"contexts":["test1"],"intents":[["background"], null]}\n')
        f.write('{"citationid":124,"citingcorpusid":457,"citedcorpusid":790,"isinfluential":true,"contexts":["test2"],"intents":[["method"], null]}\n')
        f.write('{"citationid":123,"citingcorpusid":456,"citedcorpusid":789,"isinfluential":false,"contexts":["test1"],"intents":[["background"], null]}\n')
    print(f"Edge file '{EDGE_FILE}' created.")

def extract_nodes_to_file():
    """
    Scan edge data and extract unique citing and cited node IDs.
    Write each node to TEMP_NODES_FILE as {"id": <node_id>}
    """
    nodes = set()
    with open(EDGE_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                if "citingcorpusid" in record and record["citingcorpusid"] is not None:
                    nodes.add(int(record["citingcorpusid"]))
                if "citedcorpusid" in record and record["citedcorpusid"] is not None:
                    nodes.add(int(record["citedcorpusid"]))
            except Exception as e:
                print("Error parsing line:", e)
    with open(TEMP_NODES_FILE, "w", encoding="utf-8") as fout:
        for nid in sorted(nodes):
            fout.write(json.dumps({"id": nid}) + "\n")
    print(f"Temporary nodes file '{TEMP_NODES_FILE}' created with {len(nodes)} nodes.")

def load_nodes():
    """Load TEMP_NODES_FILE into `Corpus` table via COPY FROM"""
    db = kuzu.Database(DB_PATH)
    conn = kuzu.Connection(db)
    conn.execute("LOAD EXTENSION json;")
    query = f"COPY Corpus FROM '{TEMP_NODES_FILE}' (file_format='JSON');"
    query = query.strip()
    try:
        conn.execute(query)
        print("Nodes loaded successfully!")
    except Exception as e:
        print("Error executing nodes COPY query:", e)
    conn.close()
    db.close()

def load_edges(edges_file):
    """
    Load edge data from edges_file into `Cites` table.
    Map citingcorpusid to `from`, citedcorpusid to `to`, and include edge properties.
    """
    db = kuzu.Database(DB_PATH)
    conn = kuzu.Connection(db)
    conn.execute("LOAD EXTENSION json;")
    query = (
        "COPY Cites FROM ("
        f"LOAD FROM '{edges_file}' (file_format='JSON') "
        "RETURN to_int64(citingcorpusid) AS from, "
        "to_int64(citedcorpusid) AS to, "
        "to_int64(citationid) AS citationid, "
        "isinfluential, "
        "CAST(contexts AS JSON) AS contexts, "
        "CAST(intents AS JSON) AS intents"
        ");"
    )
    query = query.strip()
    try:
        conn.execute(query)
        print("Edges loaded successfully!")
    except Exception as e:
        print("Error executing edges COPY query:", e)
    conn.close()
    db.close()

def main():
    create_schema()
    # create_edge_file()  # Uncomment if sample data is needed
    extract_nodes_to_file()
    load_nodes()
    load_edges(EDGE_FILE)

if __name__ == "__main__":
    main()

