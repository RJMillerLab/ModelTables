#!/usr/bin/env python3
"""
Author: Zhengyuan Dong
Created: 2025-03-09
Last Modified: 2025-03-14 (Added test mode, enhanced query, minimal fields option, clean mode)
Updated: 2025-03-XX (Skip records where citingcorpusid or citedcorpusid is None, line-count-based tqdm)

Usage:
    Build the graph database (store full fields):
        python build_mini_citation_neo4j.py --mode build --directory ./ --fields full

    Build the graph database (store only citationid, citingcorpusid, citedcorpusid):
        python build_mini_citation_neo4j.py --mode build --directory ./ --fields minimal

    Clean the database (remove all nodes, relationships, constraints, indexes):
        python build_mini_citation_neo4j.py --mode clean

    Query a citation or node by id:
        python build_mini_citation_neo4j.py --mode query --citationid 248811336

    Test the graph database (check nodes and edges counts, sample values):
        python build_mini_citation_neo4j.py --mode test

Description:
    This script processes all files in the specified directory whose names match "step*_file".
    Each file is assumed to be NDJSON (one JSON object per line).

    Each JSON object has the following fields:
        - citationid (INT64)
        - citingcorpusid (INT64)
        - citedcorpusid  (INT64)
        - isinfluential (BOOL)
        - contexts (JSON array)
        - intents (JSON array, e.g. list of lists)

    We create a minimal node "Corpus" that only stores the id,
    and a relationship "CITES" (from Corpus to Corpus) that stores the citation edge.

    With --fields minimal, only the first three fields are stored:
        - citationid, citingcorpusid, citedcorpusid
    With --fields full, all fields are stored:
        - citationid, citingcorpusid, citedcorpusid, isinfluential, contexts, intents

    'clean' mode can remove all data, constraints, indexes for a fresh start.

    Updated to:
      - Skip any record with citingcorpusid or citedcorpusid == None
      - Use wc -l to get file line count, so we can show a line-based tqdm.
"""

import os
import glob
import argparse
import json
import subprocess  ########
from tqdm import tqdm
from neo4j import GraphDatabase

# ######## Configuration for Neo4j (password updated to 11111111) ########
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "11111111"

CHECKPOINT_FILE = "checkpoint.json"
BATCH_SIZE = 1000000

def load_checkpoint():
    """Load checkpoint file; return empty dict if not exists."""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        return {}

def save_checkpoint(cp):
    """Save checkpoint to file."""
    with open(CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
        json.dump(cp, f, indent=2)

def get_line_count(file_path):
    """
    Use 'wc -l' to quickly get line count for large files,
    without loading them fully into Python memory.
    """
    try:
        output = subprocess.check_output(["wc", "-l", file_path]).decode().strip()
        # e.g. "12345 /path/to/file"
        return int(output.split()[0])
    except Exception as e:
        print(f"⚠️ Failed to get line count via wc -l: {e}")
        # fallback: return None or 0
        return 0

def process_ndjson_in_batches(file_path, batch_size=BATCH_SIZE):
    """
    Read NDJSON line by line in batches.
    Yield each batch as (batch_index, [records]).
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
                    print(f"⚠️ Skipping invalid JSON line in {file_path}: {e}")
                if len(batch) >= batch_size:
                    yield batch_index, batch
                    batch = []
                    batch_index += 1
        if batch:
            yield batch_index, batch

def create_constraints():
    """Create unique constraint on Corpus id to prevent duplicate nodes."""
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session() as session:
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Corpus) REQUIRE c.id IS UNIQUE;")
    driver.close()
    print("Constraints created.")

def import_batch(tx, batch_records, fields_mode):
    """
    Process a batch of records using UNWIND.
    Each record is expected to have:
      - citationid, citingcorpusid, citedcorpusid
      - (optionally) isinfluential, contexts, intents.

    If fields_mode == 'minimal', only store the first 3 fields.
    If fields_mode == 'full', also store isinfluential, contexts, intents.

    Skip any record where citingcorpusid or citedcorpusid is None.
    """

    if fields_mode == "minimal":
        query_minimal = """
        UNWIND $batch AS rec
        MERGE (a:Corpus {id: toInteger(rec.citingcorpusid)})
        MERGE (b:Corpus {id: toInteger(rec.citedcorpusid)})
        MERGE (a)-[r:CITES {citationid: toInteger(rec.citationid)}]->(b)
        """
        minimal_batch = []
        for rec in batch_records:
            citing_id = rec.get("citingcorpusid")
            cited_id = rec.get("citedcorpusid")
            if citing_id is None or cited_id is None:
                # skip
                continue
            minimal_batch.append({
                "citationid": rec.get("citationid"),
                "citingcorpusid": citing_id,
                "citedcorpusid": cited_id
            })
        if minimal_batch:
            tx.run(query_minimal, batch=minimal_batch)

    else:
        # 'full' mode
        query_full = """
        UNWIND $batch AS rec
        MERGE (a:Corpus {id: toInteger(rec.citingcorpusid)})
        MERGE (b:Corpus {id: toInteger(rec.citedcorpusid)})
        MERGE (a)-[r:CITES {citationid: toInteger(rec.citationid)}]->(b)
        ON CREATE SET r.isinfluential = rec.isinfluential,
                      r.contexts = rec.contexts,
                      r.intents = rec.intents
        ON MATCH SET r.isinfluential = rec.isinfluential,
                     r.contexts = rec.contexts,
                     r.intents = rec.intents
        """
        transformed_batch = []
        for rec in batch_records:
            citing_id = rec.get("citingcorpusid")
            cited_id = rec.get("citedcorpusid")
            if citing_id is None or cited_id is None:
                continue

            rec_copy = rec.copy()
            if rec_copy.get("contexts") is not None:
                rec_copy["contexts"] = json.dumps(rec_copy["contexts"])
            if rec_copy.get("intents") is not None:
                rec_copy["intents"] = json.dumps(rec_copy["intents"])

            transformed_batch.append(rec_copy)

        if transformed_batch:
            tx.run(query_full, batch=transformed_batch)

def load_json_to_graph(directory, fields_mode):
    """
    Process all files in the given directory that match "step*_file" (NDJSON).
    For each file, first get line count via 'wc -l' for tqdm,
    then read in batches, import to Neo4j.
    """
    checkpoint = load_checkpoint()
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    files = glob.glob(os.path.join(directory, "step*_file"))
    print(f"Found {len(files)} files to process.")

    for file_path in files:
        file_checkpoint = checkpoint.get(file_path, -1)
        print(f"Processing file: {file_path}")

        # 1) Get line count for tqdm
        line_count = get_line_count(file_path)
        pbar = tqdm(total=line_count, desc=os.path.basename(file_path), leave=False)

        # 2) Read & process in batches
        for batch_index, batch_records in process_ndjson_in_batches(file_path):
            if batch_index <= file_checkpoint:
                # skip
                pbar.update(len(batch_records))
                continue

            with driver.session() as session:
                session.execute_write(import_batch, batch_records, fields_mode)

            # update progress
            pbar.update(len(batch_records))
            print(f"[Batch {batch_index}] Processed {len(batch_records)} records.")
            checkpoint[file_path] = batch_index
            save_checkpoint(checkpoint)

        pbar.close()

    driver.close()
    print("✅ All files processed. Finished load_json_to_graph().")

def query_citation(citationid):
    """
    Query by given id: if it matches an edge's citationid or a node's id,
    return the associated relationships.
    """
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session() as session:
        query = """
        MATCH (a:Corpus)-[r:CITES]->(b:Corpus)
        WHERE r.citationid = toInteger($id)
           OR a.id = toInteger($id)
           OR b.id = toInteger($id)
        RETURN a.id AS citingcorpusid, b.id AS citedcorpusid, r.citationid AS citationid,
               r.isinfluential AS isinfluential, r.contexts AS contexts, r.intents AS intents
        """
        result = session.run(query, id=citationid)
        records = list(result)
        if records:
            for rec in records:
                print("Query result:", dict(rec))
        else:
            print(f"❌ Not found id: {citationid}")
    driver.close()

def test_graph():
    """
    Print node and edge counts, plus sample nodes and edges.
    """
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session() as session:
        node_count = session.run("MATCH (n:Corpus) RETURN count(n) AS c").single()["c"]
        edge_count = session.run("MATCH ()-[r:CITES]->() RETURN count(r) AS c").single()["c"]
        print("Total Corpus nodes:", node_count)
        print("Total CITES relationships:", edge_count)

        print("Sample nodes:")
        for rec in session.run("MATCH (n:Corpus) RETURN n LIMIT 5"):
            node = rec["n"]
            print(dict(node))

        print("Sample relationships:")
        for rec in session.run("MATCH (a:Corpus)-[r:CITES]->(b:Corpus) RETURN a.id AS a_id, b.id AS b_id, r LIMIT 5"):
            a_id = rec["a_id"]
            b_id = rec["b_id"]
            r_props = dict(rec["r"])
            print(f"{a_id} -> {b_id}, props: {r_props}")
    driver.close()

def clean_database():
    """
    Clean the entire database:
      - Remove all nodes and relationships
      - Drop all constraints
      - Drop all indexes
    """
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session() as session:
        # 1) remove all
        session.run("MATCH (n) DETACH DELETE n")
        print("All nodes and relationships have been deleted.")

        # 2) remove constraints
        constraints = session.run("SHOW CONSTRAINTS")
        for record in constraints:
            c_name = record["name"]
            session.run(f"DROP CONSTRAINT {c_name}")
        print("All constraints have been dropped.")

        # 3) remove indexes
        indexes = session.run("SHOW INDEXES")
        for record in indexes:
            i_name = record["name"]
            session.run(f"DROP INDEX {i_name}")
        print("All indexes have been dropped.")
    driver.close()
    print("Database is now completely empty.")

def main():
    parser = argparse.ArgumentParser(description="Neo4j JSON-based citation graph loader + wc-l line-based tqdm + skip null nodes.")
    parser.add_argument("--directory", type=str, help="Directory with 'step*_file' NDJSON", required=False)
    parser.add_argument("--mode", choices=["build", "query", "test", "clean"], default="build",
                        help="build: load data; query: find edges; test: check stats; clean: remove data")
    parser.add_argument("--citationid", type=int, help="For query mode, ID to look up")
    parser.add_argument("--fields", choices=["full", "minimal"], default="minimal",
                        help="Store full fields or minimal 3 fields only")
    args = parser.parse_args()

    if args.mode == "build":
        if not args.directory:
            print("Error: --directory is required in build mode.")
            return
        create_constraints()
        load_json_to_graph(args.directory, args.fields)

    elif args.mode == "query":
        if args.citationid is None:
            print("Error: --citationid is required in query mode.")
            return
        query_citation(args.citationid)

    elif args.mode == "test":
        test_graph()

    elif args.mode == "clean":
        clean_database()

if __name__ == "__main__":
    main()
