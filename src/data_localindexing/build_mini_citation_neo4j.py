#!/usr/bin/env python3

"""
Example script to load citation data into Neo4j in one pass, without creating separate node and edge phases.
Requirements:
    pip install neo4j tqdm
Usage:
    python build_mini_citation_neo4j.py --directory ./ --user neo4j --password 11111111
"""
import os
import json
from neo4j import GraphDatabase

uri = "bolt://localhost:7687"
username = "neo4j"
password = "11111111"

driver = GraphDatabase.driver(uri, auth=(username, password))

def import_citation(tx, record):
    citationid = record.get("citationid")
    citingcorpusid = record.get("citingcorpusid")
    citedcorpusid = record.get("citedcorpusid")
    isinfluential = record.get("isinfluential")
    contexts = record.get("contexts")
    intents = record.get("intents")
    
    query = """
    // MERGE citing corpus node
    MERGE (citing:Corpus {id: $citingcorpusid})
      ON CREATE SET citing.created = timestamp()
    // MERGE cited corpus node
    MERGE (cited:Corpus {id: $citedcorpusid})
      ON CREATE SET cited.created = timestamp()
    // Use citationid as the unique merge relation
    MERGE (citing)-[r:CITES {citationid: $citationid}]->(cited)
      ON CREATE SET r.isinfluential = $isinfluential,
                    r.contexts = $contexts,
                    r.intents = $intents
      ON MATCH SET r.isinfluential = $isinfluential,
                   r.contexts = $contexts,
                   r.intents = $intents
    RETURN citing, r, cited
    """
    tx.run(query,
           citationid=citationid,
           citingcorpusid=citingcorpusid,
           citedcorpusid=citedcorpusid,
           isinfluential=isinfluential,
           contexts=contexts,
           intents=intents)

def process_file(file_path):
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == '[':
            data = json.load(f)
            records.extend(data)
        else:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    records.append(rec)
                except Exception as e:
                    print(f"Error parsing line in {file_path}: {e}")
    return records

def process_directory(directory):
    files = [os.path.join(directory, f) for f in os.listdir(directory) if f.startswith("step") and f.endswith("_file")]
    print(f"Found {len(files)} files to process.")
    for file_path in files:
        print(f"Processing file: {file_path}")
        records = process_file(file_path)
        with driver.session() as session:
            for record in records:
                try:
                    session.write_transaction(import_citation, record)
                except Exception as e:
                    print(f"Error processing record {record}: {e}")

if __name__ == "__main__":
    directory = "./"
    process_directory(directory)
    driver.close()

