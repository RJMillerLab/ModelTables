#!/usr/bin/env python3
"""
Usage:
  # Build mode: Import citation NDJSON data into Elasticsearch.
  python run_citations_es.py --mode build --directory /u4/z6dong/shared_data/se_citations_250218 --index_name /u4/z6dong/shared_data/se_citations_250218/citations_index --fields minimal

  # Query mode: Given a paper title, first fuzzy-match in papers_index to get the paper id,
  # then query citations_index for any citations where this id appears as citing or cited.
  python run_citations_es.py --mode query --paper_index /u4/z6dong/shared_data/se_s2orc_250218/papers_index --index_name /u4/z6dong/shared_data/se_citations_250218/citations_index --title "BioMANIA: Simplifying bioinformatics data analysis through conversation"

  # Test mode: Display index stats and sample documents from citations_index.
  python run_citations_es.py --mode test --index_name /u4/z6dong/shared_data/se_citations_250218/citations_index

Notes:
  - In build mode, all NDJSON files (*.ndjson) in the specified directory will be processed.
  - In minimal mode, only the fields citationid, citingcorpusid and citedcorpusid are stored;
    in full mode, all fields (e.g. isinfluential, contexts, intents) are stored.
"""

import argparse
import json
import os
import warnings
import glob
import re
from tqdm import tqdm
from elasticsearch import Elasticsearch, helpers

warnings.filterwarnings("ignore")

# Elasticsearch connection parameters – adjust if needed.
ES_HOST = "http://localhost:9200"
ES_USER = "elastic"
ES_PASSWORD = "6KdUGb=SifNeWOy__lEz"

# Batch size for bulk import.
BATCH_SIZE = 2000

def create_citations_index(es, index_name):
    """
    Create the citations index with mapping.
    For keyword fields (e.g. citingcorpusid, citedcorpusid) set ignore_above to avoid errors.
    """
    index_body = {
        "settings": {
            "index": {
                "number_of_shards": 3,
                "number_of_replicas": 1,
                "refresh_interval": "30s"
            }
        },
        "mappings": {
            "properties": {
                "citationid": {"type": "keyword"},
                "citingcorpusid": {"type": "keyword", "ignore_above": 256},
                "citedcorpusid": {"type": "keyword", "ignore_above": 256},
                "isinfluential": {"type": "boolean"},
                "contexts": {"type": "text"},
                "intents": {"type": "text"}
            }
        }
    }
    if es.indices.exists(index=index_name):
        es.indices.delete(index=index_name)
        print(f"Deleted existing index: {index_name}")
    es.indices.create(index=index_name, body=index_body)
    print(f"Index {index_name} created.")

def process_ndjson_file(file_path, batch_size=BATCH_SIZE):
    """
    Read NDJSON file line by line in batches.
    Yield each batch as a list of records.
    """
    batch = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                # Skip records where citing or cited id is None.
                if record.get("citingcorpusid") is None or record.get("citedcorpusid") is None:
                    continue
                batch.append(record)
            except Exception as e:
                print(f"⚠️ Skipping invalid JSON line: {e}")
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

def build_citations_index(es, directory, index_name, fields_mode):
    """
    Process all NDJSON files in the specified directory and import data into Elasticsearch.
    """
    checkpoint_file = "checkpoint_es.json"
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            checkpoint = json.load(f)
    else:
        checkpoint = {"processed_files": []}
    create_citations_index(es, index_name)
    files = glob.glob(os.path.join(directory, "step*_file"))
    total_docs = 0
    for file_path in files:
        if file_path in checkpoint["processed_files"]:
            print(f"Skipping already processed file: {file_path}")
            continue
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                file_line_count = sum(1 for _ in f)
        except Exception:
            file_line_count = 0
        print(f"Processing file: {file_path} (approx {file_line_count} lines)")
        pbar = tqdm(total=file_line_count, desc=os.path.basename(file_path), leave=False)
        for batch in process_ndjson_file(file_path):
            actions = []
            for rec in batch:
                if fields_mode == "minimal":
                    new_rec = {
                        "citationid": rec.get("citationid"),
                        "citingcorpusid": rec.get("citingcorpusid"),
                        "citedcorpusid": rec.get("citedcorpusid")
                    }
                else:
                    # In full mode, copy all fields and perform optional conversion for nested fields.
                    new_rec = rec.copy()
                    if new_rec.get("contexts") is not None:
                        new_rec["contexts"] = json.dumps(new_rec["contexts"])
                    if new_rec.get("intents") is not None:
                        new_rec["intents"] = json.dumps(new_rec["intents"])
                actions.append({
                    "_index": index_name,
                    "_id": rec.get("citationid"),
                    "_source": new_rec
                })
            helpers.bulk(es, actions, request_timeout=300)
            total_docs += len(batch)
            pbar.update(len(batch))
        pbar.close()
        checkpoint["processed_files"].append(file_path)
        with open(checkpoint_file, "w", encoding="utf-8") as f:
            json.dump(checkpoint, f)
    print(f"Bulk import completed. Total citation documents imported: {total_docs}")

def fuzzy_search_paper(es, paper_index, title):
    """
    Perform a fuzzy search in papers_index to obtain the paper id.
    Assumes papers_index has a field 'title' with subfields 'processed' and 'exact'.
    """
    # Preprocess title: lowercase and remove non-alphanumeric/hyphen characters.
    q_str = title.strip().lower()
    q_str = re.sub(r"[^a-z0-9\s\-]", "", q_str)
    q_str = re.sub(r"\s+", " ", q_str)
    
    query_body = {
        "query": {
            "bool": {
                "should": [
                    {
                        "match_phrase": {
                            "title.processed": {
                                "query": q_str,
                                "slop": 3,
                                "boost": 3
                            }
                        }
                    },
                    {
                        "match": {
                            "title.processed": {
                                "query": q_str,
                                "fuzziness": "AUTO",
                                "boost": 2
                            }
                        }
                    },
                    {
                        "term": {
                            "title.exact": {
                                "value": q_str,
                                "boost": 5
                            }
                        }
                    }
                ],
                "minimum_should_match": 1
            }
        }
    }
    response = es.search(index=paper_index, body=query_body, size=1)
    hits = response["hits"]["hits"]
    if not hits:
        print("No matching paper found for title:", title)
        return None
    paper_id = hits[0]["_source"].get("corpusid")
    print(f"Fuzzy search result: Paper title '{title}' -> corpusid: {paper_id}")
    return paper_id

def query_citations(es, paper_index, citations_index, title):
    """
    Given a paper title, first perform fuzzy search in papers_index to get the paper id,
    then query citations_index for all citation edges where this id appears as citing or cited.
    """
    paper_id = fuzzy_search_paper(es, paper_index, title)
    if not paper_id:
        return

    query_body = {
        "query": {
            "bool": {
                "should": [
                    {"term": {"citingcorpusid": paper_id}},
                    {"term": {"citedcorpusid": paper_id}}
                ],
                "minimum_should_match": 1
            }
        }
    }
    response = es.search(index=citations_index, body=query_body, size=50)
    hits = response["hits"]["hits"]
    if not hits:
        print(f"No citation edges found for paper id: {paper_id}")
        return
    print(f"Citation edges for paper id {paper_id}:")
    for hit in hits:
        src = hit["_source"]
        print(f"CitationID: {src.get('citationid')}, Citing: {src.get('citingcorpusid')}, Cited: {src.get('citedcorpusid')}")
    
def test_index(es, index_name):
    """
    Display the total document count and sample documents from the specified index.
    """
    count_resp = es.count(index=index_name)
    print(f"Total documents in index '{index_name}':", count_resp.get("count"))
    response = es.search(index=index_name, body={"query": {"match_all": {}}}, size=5, sort="_doc")
    print("Sample citation documents:")
    for hit in response["hits"]["hits"]:
        print(hit["_source"])

def main():
    parser = argparse.ArgumentParser(description="Elasticsearch Citation Graph Loader & Query Tool")
    parser.add_argument("--mode", choices=["build", "query", "test"], required=True,
                        help="build: import citation data; query: search by paper title; test: show index samples")
    parser.add_argument("--directory", type=str, help="Directory with NDJSON citation files (for build mode)")
    parser.add_argument("--index_name", type=str, default="citations_index",
                        help="Elasticsearch index name for citations")
    parser.add_argument("--paper_index", type=str, default="papers_index",
                        help="Elasticsearch index name for papers (for query mode)")
    parser.add_argument("--title", type=str, help="Paper title to search for (query mode)")
    parser.add_argument("--fields", choices=["full", "minimal"], default="minimal",
                        help="Store full fields or only minimal fields (citationid, citingcorpusid, citedcorpusid)")
    args = parser.parse_args()

    es = Elasticsearch(
        ES_HOST,
        basic_auth=(ES_USER, ES_PASSWORD),
        verify_certs=False
    )
    #es = Elasticsearch(["http://{}:9200".format(ES_HOST)], verify_certs=False)

    if args.mode == "build":
        if not args.directory:
            print("Error: --directory is required in build mode.")
            return
        build_citations_index(es, args.directory, args.index_name, args.fields)
    elif args.mode == "query":
        if not args.title:
            print("Error: --title is required in query mode.")
            return
        query_citations(es, args.paper_index, args.index_name, args.title)
    elif args.mode == "test":
        test_index(es, args.index_name)

if __name__ == "__main__":
    main()
