#!/usr/bin/env python
"""
Usage:
  # Build mode: Import data from SQLite into Elasticsearch.
  python build_mini_s2orc_es.py --mode build --directory /u4/z6dong/shared_data/se_s2orc_250218 --index_name papers_index --db_file /u4/z6dong/shared_data/se_s2orc_250218/paper_index_mini.db

  # Query mode: Execute a combined fuzzy search on the Elasticsearch index using the provided title.
  python build_mini_s2orc_es.py --mode query --directory /u4/z6dong/shared_data/se_s2orc_250218 --index_name papers_index --query "BioMANIA: Simplifying bioinformatics data analysis through conversation"

  # Test mode: Display Elasticsearch index stats and print the first 5 documents.
  python build_mini_s2orc_es.py --mode test --directory /u4/z6dong/shared_data/se_s2orc_250218 --index_name papers_index --db_file /u4/z6dong/shared_data/se_s2orc_250218/paper_index_mini.db

Notes:
  - Build mode: Data is streamed from the SQLite database and imported into Elasticsearch using streaming_bulk.
    If some documents fail to index, the errors are logged and the process continues.
  - Query mode: Executes a bool query on the "title" field combining:
      • match_phrase on title.processed (slop=3, boost=3),
      • fuzzy match on title.processed (fuzziness=AUTO, boost=2),
      • term query on title.exact (boost=5).
  - Test mode: Displays index statistics and prints the first 5 documents.
  - Warnings (e.g., insecure TLS warnings) are suppressed.
"""

import argparse
import sqlite3
import os
import warnings
import re
from elasticsearch import Elasticsearch, helpers
from elasticsearch.helpers import streaming_bulk
from tqdm import tqdm

# Suppress warnings (InsecureRequestWarning, DeprecationWarning, etc.)
warnings.filterwarnings("ignore")

# Default settings
DATABASE_FILE = "/u4/z6dong/shared_data/se_s2orc_250218/paper_index_mini.db"
INDEX_NAME = "papers_index"

def count_rows(db_file):
    """Return the total number of rows in the 'papers' table."""
    conn = sqlite3.connect(db_file)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM papers")
    total = cur.fetchone()[0]
    conn.close()
    return total

def stream_from_db(db_file, index_name):
    """
    Stream data from SQLite and yield documents for bulk import.
    Document structure adjusted to match index mapping.
    """
    conn = sqlite3.connect(db_file)
    cur = conn.cursor()
    cur.execute("SELECT corpusid, title FROM papers")
    while True:
        row = cur.fetchone()
        if row is None:
            break
        corpusid, title = row
        orig_title = title.strip()
        # Removed manual processed fields; rely on Elasticsearch analyzer ########
        yield {
            "_index": index_name,
            "_id": corpusid,
            "_source": {
                "title": orig_title,         ######## changed: only index original title
                "corpusid": corpusid
            }
        }
    conn.close()

def build_index(es, db_file, index_name):
    """Build mode: Create index with custom mapping and import data from SQLite using streaming_bulk."""
    # Define index mapping with custom analyzer.
    index_body = {
        "settings": {
            "index": {
                "number_of_shards": 3,
                "number_of_replicas": 1,
                "refresh_interval": "30s"
            },
            "analysis": {
                "analyzer": {
                    "processed_analyzer": {
                        "type": "custom",
                        "tokenizer": "standard",
                        "filter": ["lowercase", "asciifolding"]
                    }
                }
            }
        },
        "mappings": {
            "properties": {
                "title": {
                    "type": "text",
                    "fields": {
                        "processed": {
                            "type": "text",
                            "analyzer": "processed_analyzer"
                        },
                        "exact": {
                            "type": "keyword",
                            "ignore_above": 256
                        }
                    }
                },
                "corpusid": {
                    "type": "keyword"
                }
            }
        }
    }
    # Delete existing index and create a fresh one.
    if es.indices.exists(index=index_name):
        es.indices.delete(index=index_name)
    es.indices.create(index=index_name, body=index_body)
    print(f"Index {index_name} created.")

    total_rows = count_rows(db_file)
    print(f"Total rows to import: {total_rows}")
    actions = stream_from_db(db_file, index_name)
    success_count = 0
    fail_count = 0
    fail_print_count = 0  ######## added: counter for printing error details

    with tqdm(total=total_rows, desc="Importing rows", ncols=80) as pbar:
        # Use streaming_bulk with chunk_size=2000 and do not raise on error.
        for ok, info in streaming_bulk(
            client=es,
            actions=actions,
            chunk_size=2000,
            raise_on_error=False,
            request_timeout=300
        ):
            pbar.update(1)
            if ok:
                success_count += 1
            else:
                fail_count += 1
                if fail_print_count < 5:  ######## added: print error details for first few failures
                    print(f"Error indexing document: {info}")
                    fail_print_count += 1

    print(f"Bulk import completed. Success: {success_count}, Failures: {fail_count}")

    # Debug: Analyze a sample title.
    sample_title = "BioMANIA: Simplifying bioinformatics data analysis through conversation"
    analysis = es.indices.analyze(index=index_name, body={
        "analyzer": "processed_analyzer",
        "text": sample_title
    })
    tokens = [tok["token"] for tok in analysis["tokens"]]
    print(f"Analyzed tokens for sample title: {tokens}")

def query_index(es, index_name, query_title):
    """Query mode: Execute a combined bool query on the 'title' field."""
    if not es.indices.exists(index=index_name):
        print(f"Index '{index_name}' does not exist.")
        return

    # Preprocess query string similarly to indexing.
    q_str = query_title.strip().lower()
    q_str = re.sub(r"[^a-z0-9\s\-]", "", q_str)
    q_str = re.sub(r"\s+", " ", q_str)

    # Combined bool query:
    body = {
        "query": {
            "bool": {
                "should": [
                    {  # Phrase match on processed field.
                        "match_phrase": {
                            "title.processed": {
                                "query": q_str,
                                "slop": 3,
                                "boost": 3
                            }
                        }
                    },
                    {  # Fuzzy match on processed field.
                        "match": {
                            "title.processed": {
                                "query": q_str,
                                "fuzziness": "AUTO",
                                "boost": 2
                            }
                        }
                    },
                    {  # Exact term match on exact field.
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
        },
        "highlight": {
            "fields": {
                "title": {}
            }
        }
    }
    print("Debug: Query body:")
    print(body)

    response = es.search(index=index_name, body=body, size=10)
    print(f"Query Results for title: '{query_title}'")
    hits = response["hits"]["hits"]
    if not hits:
        print("No results found.")
    for hit in hits:
        src = hit["_source"]
        print(f"Title: {src['title']} | CorpusID: {src['corpusid']} | Score: {hit['_score']}")

def test_index(es, index_name):
    """Test mode: Display index statistics and print the first 5 documents."""
    if not es.indices.exists(index=index_name):
        print(f"Index '{index_name}' does not exist.")
        return

    count = es.count(index=index_name)["count"]
    print(f"Total documents in index '{index_name}': {count}")
    body = {"query": {"match_all": {}}}
    response = es.search(index=index_name, body=body, size=5, sort="_doc")
    print("First 5 documents:")
    for hit in response["hits"]["hits"]:
        src = hit["_source"]
        print(f"CorpusID: {src['corpusid']}, Title: {src['title']}")

def main():
    parser = argparse.ArgumentParser(description="Elasticsearch Build, Query, and Test Tool")
    parser.add_argument("--mode", choices=["build", "query", "test"], required=True, help="Mode: build, query, or test")
    parser.add_argument("--directory", required=True, help="Directory containing the SQLite database (paper_mini.db)")
    parser.add_argument("--index_name", required=True, help="Elasticsearch index name")
    parser.add_argument("--db_file", default=DATABASE_FILE, help="SQLite database filename (located in the specified directory)")
    parser.add_argument("--query", help="Query string (required for query mode)")
    args = parser.parse_args()

    db_file = os.path.join(args.directory, args.db_file)

    es = Elasticsearch(
        "https://localhost:9200",
        basic_auth=("elastic", "6KdUGb=SifNeWOy__lEz"),
        verify_certs=False
    )

    if args.mode == "build":
        print("🔨 Build mode: Building a complete Elasticsearch index (chunk-based streaming)")
        if es.indices.exists(index=args.index_name):
            print(f"Index {args.index_name} already exists. Deleting it for a fresh build...")
            es.indices.delete(index=args.index_name)
        build_index(es, db_file, args.index_name)
    elif args.mode == "query":
        if not args.query:
            print("❌ Query mode requires a query string. Please use the --query parameter.")
        else:
            print("🚀 Query mode: Executing query")
            query_index(es, args.index_name, args.query)
    elif args.mode == "test":
        print("🚀 Test mode: Displaying index stats and first 5 documents")
        test_index(es, args.index_name)

if __name__ == "__main__":
    main()

