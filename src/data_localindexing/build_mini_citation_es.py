#!/usr/bin/env python3
"""
Usage:
    python build_mini_citation_es.py --mode build --directory /u4/z6dong/shared_data/se_citations_250218 --index_name citations_index --fields minimal
    python build_mini_citation_es.py --mode build --directory /u4/z6dong/shared_data/se_citations_250218 --index_name citations_index_full --fields full
    python build_mini_citation_es.py --mode query --index_name citations_index --id 8982892
    python build_mini_citation_es.py --mode test --index_name citations_index
    python build_mini_citation_es.py --mode batch --input_file corpusIds.txt --index_name citations_index_full
    python build_mini_citation_es.py --mode update --directory /u4/z6dong/shared_data/se_citations_250218 --index_name citations_index # update from minimal to full

Notes:
  - In build mode, all NDJSON files (*.ndjson) in the specified directory will be processed.
  - In minimal mode, only the fields citationid, citingcorpusid and citedcorpusid are stored;
    in full mode, all fields (e.g. isinfluential, contexts, intents) are stored.
"""

import argparse
import json, time, os, re, glob, warnings
#import time, os, re, glob, warnings
import multiprocessing as mp
from functools import partial

#import orjson as json
from tqdm import tqdm
import pandas as pd
from elasticsearch import Elasticsearch, helpers
import matplotlib.pyplot as plt
from elasticsearch.helpers import parallel_bulk

warnings.filterwarnings("ignore")

# Elasticsearch connection parameters – adjust if needed.
ES_HOST = "http://localhost:9200"
ES_USER = "elastic"
ES_PASSWORD = "6KdUGb=SifNeWOy__lEz"

# Batch size for bulk import.
BATCH_SIZE = 8000

SEARCH_BATCH = 10000

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
    es.indices.create(index=index_name, body=index_body)
    print(f"Index {index_name} created.")

def backup_index(es, index_name):
    backup_index_name = index_name + "_backup"
    if es.indices.exists(index=backup_index_name):
        es.indices.delete(index=backup_index_name)
        print(f"Deleted existing backup index: {backup_index_name}")
    reindex_body = {
        "source": {"index": index_name},
        "dest": {"index": backup_index_name}
    }
    print(f"Creating backup index: {backup_index_name} ...")
    es.reindex(body=reindex_body, wait_for_completion=True, request_timeout=300)
    print(f"Backup index {backup_index_name} created.")
    return backup_index_name

def update_index_mini2full(es, directory, index_name):
    backup_index(es, index_name)
    files = glob.glob(os.path.join(directory, "step*_file"))
    total_updated = 0
    for file_path in files:
        print(f"Processing file for update: {file_path}")
        actions = []
        for batch in process_ndjson_file(file_path):
            for rec in batch:
                action = {
                    "_op_type": "update",
                    "_index": index_name,
                    "_id": rec.get("citationid"),
                    "doc": rec,
                    "doc_as_upsert": False
                }
                actions.append(action)
            if actions:
                helpers.bulk(es, actions, request_timeout=300)
                total_updated += len(actions)
                actions = []
    print(f"Update completed. Total documents updated: {total_updated}")

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

import glob

def prepare_record(rec, fields_mode):
    if fields_mode == "minimal":
        return {
            "citationid": rec.get("citationid"),
            "citingcorpusid": rec.get("citingcorpusid"),
            "citedcorpusid": rec.get("citedcorpusid")
        }
    # full mode: copy and serialize contexts/intents
    new_rec = rec.copy()
    if new_rec.get("contexts") is not None:
        new_rec["contexts"] = json.dumps(new_rec["contexts"])
    if new_rec.get("intents") is not None:
        new_rec["intents"] = json.dumps(new_rec["intents"])
    return new_rec

def count_lines_in_dir(directory, pattern="step*_file"):
    total = 0
    for path in glob.glob(os.path.join(directory, pattern)):
        try:
            with open(path, "r", encoding="utf-8") as f:
                for _ in f:
                    total += 1
        except:
            continue
    return total

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
        if es.indices.exists(index=index_name):
            es.indices.delete(index=index_name)
        print(f"Deleted existing index: {index_name}")
        create_citations_index(es, index_name)
        # Temporarily disable refresh & replicas for faster bulk ########
        es.indices.put_settings(
            index=index_name,
            body={"index": {"refresh_interval": "-1", "number_of_replicas": 0}}
        )
    files = sorted(f for f in glob.glob(os.path.join(directory, "step*_file"))
                   if f not in checkpoint["processed_files"])

    # ---------------- 多进程索引 -----------------
    def index_one(file_path, host, idx, mode, bs):
        es_local = Elasticsearch(host, verify_certs=False)
        actions = (
            {
                "_index": idx,
                "_id": rec["citationid"],
                "_source": prepare_record(rec, mode)
            }
            for batch in process_ndjson_file(file_path, bs)
            for rec  in batch
        )
        helpers.bulk(es_local, actions,
                     chunk_size=bs,
                     request_timeout=300,
                     refresh=False)
        return file_path

    pool_sz = 8 or mp.cpu_count()
    with mp.Pool(processes=pool_sz) as pool, \
         tqdm(total=len(files), desc="Files imported") as pbar:
        for done_file in pool.imap_unordered(
                partial(index_one,
                        host=ES_HOST,
                        idx=index_name,
                        mode=fields_mode,
                        bs=BATCH_SIZE),
                files):
            checkpoint["processed_files"].append(done_file)
            with open(checkpoint_file, "w", encoding="utf-8") as f:
                json.dump(checkpoint, f)
            pbar.update(1)
    
    print(f"Bulk import completed.")
    # Restore settings & force one refresh ########
    es.indices.put_settings(
        index=index_name,
        body={"index": {"refresh_interval": "30s", "number_of_replicas": 1}}
    )  ########
    es.indices.refresh(index=index_name)  ########

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
    response = es.search(index=citations_index, body=query_body, size=10000)
    hits = response["hits"]["hits"]
    if not hits:
        print(f"No citation edges found for paper id: {paper_id}")
        return
    print(f"Citation edges for paper id {paper_id}:")
    for hit in hits:
        src = hit["_source"]
        print(f"CitationID: {src.get('citationid')}, Citing: {src.get('citingcorpusid')}, Cited: {src.get('citedcorpusid')}")
    
def query_citations_by_id_mini(es, citations_index, target_id):
    """
    Query the citations index for citation edges where target_id appears as either citingcorpusid or citedcorpusid.
    Separates and prints the results into two groups.
    Work for minimal mode and full mode.
    """
    query_citing = {
        "query": {
            "term": {"citingcorpusid": target_id}
        }
    }
    query_cited = {
        "query": {
            "term": {"citedcorpusid": target_id}
        }
    }
    response_citing = es.search(index=citations_index, body=query_citing, size=SEARCH_BATCH)
    response_cited = es.search(index=citations_index, body=query_cited, size=SEARCH_BATCH)
    hits_citing = response_citing.get("hits", {}).get("hits", [])
    hits_cited = response_cited.get("hits", {}).get("hits", [])
    
    print(f"========== Citation edges where citingcorpusid equals {target_id} ==========")
    if not hits_citing:
        print("No documents found where citingcorpusid matches.")
    else:
        for hit in hits_citing:
            src = hit.get("_source", {})
            print(f"CitationID: {src.get('citationid')}, Citing: {src.get('citingcorpusid')}, Cited: {src.get('citedcorpusid')}")
    
    print(f"========== Citation edges where citedcorpusid equals {target_id} ==========")
    if not hits_cited:
        print("No documents found where citedcorpusid matches.")
    else:
        for hit in hits_cited:
            src = hit.get("_source", {})
            print(f"CitationID: {src.get('citationid')}, Citing: {src.get('citingcorpusid')}, Cited: {src.get('citedcorpusid')}")

def query_citations_by_id(es, citations_index, target_id):
    """
    Query the citations index for citation edges where target_id appears as either
    citingcorpusid or citedcorpusid.
    Returns a dictionary with two keys:
      - "cited_papers": List of items from records where target_id is in citingcorpusid
                        (i.e. target paper cites other papers).
      - "citing_papers": List of items from records where target_id is in citedcorpusid
                        (i.e. other papers cite the target paper).
    Each item is a JSON object containing all item fields (e.g. intents, contexts, isinfluential)
    and a nested key ("citedPaper" or "citingPaper") with the corresponding paper details.

    Work only for full mode.
    """
    # Query records where target_id is in citingcorpusid (target paper is doing the citing,
    # so the other side is a cited paper)
    query_from = {"query": {"term": {"citingcorpusid": target_id}}}
    resp_from = es.search(index=citations_index, body=query_from, size=SEARCH_BATCH)
    hits_from = resp_from.get("hits", {}).get("hits", [])
    cited_papers = []
    for hit in hits_from:
        src = hit.get("_source", {})
        item = {}
        item["intents"] = src.get("intents", [])
        item["contexts"] = src.get("contexts", [])
        item["isInfluential"] = src.get("isinfluential", False)
        item["citedPaper"] = {
            "paperId": src.get("citedcorpusid"),
            "title": src.get("citedtitle", None),
            "abstract": src.get("citedabstract", None)
        }
        cited_papers.append(item)
    
    # Query records where target_id is in citedcorpusid (target paper is being cited,
    # so the other side is a citing paper)
    query_to = {"query": {"term": {"citedcorpusid": target_id}}}
    resp_to = es.search(index=citations_index, body=query_to, size=SEARCH_BATCH)
    hits_to = resp_to.get("hits", {}).get("hits", [])
    citing_papers = []
    for hit in hits_to:
        src = hit.get("_source", {})
        item = {}
        item["intents"] = src.get("intents", [])
        item["contexts"] = src.get("contexts", [])
        item["isInfluential"] = src.get("isinfluential", False)
        item["citingPaper"] = {
            "paperId": src.get("citingcorpusid"),
            "title": src.get("citingtitle", None),
            "abstract": src.get("citingabstract", None)
        }
        citing_papers.append(item)
    
    return {"cited_papers": cited_papers, "citing_papers": citing_papers}
    
def batch_query(es, citations_index, input_file, output_file):
    """
    Batch query mode: Process an input file containing a list of paper IDs (one per line),
    display progress and elapsed time, and save non-empty results to a parquet file with columns:
      - corpusId
      - cited_papers
      - citing_papers
    Only papers with at least one citing or cited entry are saved.
    """
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} does not exist.")
        return
    with open(input_file, "r", encoding="utf-8") as f:
        corpusIds = [line.strip() for line in f if line.strip()]
    
    total = len(corpusIds)
    print(f"🚀 Starting batch query for {total} paper IDs...")
    start_time = time.time()

    results = []
    for pid in tqdm(corpusIds, desc="Querying papers"):
        cited = []
        citing = []
        for src in scan_citations(es, citations_index, pid):
            if src["citingcorpusid"] == pid:
                cited.append(src)
            else:
                citing.append(src)
        if cited or citing:
            results.append({
                "corpusId": pid,
                "cited_papers": cited,
                "citing_papers": citing
            })
    elapsed = time.time() - start_time
    if results:
        df = pd.DataFrame(results)
        df.to_parquet(output_file, compression='zstd', engine='pyarrow', index=False)
        # —— add visualization for the count histogram —— #
        ref_counts = df["cited_papers"].apply(len)
        cit_counts = df["citing_papers"].apply(len)

        plt.figure()
        plt.hist(ref_counts, bins=50, color="#add8e6")
        plt.title("Distribution of References per Paper")
        plt.xlabel("Number of References")
        plt.ylabel("Frequency")
        plt.savefig("references_histogram.png")

        plt.figure()
        plt.hist(cit_counts, bins=50, color="#ffcc99")
        plt.title("Distribution of Citations per Paper")
        plt.xlabel("Number of Citations")
        plt.ylabel("Frequency")
        plt.savefig("citations_histogram.png")

        print(f"💾 Batch results saved to {output_file} and histograms generated")
     
    else:
        print("⚠️ No records to save; all results empty.")
    print(f"⏱ Total time: {elapsed:.2f}s for {total} queries")

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

def scan_citations(es, citations_index, pid):
    # only do one compound query, scan all citingcorpusid or citedcorpusid=pid
    query = {
        "query": {
            "bool": {
                "should": [
                    {"term": {"citingcorpusid": pid}},
                    {"term": {"citedcorpusid": pid}}
                ],
                "minimum_should_match": 1
            }
        }
    }
    # helpers.scan will automatically handle the scroll context and fetch all data
    for hit in helpers.scan(
        client=es,
        index=citations_index,
        query=query,
        scroll="5m",       # scroll context time
        size=1000          # get 1000 records at a time
    ):
        yield hit["_source"]


def main():
    parser = argparse.ArgumentParser(description="Elasticsearch Citation Graph Loader & Query Tool")
    parser.add_argument("--mode", choices=["build", "query", "test", "batch", "update", "prepare_ids"], required=True,
                        help="build: import citation data; query: search by paper title; test: show index samples")
    parser.add_argument("--directory", type=str, help="Directory with NDJSON citation files (for build mode)")
    parser.add_argument("--index_name", type=str, default="citations_index",
                        help="Elasticsearch index name for citations")
    parser.add_argument("--fields", choices=["full", "minimal"], default="minimal",
                        help="Store full fields or only minimal fields (citationid, citingcorpusid, citedcorpusid)")
    parser.add_argument("--id", type=str, default="150223110", help="Paper id to search for (query mode)")
    parser.add_argument("--input_file", type=str, default="tmp.txt", help="Input file with paper IDs for batch query mode")
    parser.add_argument("--output_file", type=str, default="batch_results.parquet", help="Input file with paper IDs for batch query mode")
    args = parser.parse_args()

    es = Elasticsearch(
        ES_HOST,
        basic_auth=(ES_USER, ES_PASSWORD),
        verify_certs=False
    )
    """es.indices.put_settings(
        index="citations_index",
        body={
            "index": {
                "max_result_window": 50
            }
        }
    )"""
    #es = Elasticsearch(["http://{}:9200".format(ES_HOST)], verify_certs=False)

    if args.mode == "build":
        if not args.directory:
            print("Error: --directory is required in build mode.")
            return
        build_citations_index(es, args.directory, args.index_name, args.fields)
    elif args.mode == "query":
        #all_indices = es.indices.get_alias("*").keys()
        #print("Available indices in the cluster:", list(all_indices))

        result = query_citations_by_id(es, args.index_name, args.id)
        print(json.dumps(result, indent=2))
    elif args.mode == "test":
        test_index(es, args.index_name)
    elif args.mode == "batch":
        if not args.input_file:
            print("Error: --input_file is required for batch mode.")
            return
        batch_query(es, args.index_name, args.input_file, args.output_file)
    elif args.mode == "update":
        if not args.directory:
            print("Error: --directory is required in update mode.")
            return
        update_index_mini2full(es, args.directory, args.index_name)
    elif args.mode == "prepare_ids":                                        
        # load all paperId from titles cache, write to tmp_ids.txt
        TITLES_CACHE_FILE = "./s2orc_titles2ids.parquet"
        df = pd.read_parquet(TITLES_CACHE_FILE)
        ids = df["corpusId"].drop_duplicates().astype(str).tolist()
        tmp_file = "tmp_local_ids.txt"
        with open(tmp_file, "w") as f:
            f.write("\n".join(ids))
        print(f"🎉 Saved {len(ids)} paper IDs to {tmp_file}")

if __name__ == "__main__":
    main()
