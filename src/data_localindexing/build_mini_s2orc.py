"""
Author: Zhengyuan Dong
Created: 2025-03-09
Last Modified: 2025-03-09

python build_mini_s2orc.py build --directory /u4/z6dong/shared_data/se_s2orc_250218
python build_mini_s2orc.py query --title "BioMANIA: Simplifying bioinformatics data analysis through conversation" --directory /u4/z6dong/shared_data/se_s2orc_250218
"""

import os
import sqlite3
import json
import glob
from tqdm import tqdm
import argparse

CHECKPOINT_FILE = "checkpoint.txt"
DATABASE_FILE = "paper_index_mini.db"

def get_step_files(directory):
    """Retrieve all stepXX_file in sorted order."""
    return sorted(glob.glob(os.path.join(directory, 'step*_file')))

def read_checkpoint():
    """Read the checkpoint file to track processed files."""
    if not os.path.exists(CHECKPOINT_FILE):
        return set()
    with open(CHECKPOINT_FILE, 'r') as f:
        return set(line.strip() for line in f)

def write_checkpoint(filename):
    """Write processed files to the checkpoint to avoid reprocessing."""
    with open(CHECKPOINT_FILE, 'a') as f:
        f.write(filename + '\n')

def reset_database(db_path=DATABASE_FILE):
    """Reset the database by removing the existing file and recreating the schema."""
    if os.path.exists(db_path):
        os.remove(db_path)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute('''
        CREATE TABLE papers (
            title TEXT PRIMARY KEY,
            corpusid INTEGER,
            filename TEXT,
            line_index INTEGER
        )
    ''')
    conn.commit()
    conn.close()

def extract_annotations(data, key):
    raw_data = data.get("content", {}).get("annotations", {}).get(key, "[]")
    if not raw_data:  # Explicitly handle None values
        return []
    try:
        return json.loads(raw_data)
    except json.JSONDecodeError:
        return []

def extract_references(data, content_text):
    extracted_data = {}
    annotations = {
        "extracted_title": "title",
    }
    for key, annotation_key in annotations.items():
        extracted = [
            {
                "start": item["start"],
                "end": item["end"],
                "extracted_text": content_text[item["start"]:item["end"]],
                **({"id": item.get("attributes", {}).get("id", "unknown")} if "id" in item.get("attributes", {}) else {}),
                **({"figure_ref_id": item.get("attributes", {}).get("ref_id", "unknown")} if "ref_id" in item.get("attributes", {}) else {})
            }
            for item in extract_annotations(data, annotation_key)
            if isinstance(item.get("start"), int) and isinstance(item.get("end"), int) and item["start"] < item["end"]
        ]
        extracted_data[key] = extracted
    return extracted_data

def build_paper_index(directory, db_path=DATABASE_FILE):
    reset_database(db_path)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    processed_files = read_checkpoint()
    all_files = get_step_files(directory)
    for file in tqdm(all_files, desc="Building index from step files"):
        filename = os.path.basename(file)
        if filename in processed_files:
            continue
        filepath = os.path.join(directory, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            for index, line in enumerate(f):
                try:
                    paper = json.loads(line.strip())
                    content = paper.get('content', {})
                    text = content.get('text', '')
                    extracted_data = extract_references(paper, text)
                    #print(extracted_data)
                    #title = extracted_data['extracted_title']
                    if extracted_data['extracted_title'] and len(extracted_data['extracted_title'])>0:
                        title = extracted_data['extracted_title'][0]['extracted_text'].strip().lower()
                    else:
                        title = ""
                    corpusid = paper.get('corpusid')

                    if title and corpusid:
                        #print(f"Inserting: {title} -> {corpusid} (File: {filename}, Line: {index})")
                        cur.execute("""
                            INSERT OR IGNORE INTO papers 
                            (title, corpusid, filename, line_index)
                            VALUES (?, ?, ?, ?)
                        """, (title, corpusid, filename, index))

                except json.JSONDecodeError as e:
                    print(f"⚠️ JSON Decode Error in {filename}: {e}")
        conn.commit()
        write_checkpoint(filename)
    conn.close()

def query_paper_info(title, data_directory, db_path=DATABASE_FILE):
    """Retrieve corpusid, filename, and line index, then load JSON from the source file."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT corpusid, filename, line_index FROM papers WHERE title = ?", (title.lower(),))
    result = cur.fetchone()
    conn.close()

    if not result:
        print(f"❌ Paper '{title}' not found in database!")
        return None

    corpusid, filename, line_index = result
    filepath = os.path.join(data_directory, filename)
    print(f"✅ Found paper: {title} (CorpusID: {corpusid}) in {filename} at line {line_index}")

    if not os.path.exists(filepath):
        print(f"❌ Error: File {filename} not found in {data_directory}")
        return None

    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        if line_index >= len(lines):
            print(f"❌ Error: Line index {line_index} out of range (Total lines: {len(lines)})")
            return None
        try:
            paper = json.loads(lines[line_index].strip())
            content_text = paper.get('content', {}).get('text', '')
            # Extract tables and figures
            extracted_data = extract_references(paper, content_text)
            tables = extracted_data.get('extracted_tables', [])
            figures = extracted_data.get('extracted_figures', [])
            result_data = {
                "corpusid": corpusid,
                "title": title,
                "filename": filename,
                "line_index": line_index,
                "tables": tables,
                "figures": figures
            }
            #print(json.dumps(result_data, indent=2, ensure_ascii=False))
            return result_data
        except json.JSONDecodeError:
            print(f"❌ Error decoding JSON for '{title}' at {filename}:{line_index}")
            return None

def query_paper_by_corpusid(corpusid, data_directory, db_path=DATABASE_FILE):  
    conn = sqlite3.connect(db_path)                                            
    cur  = conn.cursor()                                                       
    cur.execute("""SELECT title, filename, line_index                          
                  FROM papers WHERE corpusid = ?""", (corpusid,))             
    result = cur.fetchone()                                                   
    conn.close()                                                              
                                                                               
    if not result:                                                            
        print(f"❌ CorpusID {corpusid} not found!")                            
        return None                                                           
                                                                               
    title, filename, line_index = result                                      
    filepath = os.path.join(data_directory, filename)                         
    print(f"✅ Found corpusid {corpusid} in {filename} at line {line_index}")  
                                                                               
    if not os.path.exists(filepath):                                          
        print(f"❌ File {filename} missing in {data_directory}")               
        return None                                                           
                                                                               
    with open(filepath, "r", encoding="utf-8") as fh:                         
        lines = fh.readlines()                                                
        if line_index >= len(lines):                                          
            print(f"❌ Line {line_index} out of range!")                       
            return None                                                       
        paper = json.loads(lines[line_index].strip())                         
    return paper                                                              

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build/query paper index with extraction & checkpointing")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--directory", type=str, required=True, help="Directory containing stepXX_file")

    query_parser = subparsers.add_parser("query")
    query_parser.add_argument("--title", type=str, required=True, help="Paper title to query")
    query_parser.add_argument("--directory", type=str, required=True, help="Directory containing stepXX_file")

    cid_parser = subparsers.add_parser("query_cid")                                
    cid_parser.add_argument("--corpusid", type=int, required=True,                
                            help="CorpusID to query")                              
    cid_parser.add_argument("--directory", type=str, required=True,               
                            help="Directory containing stepXX_file")              

    args = parser.parse_args()

    if args.mode == "build":
        build_paper_index(args.directory)
    elif args.mode == "query":
        paper = query_paper_info(args.title, args.directory)
        if paper:
            print(json.dumps(paper, indent=2, ensure_ascii=False))
    elif args.mode == "query_cid":                                                  
        paper = query_paper_by_corpusid(args.corpusid, args.directory)            
        if paper:                                                                 
            print(json.dumps(paper, indent=2, ensure_ascii=False))                

