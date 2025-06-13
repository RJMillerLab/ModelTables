import json
import os
import argparse
from pathlib import Path

def load_corpus(corpus_file):
    """Load corpus from jsonl file."""
    corpus = []
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                corpus.append(entry)
    return corpus

def create_queries_tsv(corpus, output_file):
    """Create queries.tsv from corpus entries using original ID."""
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in corpus:
            # Use the original ID directly
            query_id = entry['id']
            
            # Clean the content: remove newlines and tabs that could break TSV format
            query_text = entry['contents'].replace('\n', ' ').replace('\t', ' ').replace('\r', ' ')
            
            f.write(f"{query_id}\t{query_text}\n")
    
    print(f"Created {len(corpus)} queries in {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Create queries.tsv from corpus for pyserini')
    parser.add_argument('--corpus', default='data/tmp/corpus/collection.jsonl', 
                        help='Path to corpus jsonl file')
    parser.add_argument('--output-dir', default='data/tmp/', 
                        help='Output directory for query files')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"🔄 Loading corpus from {args.corpus}...")
    corpus = load_corpus(args.corpus)
    
    # Create queries file
    queries_file = os.path.join(args.output_dir, 'queries.tsv')
    create_queries_tsv(corpus, queries_file)

if __name__ == "__main__":
    main()