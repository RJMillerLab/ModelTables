"""
Model Card Retrieval pipeline, based on Sentence-BERT to encode model card text, and use FAISS for nearest neighbor search.
Subcommands:
  1. filter       : filter model cards from parquet, and generate JSONL (each line {id, contents})
  2. encode       : encode JSONL content with SBERT, and save NPZ
  3. build_faiss  : build FAISS index
  4. search       : search all embeddings, and output neighbor mapping JSON
Usage:
# build corpus
python src/modelsearch/pipeline_mc/modelcard_retrieval_pipeline.py \
filter --parquet data/processed/modelcard_step1.parquet \
        --field card \
        --output_jsonl output/baseline_mc/modelcard_corpus.jsonl \
        --model_name all-MiniLM-L6-v2 \
        --device cuda

python src/modelsearch/pipeline_mc/modelcard_retrieval_pipeline.py \
encode --jsonl output/baseline_mc/modelcard_corpus.jsonl \
        --model_name all-MiniLM-L6-v2 \
        --batch_size 256 \
        --output_npz output/baseline_mc/modelcard_embeddings.npz \
        --device cuda

python src/modelsearch/pipeline_mc/modelcard_retrieval_pipeline.py \
build_faiss --emb_npz output/baseline_mc/modelcard_embeddings.npz \
            --output_index output/baseline_mc/modelcard.faiss

python src/modelsearch/pipeline_mc/modelcard_retrieval_pipeline.py \
search --emb_npz output/baseline_mc/modelcard_embeddings.npz \
        --faiss_index output/baseline_mc/modelcard.faiss \
        --top_k 5 \
        --output_json output/baseline_mc/modelcard_neighbors.json
"""
import os
import json
import argparse
import numpy as np
from tqdm import tqdm
import faiss
from sentence_transformers import SentenceTransformer
import pandas as pd
from typing import Tuple, List
from pathlib import Path


def load_combined_data(data_type: str, file_path: str, columns: List[str]) -> pd.DataFrame:
    """Load combined data from parquet files."""
    import glob
    from pathlib import Path
    
    file_path = Path(file_path).expanduser()
    pattern = f"{file_path}/**/{data_type}_step1.parquet"
    files = glob.glob(str(pattern), recursive=True)
    
    if not files:
        raise FileNotFoundError(f"No {data_type}_step1.parquet files found in {file_path}")
    
    print(f"Found {len(files)} parquet files: {files}")
    
    dfs = []
    for file in files:
        try:
            df = pd.read_parquet(file, columns=columns)
            dfs.append(df)
        except Exception as e:
            print(f"Error loading {file}: {e}")
            continue
    
    if not dfs:
        raise ValueError("No valid parquet files could be loaded")
    
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"Combined data shape: {combined_df.shape}")
    return combined_df


def filter_to_jsonl(parquet_path: str, field: str, output_jsonl: str,
                    model_name: str, device: str):
    """
    each line: {"id": modelId, "contents": processed model card text}
    """
    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)
    model = SentenceTransformer(model_name, device=device)
    model.eval()

    # Load model card data
    print(f"Loading model card data from {parquet_path}...")
    df = load_combined_data("modelcard", file_path="~/Repo/CitationLake/data/raw", 
                           columns=['modelId', field])
    
    written = 0
    with open(output_jsonl, 'w', encoding='utf-8') as fout:
        for _, row in df.iterrows():
            model_id = row['modelId']
            contents = str(row[field])
            
            if not contents or contents.lower() == "nan" or contents.strip() == "":
                continue
                
            doc = {'id': model_id, 'contents': contents}
            fout.write(json.dumps(doc, ensure_ascii=False) + '\n')
            written += 1
    
    print(f'Generated JSONL: {output_jsonl}, entries written: {written}/{len(df)}')


def encode_corpus(jsonl: str, model_name: str, batch_size: int,
                  output_npz: str, device: str):
    """
    Batch encode JSONL text with SBERT, save embeddings and ids.
    """
    os.makedirs(os.path.dirname(output_npz), exist_ok=True)
    model = SentenceTransformer(model_name, device=device)
    model.eval()

    ids: List[str] = []
    texts: List[str] = []
    with open(jsonl, 'r', encoding='utf-8') as fin:
        for line in fin:
            try:
                obj = json.loads(line)
                ids.append(obj['id'])
                texts.append(obj['contents'])
            except json.JSONDecodeError:
                continue

    if not texts:
        print(f'No documents found in {jsonl}, skipping encode.')
        return

    all_embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc='Encoding'):
        batch = texts[i:i+batch_size]
        try:
            embs = model.encode(
                batch, convert_to_numpy=True,
                show_progress_bar=False, batch_size=batch_size
            )
            if embs.size > 0:
                all_embs.append(embs)
        except Exception as e:
            print(f'Error encoding batch {i}:{e}')
            continue

    if not all_embs:
        print('No embeddings generated, skipping save.')
        return

    embs_array = np.vstack(all_embs).astype('float32')
    np.savez_compressed(output_npz,
                        embeddings=embs_array,
                        ids=np.array(ids))
    print(f'Saved embeddings: {output_npz}, shape={embs_array.shape}')


def build_faiss(emb_npz: str, output_index: str):
    """
    Build FAISS index for inner product retrieval, and save to disk.
    """
    data = np.load(emb_npz)
    embs = data['embeddings']
    faiss.normalize_L2(embs)
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embs)
    os.makedirs(os.path.dirname(output_index), exist_ok=True)
    faiss.write_index(index, output_index)
    print(f'Saved FAISS index: {output_index}, vectors={index.ntotal}')


def search_neighbors(emb_npz: str, faiss_index: str,
                     top_k: int, output_json: str):
    """
    Search all embeddings, remove self, save neighbor mapping.
    """
    data = np.load(emb_npz)
    embs = data['embeddings']
    ids = data['ids'].tolist()
    index = faiss.read_index(faiss_index)

    D, I = index.search(embs, top_k+1)
    neighbors = {}
    for i, neigh in enumerate(I):
        base = ids[i]
        nb = [ids[j] for j in neigh if j != i][:top_k]
        neighbors[base] = nb

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, 'w', encoding='utf-8') as fout:
        json.dump(neighbors, fout, ensure_ascii=False, indent=2)
    print(f'Saved neighbor mapping: {output_json}')


def search_query(query_text: str, faiss_index: str, emb_npz: str, 
                 model_name: str, top_k: int, device: str):
    """
    Search for a single query and return top-k candidates.
    """
    # Load model and encode query
    model = SentenceTransformer(model_name, device=device)
    query_emb = model.encode([query_text], convert_to_numpy=True)
    faiss.normalize_L2(query_emb)
    
    # Load index and search
    index = faiss.read_index(faiss_index)
    D, I = index.search(query_emb, top_k)
    
    # Load document IDs
    data = np.load(emb_npz)
    ids = data['ids'].tolist()
    
    # Return top-k candidates
    candidates = [ids[i] for i in I[0]]
    return candidates


def main():
    parser = argparse.ArgumentParser(
        description='Model Card Retrieval Pipeline: filter->encode->build_faiss->search'
    )
    sub = parser.add_subparsers(dest='cmd')

    p = sub.add_parser('filter')
    p.add_argument('--parquet', required=True,
                   help='Path to modelcard parquet file')
    p.add_argument('--field', default='card',
                   help='Field name to extract from parquet (default: card)')
    p.add_argument('--output_jsonl', required=True)
    p.add_argument('--model_name', default='all-MiniLM-L6-v2')
    p.add_argument('--device', default='cuda')

    e = sub.add_parser('encode')
    e.add_argument('--jsonl', required=True)
    e.add_argument('--model_name', default='all-MiniLM-L6-v2')
    e.add_argument('--batch_size', type=int, default=256)
    e.add_argument('--output_npz', required=True)
    e.add_argument('--device', default='cuda')

    b = sub.add_parser('build_faiss')
    b.add_argument('--emb_npz', required=True)
    b.add_argument('--output_index', required=True)

    s = sub.add_parser('search')
    s.add_argument('--emb_npz', required=True)
    s.add_argument('--faiss_index', required=True)
    s.add_argument('--top_k', type=int, default=5)
    s.add_argument('--output_json', required=True)

    q = sub.add_parser('query')
    q.add_argument('--query', required=True, help='Query text to search')
    q.add_argument('--faiss_index', required=True)
    q.add_argument('--emb_npz', required=True)
    q.add_argument('--model_name', default='all-MiniLM-L6-v2')
    q.add_argument('--top_k', type=int, default=5)
    q.add_argument('--device', default='cuda')

    args = parser.parse_args()
    if args.cmd == 'filter':
        filter_to_jsonl(args.parquet, args.field,
                        args.output_jsonl,
                        args.model_name, args.device)
    elif args.cmd == 'encode':
        encode_corpus(args.jsonl, args.model_name,
                      args.batch_size, args.output_npz,
                      args.device)
    elif args.cmd == 'build_faiss':
        build_faiss(args.emb_npz, args.output_index)
    elif args.cmd == 'search':
        search_neighbors(args.emb_npz, args.faiss_index,
                         args.top_k, args.output_json)
    elif args.cmd == 'query':
        candidates = search_query(args.query, args.faiss_index, args.emb_npz,
                                 args.model_name, args.top_k, args.device)
        print(f"Top-{args.top_k} candidates for query '{args.query}':")
        for i, candidate in enumerate(candidates, 1):
            print(f"{i}. {candidate}")
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
