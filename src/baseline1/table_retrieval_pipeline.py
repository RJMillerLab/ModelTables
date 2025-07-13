"""
Dense Retrieval pipeline, based on Sentence-BERT to encode table text, and use FAISS for nearest neighbor search.
Subcommands:
  1. filter       : filter CSV from mask file, and generate JSONL (each line {id, contents})
  2. encode       : encode JSONL content with SBERT, and save NPZ
  3. build_faiss  : build FAISS index
  4. search       : search all embeddings, and output neighbor mapping JSON
Usage:
# build corpus
python src/baseline_1/table_retrieval_pipeline.py \
filter --base_path data/processed \
        --mask_file data/analysis/all_valid_title_valid.txt \
        --output_jsonl data/baseline/valid_tables.jsonl \
        --model_name all-MiniLM-L6-v2 \
        --device cuda

python src/baseline_1/table_retrieval_pipeline.py \
encode --jsonl data/baseline/valid_tables.jsonl \
        --model_name all-MiniLM-L6-v2 \
        --batch_size 256 \
        --output_npz data/baseline/valid_tables_embeddings.npz \
        --device cuda

python src/baseline_1/table_retrieval_pipeline.py \
build_faiss --emb_npz data/baseline/valid_tables_embeddings.npz \
            --output_index data/baseline/valid_tables.faiss

python src/baseline_1/table_retrieval_pipeline.py \
search --emb_npz data/baseline/valid_tables_embeddings.npz \
        --faiss_index data/baseline/valid_tables.faiss \
        --top_k 5 \
        --output_json data/baseline/table_neighbors.json
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


def safe_load_csv(file_path: str) -> Tuple[str, pd.DataFrame]:
    filename = os.path.basename(file_path)
    encodings = ['utf-8', 'latin1', 'cp1252']
    for enc in encodings:
        try:
            df = pd.read_csv(file_path, encoding=enc, low_memory=False)
            if df is not None and not df.empty:
                return filename, df
        except:
            pass
        try:
            df = pd.read_csv(file_path, encoding=enc, engine='python', on_bad_lines='skip')
            if df is not None and not df.empty:
                return filename, df
        except:
            pass
    return filename, None


def filter_to_jsonl(base_path: str, mask_file: str, output_jsonl: str,
                    model_name: str, device: str, mode: str = None):
    """
    each line: {"id": basename, "contents": processed text}
    mode: str, tr, tr_str, or None for base version
    """
    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)
    model = SentenceTransformer(model_name, device=device)
    model.eval()

    # read all mask entries
    with open(mask_file, 'r', encoding='utf-8') as f:
        entries = [line.strip() for line in f if line.strip()]

    written = 0
    with open(output_jsonl, 'w', encoding='utf-8') as fout:
        for rel_path in entries:
            # support absolute or relative base_path path
            if os.path.isabs(rel_path) or rel_path.startswith(base_path):
                csv_path = rel_path
            else:
                csv_path = os.path.join(base_path, rel_path)
            
            # Apply mode transformation if specified
            if mode and mode in ['str', 'tr', 'tr_str']:
                csv_path = apply_mode_transformation(csv_path, mode)
            
            if not os.path.exists(csv_path):
                print(f"Warning: file not found {csv_path}, skipping")
                continue
            basename = os.path.splitext(os.path.basename(csv_path))[0]
            _, df = safe_load_csv(csv_path)
            if df is None or df.empty:
                continue
            # row concatenation + character truncation
            df_str = df.astype(str)
            rows = df_str.agg(' '.join, axis=1).str.strip()
            rows = rows[rows.astype(bool)]
            text = ' '.join(rows.tolist())#[:1000]
            if not text:
                continue
            doc = {'id': basename, 'contents': text}
            fout.write(json.dumps(doc, ensure_ascii=False) + '\n')
            written += 1
    print(f'Generated JSONL: {output_jsonl}, entries written: {written}/{len(entries)}')


def apply_mode_transformation(csv_path: str, mode: str) -> str:
    """
    Transform CSV path based on mode:
    - str: add _str suffix to basename and look in *_str folders
    - tr: add _tr suffix to basename and look in *_tr folders  
    - tr_str: add _tr_str suffix to basename and look in *_tr_str folders
    """
    dir_path = os.path.dirname(csv_path)
    basename = os.path.basename(csv_path)
    name, ext = os.path.splitext(basename)
    
    # Add mode suffix to basename
    if mode == 'str':
        new_basename = f"{name}_s{ext}"
    elif mode == 'tr':
        new_basename = f"{name}_t{ext}"
    elif mode == 'tr_str':
        new_basename = f"{name}_s_t{ext}"
    else:
        return csv_path
    
    # Transform directory path to look in augmented folders
    if mode == 'str':
        dir_path = dir_path.replace('deduped_hugging_csvs', 'deduped_hugging_csvs_str')
        dir_path = dir_path.replace('deduped_github_csvs', 'deduped_github_csvs_str')
        dir_path = dir_path.replace('llm_tables', 'llm_tables_str')
        dir_path = dir_path.replace('tables_output', 'tables_output_str')
    elif mode == 'tr':
        dir_path = dir_path.replace('deduped_hugging_csvs', 'deduped_hugging_csvs_tr')
        dir_path = dir_path.replace('deduped_github_csvs', 'deduped_github_csvs_tr')
        dir_path = dir_path.replace('llm_tables', 'llm_tables_tr')
        dir_path = dir_path.replace('tables_output', 'tables_output_tr')
    elif mode == 'tr_str':
        dir_path = dir_path.replace('deduped_hugging_csvs', 'deduped_hugging_csvs_tr_str')
        dir_path = dir_path.replace('deduped_github_csvs', 'deduped_github_csvs_tr_str')
        dir_path = dir_path.replace('llm_tables', 'llm_tables_tr_str')
        dir_path = dir_path.replace('tables_output', 'tables_output_tr_str')
    
    return os.path.join(dir_path, new_basename)

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


def main():
    parser = argparse.ArgumentParser(
        description='Dense Retrieval Pipeline: filter->encode->build_faiss->search'
    )
    sub = parser.add_subparsers(dest='cmd')

    p = sub.add_parser('filter')
    p.add_argument('--base_path', required=True,
                help='data/processed root directory, containing subfolders')
    p.add_argument('--mask_file', required=True,
                   help='text list with relative or absolute CSV paths')
    p.add_argument('--output_jsonl', required=True)
    p.add_argument('--model_name', default='all-MiniLM-L6-v2')
    p.add_argument('--device', default='cuda')
    p.add_argument('--mode', choices=['str', 'tr', 'tr_str'], default=None,
                   help='augmentation mode: str, tr, tr_str, or None for base version')

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

    args = parser.parse_args()
    if args.cmd == 'filter':
        filter_to_jsonl(args.base_path, args.mask_file,
                        args.output_jsonl,
                        args.model_name, args.device, args.mode)
    elif args.cmd == 'encode':
        encode_corpus(args.jsonl, args.model_name,
                      args.batch_size, args.output_npz,
                      args.device)
    elif args.cmd == 'build_faiss':
        build_faiss(args.emb_npz, args.output_index)
    elif args.cmd == 'search':
        search_neighbors(args.emb_npz, args.faiss_index,
                         args.top_k, args.output_json)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
