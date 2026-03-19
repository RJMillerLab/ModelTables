"""
Dense Retrieval pipeline, based on Sentence-BERT to encode table text, and use FAISS for nearest neighbor search.
Subcommands:
  1. encode       : stream CSV->text->SBERT encode and save NPZ
  2. search       : search all embeddings, and output neighbor mapping JSON
Usage:
# build corpus
python src/baseline1/table_retrieval_pipeline.py encode --base_path /u501/z6dong/Repo/ModelTables --mask_file /u501/z6dong/Repo/ModelTables/data/analysis/all_valid_title_valid_v2_251117.txt --model_name all-MiniLM-L6-v2 --batch_size 512 --output_npz /u501/z6dong/Repo/ModelTables/data/baseline1_251117/valid_tables_v2_251117_embeddings.npz

python src/baseline1/table_retrieval_pipeline.py search --emb_npz /u501/z6dong/Repo/ModelTables/data/baseline1_251117/valid_tables_v2_251117_embeddings.npz --top_k 5 --output_json /u501/z6dong/Repo/ModelTables/data/baseline1_251117/table_neighbors_v2_251117.json

python src/baseline1/table_retrieval_pipeline.py postprocess --input_json /u501/z6dong/Repo/ModelTables/data/baseline1_251117/table_neighbors_v2_251117.json --output_json /u501/z6dong/Repo/ModelTables/data/baseline1_251117/table_neighbors_v2_251117_processed.json
"""
import os
import json
import argparse
import numpy as np
from tqdm import tqdm
import faiss
from sentence_transformers import SentenceTransformer
import pandas as pd
from typing import List
import torch
import time


def safe_load_csv(file_path: str) -> pd.DataFrame:
    encodings = ['utf-8', 'latin1', 'cp1252']
    for enc in encodings:
        try:
            df = pd.read_csv(file_path, encoding=enc, low_memory=False)
            if df is not None and not df.empty:
                return df
        except:
            pass
        try:
            df = pd.read_csv(file_path, encoding=enc, engine='python', on_bad_lines='skip')
            if df is not None and not df.empty:
                return df
        except:
            pass
    return None


def encode_corpus_from_mask(base_path: str, mask_file: str, model_name: str, batch_size: int, output_npz: str, mode: str = None):
    """
    Stream CSV->text->SBERT encode in batches, and save FAISS index + ids sidecar.

    This avoids materializing all texts (and avoids the intermediate JSONL pass).
    """
    os.makedirs(os.path.dirname(output_npz), exist_ok=True)

    # SentenceTransformer model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer(model_name, device=device)
    model.eval()

    # read all mask entries
    with open(mask_file, 'r', encoding='utf-8') as f:
        entries = [line.strip() for line in f if line.strip()]

    # apply mode transformation if specified
    if mode != 'base':
        entries = [apply_mode_transformation(entry, mode) for entry in entries]
        print('applied mode transformation, add str/tr/tr_str/base to path')
        print(len(entries), ': num of entries')
        print(entries[0])

    # apply base path -> full path
    entries = [os.path.join(base_path, entry) for entry in entries]
    print('processed basepath->fullpath')
    print(len(entries), ': num of entries')
    print(entries[0])

    all_embs = []
    ids: List[str] = []

    batch_texts: List[str] = []
    batch_ids: List[str] = []
    written = 0

    for csv_path in tqdm(entries, desc='Encoding', unit='file'):
        basename = os.path.splitext(os.path.basename(csv_path))[0]
        df = safe_load_csv(csv_path)
        if df is None or df.empty:
            print(f"Warning: failed to load CSV {csv_path} due to empty or None, skipping")
            continue

        # row concatenation + character truncation
        df_str = df.astype(str)
        rows = df_str.agg(' '.join, axis=1).str.strip()
        rows = rows[rows.astype(bool)]
        text = ' '.join(rows.tolist())
        if not text:
            print(f"Warning: failed to generate text from CSV {csv_path} due to empty text, skipping")
            continue

        batch_ids.append(basename)
        batch_texts.append(text)

        # When batch is ready, encode immediately.
        if len(batch_texts) == batch_size:
            try:
                embs = model.encode(batch_texts, convert_to_numpy=True, show_progress_bar=False, batch_size=len(batch_texts))
                if getattr(embs, "size", 0) > 0:
                    all_embs.append(embs)
                    ids.extend(batch_ids)
                    written += len(batch_ids)
            except Exception as e:
                print(f'Error encoding batch ending at csv={csv_path}: {e}')
            finally:
                batch_texts = []
                batch_ids = []

    # Flush last partial batch
    if batch_texts:
        try:
            embs = model.encode(batch_texts, convert_to_numpy=True, show_progress_bar=False, batch_size=len(batch_texts))
            if getattr(embs, "size", 0) > 0:
                all_embs.append(embs)
                ids.extend(batch_ids)
                written += len(batch_ids)
        except Exception as e:
            print(f'Error encoding final batch: {e}')

    if not all_embs:
        print('No embeddings generated, skipping save.')
        return

    embs_array = np.vstack(all_embs).astype('float32')
    np.savez_compressed(output_npz, embeddings=embs_array, ids=np.array(ids))
    print(f'Saved embeddings: {output_npz}, shape={embs_array.shape}, vectors={written}')


def apply_mode_transformation(csv_path: str, mode: str) -> str:
    """
    Transform CSV path based on mode:
    - str: add _str suffix to basename and look in *_str folders
    - tr: add _tr suffix to basename and look in *_tr folders
    - tr_str: add _tr_str suffix to basename and look in *_tr_str folders
    """
    # example:
    #   data/processed/deduped_github_csvs_v2_251117/xxxx.csv
    #   -> data/processed/deduped_github_csvs_v2_251117_tr/xxxx_t.csv
    #
    # General rule:
    #   - file basename: append a mode-specific suffix before extension
    #   - parent folder name: append mode-specific suffix at the end
    MODE_MAP = {
        "str": ("_s", "_str"),
        "tr": ("_t", "_tr"),
        "tr_str": ("_s_t", "_tr_str"),
        "base": ("", ""),
    }
    if mode not in MODE_MAP:
        return csv_path
    file_suffix, dir_suffix = MODE_MAP[mode]
    dir_path = os.path.dirname(csv_path)
    basename = os.path.basename(csv_path)
    name, ext = os.path.splitext(basename)
    new_basename = f"{name}{file_suffix}{ext}"
    dir_name = os.path.basename(dir_path)
    new_dir_name = dir_name if dir_name.endswith(dir_suffix) else dir_name + dir_suffix
    parent_dir = os.path.dirname(dir_path)
    if parent_dir in ("", "."):
        new_dir_path = new_dir_name
    else:
        new_dir_path = os.path.join(parent_dir, new_dir_name)
    return os.path.join(new_dir_path, new_basename)

def build_faiss(embs, ids):
    """
    Build FAISS index for inner product retrieval, and save to disk.
    """
    #data = np.load(emb_npz, allow_pickle=True)
    #embs = data['embeddings']
    #ids = data['ids'] if 'ids' in data else None
    faiss.normalize_L2(embs)
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embs)
    '''os.makedirs(os.path.dirname(output_index), exist_ok=True)
    faiss.write_index(index, output_index)
    print(f'Saved FAISS index: {output_index}, vectors={index.ntotal}')

    # Save ids order as a tiny sidecar so search doesn't need emb_npz.
    if ids is not None:
        ids_path = output_index + '.ids.npy'
        np.save(ids_path, ids)
        print(f'Saved ids sidecar: {ids_path}, ids={len(ids)}')'''
    return index

def search_neighbors(emb_npz: str, top_k: int, output_json: str):
    """
    Search all embeddings, remove self, save neighbor mapping.
    """
    # build index
    data = np.load(emb_npz)
    embs = np.asarray(data['embeddings'], dtype='float32')
    ids = data['ids'].tolist()
    print('Building FAISS index')
    t1 = time.time()
    index = build_faiss(embs, ids)
    print('Time taken to build FAISS index: ', time.time() - t1, 'seconds')

    print('Searching for neighbors')
    t1 = time.time()
    D, I = index.search(embs, top_k+1)
    print('Time taken to search: ', time.time() - t1, 'seconds')

    print('Postprocessing neighbors')
    t1 = time.time()
    neighbors = {}
    for i, neigh in enumerate(tqdm(I, desc='Postprocessing neighbors', unit='vec')):
        base = ids[i]
        nb = [ids[j] for j in neigh if j != i][:top_k]
        neighbors[base] = nb
    print('Time taken to postprocess neighbors: ', time.time() - t1, 'seconds')

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, 'w', encoding='utf-8') as fout:
        json.dump(neighbors, fout, ensure_ascii=False, indent=2)
    print(f'Saved neighbor mapping: {output_json}')

def append_csv_suffix(results: dict):
    """
    Transform neighbor mapping JSON:
    {"table_id": ["neighbor_id1", ...], ...}
    -> {"table_id.csv": ["neighbor_id1.csv", ...], ...}
    """
    new_results = {
        key + '.csv': [neighbor + '.csv' for neighbor in neighbors]
        for key, neighbors in results.items()
    }
    return new_results

def remove_self_hit(results: dict):
    """Remove self-hit from results."""
    # simplify as one line
    new_results = {query_id: hits[1:] if hits[0] == query_id else hits for query_id, hits in results.items()}
    return new_results

def main():
    parser = argparse.ArgumentParser(description='Dense Retrieval Pipeline: encode->build_faiss->search')
    sub = parser.add_subparsers(dest='cmd')

    e = sub.add_parser('encode')
    e.add_argument('--base_path', required=True, help='/u501/z6dong/Repo/ModelTables root directory, containing subfolders')
    e.add_argument('--mask_file', required=True, help='text list with relative or absolute CSV paths')
    e.add_argument('--model_name', default='all-MiniLM-L6-v2')
    e.add_argument('--batch_size', type=int, default=256, help='encoding batch size')
    e.add_argument('--output_npz', required=True, help='output embeddings npz path, e.g. data/baseline/*_embeddings.npz')
    e.add_argument('--mode', choices=['str', 'tr', 'tr_str', 'base'], default='base', help='augmentation mode: str, tr, tr_str, or base for base version')

    s = sub.add_parser('search')
    s.add_argument('--emb_npz', required=True, help='embeddings npz generated by encode')
    s.add_argument('--top_k', type=int, default=5)
    s.add_argument('--output_json', required=True)

    p = sub.add_parser('postprocess')
    p.add_argument('--input_json', required=True, help='input neighbor mapping json')

    args = parser.parse_args()
    if args.cmd == 'encode':
        encode_corpus_from_mask(base_path=args.base_path, mask_file=args.mask_file, model_name=args.model_name, batch_size=args.batch_size, output_npz=args.output_npz, mode=args.mode)
    elif args.cmd == 'search':
        search_neighbors(args.emb_npz, args.top_k, args.output_json)
    elif args.cmd == 'postprocess':
        with open(args.input_json, 'r', encoding='utf-8') as f:
            results = json.load(f)
        results = append_csv_suffix(results)
        results = remove_self_hit(results)
        with open(args.input_json, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f'Saved postprocessed json: {args.input_json}')
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
