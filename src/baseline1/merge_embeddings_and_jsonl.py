import numpy as np
import json
import sys

def merge_npz(npz_files, out_npz):
    all_embs = []
    all_ids = []
    for npz in npz_files:
        data = np.load(npz)
        all_embs.append(data['embeddings'])
        all_ids.extend(data['ids'].tolist())
    merged_embs = np.vstack(all_embs)
    np.savez_compressed(out_npz, embeddings=merged_embs, ids=np.array(all_ids))
    print(f"Saved merged npz: {out_npz}, shape={merged_embs.shape}")

def merge_jsonl(jsonl_files, out_jsonl):
    with open(out_jsonl, 'w', encoding='utf-8') as fout:
        for jf in jsonl_files:
            with open(jf, 'r', encoding='utf-8') as fin:
                for line in fin:
                    fout.write(line)
    print(f"Saved merged jsonl: {out_jsonl}")

if __name__ == "__main__":
    # usage: python merge_embeddings_and_jsonl.py npz1 npz2 ... --out_npz out.npz jsonl1 jsonl2 ... --out_jsonl out.jsonl
    npz_files = []
    jsonl_files = []
    out_npz = None
    out_jsonl = None
    mode = 'npz'
    for arg in sys.argv[1:]:
        if arg == '--out_npz':
            mode = 'out_npz'
        elif arg == '--out_jsonl':
            mode = 'out_jsonl'
        elif mode == 'npz':
            npz_files.append(arg)
        elif mode == 'out_npz':
            out_npz = arg
            mode = 'jsonl'
        elif mode == 'jsonl':
            jsonl_files.append(arg)
        elif mode == 'out_jsonl':
            out_jsonl = arg
    assert out_npz and out_jsonl
    merge_npz(npz_files, out_npz)
    merge_jsonl(jsonl_files, out_jsonl)
