import numpy as np
import argparse

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

def main():
    parser = argparse.ArgumentParser(description="Merge multiple embeddings .npz files.")
    parser.add_argument("--v2_mode", action="store_true", help="Use v2 mode.")
    parser.add_argument("--tag", type=str, default=None, help="Tag suffix for versioning (e.g., 251117).")
    args = parser.parse_args()
    v2_suffix = "_v2" if args.v2_mode else ""
    suffix = f"_{args.tag}" if args.tag else ""
    # merge base+tr, base+str, base+tr+str
    merge_npz([f'data/baseline1{suffix}/valid_tables{v2_suffix}{suffix}_embeddings.npz', f'data/baseline1{suffix}/valid_tables_tr{v2_suffix}{suffix}_embeddings.npz'], f'data/baseline1{suffix}/valid_tables_ori_tr{v2_suffix}{suffix}_embeddings.npz')
    merge_npz([f'data/baseline1{suffix}/valid_tables{v2_suffix}{suffix}_embeddings.npz', f'data/baseline1{suffix}/valid_tables_str{v2_suffix}{suffix}_embeddings.npz'], f'data/baseline1{suffix}/valid_tables_ori_str{v2_suffix}{suffix}_embeddings.npz')
    merge_npz([f'data/baseline1{suffix}/valid_tables{v2_suffix}{suffix}_embeddings.npz', f'data/baseline1{suffix}/valid_tables_tr{v2_suffix}{suffix}_embeddings.npz', f'data/baseline1{suffix}/valid_tables_str{v2_suffix}{suffix}_embeddings.npz'], f'data/baseline1{suffix}/valid_tables_mixed{v2_suffix}{suffix}_embeddings.npz')

if __name__ == "__main__":
    main()