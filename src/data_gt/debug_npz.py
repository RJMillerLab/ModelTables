import argparse
import pickle
import numpy as np
from scipy.sparse import load_npz

def inspect_npz(matrix_path, csvlist_path, row_idx):
    print(f"🔍 Loading matrix from: {matrix_path}")
    M = load_npz(matrix_path).tocsr()
    print(f"  ▸ type(M): {type(M)}")
    print(f"  ▸ shape: {M.shape!s}")
    print(f"  ▸ nnz (nonzeros): {M.nnz}")
    print(f"  ▸ M.dtype: {M.dtype}")
    print(f"  ▸ M.data.dtype: {M.data.dtype}")
    print(f"  ▸ len(indptr): {len(M.indptr)}, len(indices): {len(M.indices)}, len(data): {len(M.data)}")

    # 单行检查
    if 0 <= row_idx < M.shape[0]:
        start, end = M.indptr[row_idx], M.indptr[row_idx+1]
        cols = M.indices[start:end]
        vals = M.data[start:end]
        print(f"\nNumber of non-zero elements in row {row_idx}: {end-start}")
        print(f"  ▸ cols sample: {cols[:10].tolist()}")
        print(f"  ▸ vals sample: {vals[:10].tolist()}")
        print(f"  ▸ cols.dtype: {cols.dtype}, vals.dtype: {vals.dtype}")
    else:
        print(f"\n⚠️ Row index {row_idx} out of bounds")

    # CSV 列表
    with open(csvlist_path, "rb") as f:
        csv_list = pickle.load(f)
    print(f"\n📄 CSV list: len = {len(csv_list)}, type = {type(csv_list[0]) if csv_list else 'N/A'}")
    print(f"  ▸ sample entries: {csv_list[:5]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Debug & inspect a compressed CSR matrix (.npz) and its csv_list"
    )
    parser.add_argument("--matrix",   required=True, help="Path to the .npz sparse matrix")
    parser.add_argument("--csvlist",  required=True, help="Path to csv_list_*.pkl")
    parser.add_argument("--row",      type=int, default=0, help="Which row to check (0-based)")
    args = parser.parse_args()

    inspect_npz(args.matrix, args.csvlist, args.row)

