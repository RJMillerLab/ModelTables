import os
import argparse
import pickle
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, save_npz


def load_adjacency_dict(pickle_path):
    """
    Load an adjacency dictionary from a pickle file.
    The dictionary maps each CSV filename to a list of neighbor CSV filenames.
    """
    with open(pickle_path, 'rb') as f:
        return pickle.load(f)


def dict_to_boolean_csr(adj_dict):
    """
    Convert a CSV-to-CSV adjacency dictionary into a boolean CSR matrix and a CSV list.

    Returns:
      - M: a scipy.sparse.csr_matrix of shape (N, N), dtype=bool
      - csv_list: list of N unique CSV filenames in "first-seen" order
    """
    # Flatten keys and values preserving first-seen order
    flat = []
    for src, neighbors in adj_dict.items():
        flat.append(src)
        flat.extend(neighbors)
    # Remove duplicates while keeping order
    csv_list = list(dict.fromkeys(flat))

    # Build index mapping
    index = {name: idx for idx, name in enumerate(csv_list)}

    # Gather edges (undirected: add both directions)
    rows, cols = [], []
    for src, neighbors in adj_dict.items():
        i = index[src]
        for dst in neighbors:
            j = index[dst]
            rows.extend([i, j])
            cols.extend([j, i])

    data = np.ones(len(rows), dtype=bool)

    # Create COO then convert to CSR
    M = coo_matrix((data, (rows, cols)), shape=(len(csv_list), len(csv_list)), dtype=bool)
    M = M.tocsr()
    # Clear diagonal (no self-links)
    M.setdiag(False)
    return M, csv_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Convert a CSV adjacency pickle into a boolean CSR matrix + CSV list"
    )
    parser.add_argument(
        '--input', required=True,
        help="Path to the input pickle file (adjacency dict)"
    )
    parser.add_argument(
        '--output-prefix', required=True,
        help="Output prefix for the .npz and _csv_list.pkl files"
    )
    args = parser.parse_args()

    # Load the adjacency dictionary
    print(f"Loading adjacency dict from: {args.input}")
    adjacency = load_adjacency_dict(args.input)

    # Convert to boolean CSR and CSV list
    print("Converting to boolean CSR matrix + CSV list...")
    M, csv_list = dict_to_boolean_csr(adjacency)

    # Save the CSR matrix
    npz_path = f"{args.output_prefix}.npz"
    print(f"Saving boolean CSR matrix to: {npz_path}")
    save_npz(npz_path, M, compressed=True)

    # Save the CSV list
    list_path = f"{args.output_prefix}_csv_list.pkl"
    print(f"Saving CSV list to: {list_path}")
    with open(list_path, 'wb') as f:
        pickle.dump(csv_list, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("Done.")
    print(f"Matrix shape: {M.shape}, nnz: {M.nnz}")
    print(f"CSV list length: {len(csv_list)}")

