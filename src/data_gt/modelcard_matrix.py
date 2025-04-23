# build_model_relation.py

import re                                                        ########
import pandas as pd
import numpy as np
import pickle
from scipy.sparse import dok_matrix, csr_matrix, save_npz
import os                                                       ########

MODEL_REL_PATH      = "data/tmp/model_relation_adjacency.npz"
MODEL_REL_INDEXPATH = "data/tmp/model_relation_index.pickle"

def build_model_relation_matrix(
    df: pd.DataFrame,
    src_col: str = "modelId",
    tgt_col: str = "extracted_base_model",
    binary: bool = True
) -> (csr_matrix, list):
    df_rel = df.dropna(subset=[tgt_col]).copy()

    all_models = set(df_rel[src_col]) | set(df_rel[tgt_col])
    model_index = sorted(all_models)
    pos = {m:i for i,m in enumerate(model_index)}

    M = len(model_index)
    mat = dok_matrix((M, M), dtype=np.int8)

    for _, row in df_rel.iterrows():
        i = pos[row[src_col]]
        j = pos[row[tgt_col]]
        if binary:
            mat[i, j] = 1
        else:
            mat[i, j] += 1

    for i in range(M):
        mat[i, i] = 1

    return mat.tocsr(), model_index


if __name__ == "__main__":
    df = pd.read_parquet("data/processed/modelcard_step3_merged.parquet")

    mask = df['card_tags_x'].str.contains(r'(?i)base_model', na=False)          ########
    df = df[mask].copy()                                                         ########
    df['extracted_base_model'] = df['card_tags_x'] \
        .str.extract(r'base_model:\s*([^\s]+)', flags=re.IGNORECASE)            ########
    df['extracted_relation'] = df['card_tags_x'] \
        .str.extract(r'base_model_relation:\s*([^\s]+)', flags=re.IGNORECASE)   ########

    os.makedirs("data/tmp", exist_ok=True)                                        ########

    adj, index = build_model_relation_matrix(
        df,
        src_col="modelId",
        tgt_col="extracted_base_model",
        binary=True
    )                                                                             ########

    save_npz(MODEL_REL_PATH, adj)
    with open(MODEL_REL_INDEXPATH, "wb") as f:
        pickle.dump(index, f)

    print(f"✔️  Saved model-level adjacency to {MODEL_REL_PATH}")
    print(f"✔️  Saved model index list to {MODEL_REL_INDEXPATH}")

