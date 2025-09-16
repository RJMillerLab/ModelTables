import json
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer


def encode_texts(
    texts: Iterable[str],
    model_name: str = "sentence-transformers/all-mpnet-base-v2",
    batch_size: int = 32,
    device: str | None = None,
) -> np.ndarray:
    """Encode an iterable of texts into a numpy array using SBERT.

    Parameters
    ----------
    texts : Iterable[str]
        Texts to encode.
    model_name : str, optional
        HuggingFace model name, by default "sentence-transformers/all-mpnet-base-v2".
    batch_size : int, optional
        Batch size, by default 32.
    device : str | None, optional
        Torch device. If None, use default device.

    Returns
    -------
    np.ndarray
        Array of shape (n_texts, dim).
    """
    model = SentenceTransformer(model_name, device=device)
    texts_list = list(texts)
    arr = model.encode(
        texts_list,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    return arr


def encode_jsonl(
    jsonl_path: str | Path,
    output_npy: str | Path,
    text_key: str = "contents",
    **kwargs,
):
    """Encode a jsonl corpus file and save embeddings to .npy.

    Each line of jsonl should be a dict with at least the `text_key` field.
    """
    jsonl_path = Path(jsonl_path)
    output_npy = Path(output_npy)
    with jsonl_path.open() as f:
        texts = [json.loads(line)[text_key] for line in f]
    emb = encode_texts(texts, **kwargs)
    np.save(output_npy, emb)
    print(f"Saved embeddings to {output_npy}")
