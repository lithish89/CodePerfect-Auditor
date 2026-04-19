"""
ICD Semantic Search Engine
--------------------------
Encodes a clinical query using a sentence transformer and searches the
pre-built FAISS index to retrieve the most semantically similar ICD-10 codes.

Requires:
  - app/data/icd_index.faiss   (built by build_icd_index.py)
  - app/data/icd_data.pkl      (built by build_icd_index.py)

If these files don't exist yet, run:
    python build_icd_index.py
"""

from __future__ import annotations

import os
from typing import Optional

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Paths (configurable via env vars for containerised deployments)
# ---------------------------------------------------------------------------
INDEX_PATH = os.environ.get("ICD_INDEX_PATH", "app/data/icd_index.faiss")
DATA_PATH  = os.environ.get("ICD_DATA_PATH",  "app/data/icd_data.pkl")
MODEL_NAME = os.environ.get("ICD_MODEL_NAME", "all-MiniLM-L6-v2")

# ---------------------------------------------------------------------------
# Lazy singletons — loaded once on first call
# ---------------------------------------------------------------------------
_model: Optional[SentenceTransformer] = None
_index: Optional[faiss.Index]         = None
_icd_df: Optional[pd.DataFrame]       = None
_load_error: Optional[str]            = None


def _load_resources() -> None:
    """Load model, FAISS index, and ICD dataframe (once)."""
    global _model, _index, _icd_df, _load_error

    if _model is not None:
        return  # Already loaded
    if _load_error is not None:
        return  # Already failed — don't retry every call

    try:
        _model = SentenceTransformer(MODEL_NAME)
    except Exception as e:
        _load_error = f"Could not load SentenceTransformer '{MODEL_NAME}': {e}"
        return

    try:
        _index = faiss.read_index(INDEX_PATH)
    except Exception as e:
        _load_error = f"Could not load FAISS index at '{INDEX_PATH}': {e}. Run build_icd_index.py first."
        return

    try:
        _icd_df = pd.read_pickle(DATA_PATH)
    except Exception as e:
        _load_error = f"Could not load ICD data at '{DATA_PATH}': {e}. Run build_icd_index.py first."
        return


def search_icd(query: str, top_k: int = 3) -> list[dict]:
    """
    Search ICD-10 codes semantically for a clinical query string.

    Args:
        query:  Clinical text, e.g. "uncontrolled type 2 diabetes with neuropathy"
        top_k:  Number of results to return (default 3)

    Returns:
        List of dicts:
          [
            {
              "icd_code":    "E11.40",
              "description": "Type 2 diabetes mellitus with diabetic neuropathy, unspecified",
              "similarity":  0.87      # cosine similarity 0→1, HIGHER is better
            },
            ...
          ]
        Returns [] if the engine is not available.
    """
    _load_resources()

    if _load_error:
        print(f"[ICD Engine] {_load_error}")
        return []

    if not query or not query.strip():
        return []

    # Encode query
    query_embedding = _model.encode(
        [query.strip()],
        convert_to_numpy=True,
        normalize_embeddings=True,   # cosine via inner product on normalised vecs
    ).astype("float32")

    # FAISS search
    distances, indices = _index.search(query_embedding, top_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(_icd_df):
            continue  # FAISS returns -1 for padding
        row = _icd_df.iloc[idx]
        results.append({
            "icd_code":    str(row.get("Full Code", row.get("code", ""))).strip(),
            "description": str(row.get("Description", row.get("description", ""))).strip(),
            "similarity":  round(float(dist), 4),   # cosine similarity: higher = better match
        })

    return results


def is_available() -> bool:
    """Return True if the engine loaded successfully."""
    _load_resources()
    return _load_error is None


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    queries = [
        "patient with high blood pressure",
        "uncontrolled diabetes with kidney complications",
        "community acquired pneumonia",
        "acute chest pain radiating to left arm",
    ]
    print(f"Engine available: {is_available()}")
    if is_available():
        for q in queries:
            print(f"\nQuery: '{q}'")
            for r in search_icd(q, top_k=3):
                print(f"  [{r['icd_code']}] {r['description']}  (similarity={r['similarity']:.3f})")
    else:
        print("Engine not available — run build_icd_index.py to build the FAISS index.")