"""
build_icd_index.py
------------------
One-time script to build the FAISS semantic search index from ICD10_codes.csv.

Run this before starting the API server:
    python build_icd_index.py

Output:
    app/data/icd_index.faiss   — FAISS flat L2 index
    app/data/icd_data.pkl      — pickled DataFrame of ICD codes and descriptions

Requirements:
    pip install faiss-cpu pandas sentence-transformers openpyxl
"""

from __future__ import annotations

import os
import sys
import pickle
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CSV_PATH   = "app/data/ICD10_codes.csv"
INDEX_PATH = "app/data/icd_index.faiss"
DATA_PATH  = "app/data/icd_data.pkl"
MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 256   # Encode this many descriptions at a time

# Column names in ICD10_codes.csv (adjust if yours differ)
CODE_COLUMN        = "Full Code"
DESCRIPTION_COLUMN = "Description"


def build_index() -> None:
    # ---- Load CSV -----------------------------------------------------------
    print(f"[1/4] Loading ICD-10 codes from '{CSV_PATH}' ...")
    if not os.path.exists(CSV_PATH):
        print(f"  ERROR: '{CSV_PATH}' not found.")
        print("  Place ICD10_codes.csv in app/data/ and re-run.")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH, dtype=str, encoding="utf-8", on_bad_lines="skip")
    df.columns = [c.strip() for c in df.columns]

    # Detect column names flexibly
    code_col = next(
        (c for c in df.columns if "code" in c.lower() and "full" in c.lower()),
        next((c for c in df.columns if "code" in c.lower()), None),
    )
    desc_col = next(
        (c for c in df.columns if "descrip" in c.lower()),
        None,
    )

    if code_col is None or desc_col is None:
        print(f"  ERROR: Could not find code/description columns. Columns found: {list(df.columns)}")
        sys.exit(1)

    print(f"  Detected columns: code='{code_col}', description='{desc_col}'")

    df = df[[code_col, desc_col]].dropna()
    df = df.rename(columns={code_col: "Full Code", desc_col: "Description"})
    df = df.reset_index(drop=True)
    print(f"  Loaded {len(df):,} ICD-10 entries.")

    # ---- Load model ---------------------------------------------------------
    print(f"[2/4] Loading sentence transformer model '{MODEL_NAME}' ...")
    model = SentenceTransformer(MODEL_NAME)

    # ---- Encode descriptions ------------------------------------------------
    print(f"[3/4] Encoding descriptions (batch_size={BATCH_SIZE}) ...")
    descriptions = df["Description"].tolist()
    embeddings = model.encode(
        descriptions,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    print(f"  Embedding shape: {embeddings.shape}")

    # ---- Build FAISS index --------------------------------------------------
    print(f"[4/4] Building FAISS IndexFlatIP (inner product on normalised vecs = cosine) ...")
    dimension = embeddings.shape[1]
    # IndexFlatIP + normalised embeddings = cosine similarity
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    os.makedirs("app/data", exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    df.to_pickle(DATA_PATH)

    print(f"\n  FAISS index saved to '{INDEX_PATH}'  ({index.ntotal:,} vectors)")
    print(f"  ICD data saved to  '{DATA_PATH}'")
    print("\nDone! You can now start the API server with:")
    print("  uvicorn main:app --reload")


if __name__ == "__main__":
    build_index()