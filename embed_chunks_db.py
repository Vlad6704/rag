# embed_chunks_db.py
# Creates embeddings for chunks in JSONL and saves in vector DB:
# 1) embeddings.npy  (float32 matrix)
# 2) embeddings.meta.jsonl (id + metadata + text pointers)

from rich import print

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer, SparseEncoder

from qdrant_client import QdrantClient, models

from constants import DATA_DOC_NAME, MODEL_NAME_DENSE, MODEL_NAME_SPARSE, DB_COLLECTION_NAME

CHUNKS_PATH = Path(f"data/{DATA_DOC_NAME}.chunks.jsonl")
OUT_EMB = Path(f"data/{DATA_DOC_NAME}.embeddings.npy")
OUT_META = Path(f"data/{DATA_DOC_NAME}.embeddings.meta.jsonl")




BATCH_SIZE = 64


def main():

    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(f"Missing {CHUNKS_PATH.resolve()}")

    db_client = QdrantClient(host="localhost", port=6333)

    model_dense = SentenceTransformer(MODEL_NAME_DENSE)
    model_sparse =  SparseEncoder(MODEL_NAME_SPARSE)

    data = []

    # Expect each line: {"id": "...", "text": "...", "metadata": {...}}  (or similar)
    with CHUNKS_PATH.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            obj = json.loads(line)
            text = obj.get("text") or obj.get("content")
            if not text:
                raise ValueError(f"Line {i}: missing 'text' (or 'content')")

            meta = obj.get("meta") or {}

            data.append([text, {**meta, "text": text}])

    # Encode in batches; normalize_embeddings=True is handy for cosine similarity
    emb_dense = model_dense.encode(
        [el[0] for el in data],
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    emb_sparse = model_sparse.encode(
        [el[0] for el in data],
    )





    points = []
    for i, el in enumerate(data):
        points.append(
            models.PointStruct(
                id=i,
                payload=el[1],
                vector={
                    "dense": emb_dense[i],
                    "sparse": models.SparseVector(
                        indices=emb_sparse[i].coalesce().indices().tolist()[0],
                        values=emb_sparse[i].coalesce().values().tolist(),
                    )
                }
            )
        )

    db_client.upsert(
        collection_name=DB_COLLECTION_NAME,
        points=points,
    )


    # print(points)
    print(db_client.get_collection(DB_COLLECTION_NAME))



if __name__ == "__main__":
    main()
