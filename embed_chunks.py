# embed_chunks.py
# Creates embeddings for chunks in JSONL and saves:
# 1) embeddings.npy  (float32 matrix)
# 2) embeddings.meta.jsonl (id + metadata + text pointers)

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from constants import MODEL_NAME_DENSE, BATCH_SIZE

from constants import DATA_DOC_NAME

CHUNKS_PATH = Path(f"data/{DATA_DOC_NAME}.chunks.jsonl")
OUT_EMB = Path(f"data/{DATA_DOC_NAME}.embeddings.npy")
OUT_META = Path(f"data/{DATA_DOC_NAME}.embeddings.meta.jsonl")

def main():
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(f"Missing {CHUNKS_PATH.resolve()}")

    model = SentenceTransformer(MODEL_NAME_DENSE)

    texts = []
    metas = []

    # Expect each line: {"id": "...", "text": "...", "metadata": {...}}  (or similar)
    with CHUNKS_PATH.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            obj = json.loads(line)
            text = obj.get("text") or obj.get("content")
            if not text:
                raise ValueError(f"Line {i}: missing 'text' (or 'content')")

            chunk_id = obj.get("id") or obj.get("chunk_id") or str(i)
            meta = obj.get("metadata") or {}
            meta_out = {
                "id": chunk_id,
                "metadata": meta,
                # keep small pointer fields if you want:
                "text_len": len(text),
            }

            texts.append(text)
            metas.append(meta_out)

    # Encode in batches; normalize_embeddings=True is handy for cosine similarity
    emb = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    np.save(OUT_EMB, emb)

    with OUT_META.open("w", encoding="utf-8") as f:
        for m in metas:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    print(f"Chunks: {len(texts)}")
    print(f"Embedding dim: {emb.shape[1]}")
    print(f"Saved: {OUT_EMB} and {OUT_META}")

if __name__ == "__main__":
    main()
