from rich import print

import numpy as np
from sentence_transformers import SentenceTransformer, SparseEncoder

from qdrant_client import QdrantClient, models
from constants import BATCH_SIZE, MODEL_NAME_DENSE, MODEL_NAME_SPARSE, DB_COLLECTION_NAME

db_client = QdrantClient(host="localhost", port=6333)

search_text = 'San Francisco'

model = SentenceTransformer(MODEL_NAME_DENSE)
model_sparse =  SparseEncoder(MODEL_NAME_SPARSE)

emb = model.encode(
    search_text,
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True,
).astype(np.float32)

emb_sparse = model_sparse.encode(
    search_text,
)

search_result = db_client.query_points(
    collection_name=DB_COLLECTION_NAME,
    prefetch=[
        models.Prefetch(
            query=models.SparseVector(
                indices=emb_sparse.coalesce().indices().tolist()[0],
                values=emb_sparse.coalesce().values().tolist()),
            using="sparse",
            limit=5,
        ),
        models.Prefetch(
            query=emb,
            using="dense",
            limit=5,
        ),
    ],
    query=models.FusionQuery(fusion=models.Fusion.RRF),
).points
print(search_result)