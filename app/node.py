from fastapi import FastAPI
from pydantic import BaseModel
import faiss
import numpy as np
import boto3
import os

app = FastAPI()

BUCKET = os.getenv("S3_BUCKET")
SHARD_KEY = os.getenv("SHARD_KEY")

s3 = boto3.client("s3")
s3.download_file(BUCKET, SHARD_KEY, "shard.npy")

vectors = np.load("shard.npy").astype("float32")

index = faiss.IndexFlatL2(vectors.shape[1])
index.add(vectors)


class LocalSearch(BaseModel):
    query: list[float]
    k: int = 10


@app.post("/search_local")
def search_local(req: LocalSearch):
    q = np.array(req.query, dtype="float32").reshape(1, -1)
    D, I = index.search(q, req.k)
    return {"distances": D.tolist(), "indices": I.tolist()}
