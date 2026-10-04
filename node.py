from fastapi import FastAPI
import faiss
import numpy as np
import boto3
import os

app = FastAPI()

BUCKET = os.getenv("S3_BUCKET")
SHARD_KEY = os.getenv("SHARD_KEY")   # shard_0.npy, shard_1.npy, shard_2.npy

# Download shard from S3
s3 = boto3.client("s3")
s3.download_file(BUCKET, SHARD_KEY, "shard.npy")

# Load vectors
vectors = np.load("shard.npy").astype("float32")

# Build FAISS index
index = faiss.IndexFlatL2(vectors.shape[1])
index.add(vectors)

@app.post("/search_local")
def search_local(query: list, k: int = 10):
    q = np.array(query).astype("float32").reshape(1, -1)
    D, I = index.search(q, k)
    return {"distances": D.tolist(), "indices": I.tolist()}
