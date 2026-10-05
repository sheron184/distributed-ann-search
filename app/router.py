from fastapi import FastAPI
import requests
import numpy as np
import os

app = FastAPI()

NODE1 = os.getenv("NODE1_URL")
NODE2 = os.getenv("NODE2_URL")
NODE3 = os.getenv("NODE3_URL")

NODES = [NODE1, NODE2, NODE3]

@app.post("/search")
def search(query: list, k: int = 10):
    all_results = []

    for url in NODES:
        r = requests.post(url, json={"query": query, "k": k})
        data = r.json()
        for idx, dist in zip(data["indices"][0], data["distances"][0]):
            all_results.append((idx, dist))

    # Sort by distance
    all_results.sort(key=lambda x: x[1])

    # Take global top-k
    final = all_results[:k]

    return {"results": final}
