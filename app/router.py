import os
from concurrent.futures import ThreadPoolExecutor

import json
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

app = FastAPI()
NODES = [u for u in (os.getenv("NODE1_URL"), os.getenv("NODE2_URL"), os.getenv("NODE3_URL")) if u]
MODEL = SentenceTransformer("all-MiniLM-L6-v2")

TEXTS = json.load(open("agnews_20k_texts.json"))
_S = len(TEXTS) // 3
OFFSETS = [0, _S, 2 * _S]

class SearchRequest(BaseModel):
    text: str
    k: int = 5


def query_node(shard, url, vec, k):
    try:
        r = requests.post(url, json={"query": vec, "k": k}, timeout=5)
        r.raise_for_status()
        d = r.json()
        return [{"shard": shard, "local_id": int(i), "distance": float(x),
                          "text": TEXTS[OFFSETS[shard] + int(i)]}
                        for i, x in zip(d["indices"][0], d["distances"][0]) if i != -1]
    except Exception as e:
        print(f"shard {shard} failed: {e}", flush=True)
        return []


@app.post("/search")
def search(req: SearchRequest):
    vec = MODEL.encode(req.text).tolist()
    with ThreadPoolExecutor(max_workers=3) as ex:
        outs = list(ex.map(lambda a: query_node(a[0], a[1], vec, req.k), enumerate(NODES)))
    results = sorted([r for o in outs for r in o], key=lambda r: r["distance"])
    return {"results": results[: req.k]}
