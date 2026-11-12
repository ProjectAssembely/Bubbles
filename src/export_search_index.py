"""Export a compact search index so the site can do meaning-based search in the browser.

GitHub Pages can't run the Python search server (src/serve.py), so the browser runs a small model
itself (all-MiniLM-L6-v2 via transformers.js, ~23 MB, downloaded once) and compares the query with
the job vectors exported here. Scoring mirrors src/search.py: 60% full job profile + 40% title.

Run from the project root, after `python src/embed.py all-MiniLM-L6-v2`:
    .venv\\Scripts\\python src\\export_search_index.py            # export
    .venv\\Scripts\\python src\\export_search_index.py calibrate  # print scores to choose MIN_SCORE

Outputs:
    web/data/search_index.json  model name, settings and job codes (row order)
    web/data/search_index.bin   int8 vectors: profiles (n x dim) followed by titles (n x dim)
"""
import json
import sys

import numpy as np
import pandas as pd

from load_onet import PROCESSED, ROOT
from text_clustering import load_embeddings

PY_MODEL = "all-MiniLM-L6-v2"              # sentence-transformers name (Python side)
JS_MODEL = "Xenova/all-MiniLM-L6-v2"       # same model converted to ONNX for transformers.js
PROFILE_WEIGHT = 0.6                       # same blend as src/search.py
# Calibrated with `calibrate` below (MiniLM scores run lower than mpnet's): see the printed table.
MIN_SCORE = 0.30
SCALE = 127                                # unit vectors -> int8

QUERIES = [
    "I like animals", "fixing cars", "helping sick people", "working outdoors", "math and numbers",
    "cooking", "computers", "I want to draw", "teaching kids", "law", "money", "pizza", "space",
    "delivering packages", "driving a bus",
    "asdfghjkl", "qwerty zxcv", "the", "banana", "my favorite color is blue", "lorem ipsum dolor sit amet",
    "2 + 2", "hello", "purple elephants dancing",
]


def load_index():
    from sentence_transformers import SentenceTransformer

    X, codes = load_embeddings(PY_MODEL)
    titles = pd.read_csv(PROCESSED / "hierarchy.csv", index_col=0).reindex(codes)["title"]
    model = SentenceTransformer(PY_MODEL)
    T = model.encode(titles.tolist(), normalize_embeddings=True, batch_size=64)
    return model, X, T, codes, titles


def calibrate() -> None:
    model, X, T, codes, titles = load_index()
    for q in QUERIES:
        v = model.encode([q], normalize_embeddings=True)[0]
        s = PROFILE_WEIGHT * (X @ v) + (1 - PROFILE_WEIGHT) * (T @ v)
        top = np.argsort(s)[::-1][:3]
        print(f"{s[top[0]]:.3f}  {q:28} -> " + " | ".join(titles.iloc[i][:28] for i in top))


def export() -> None:
    _, X, T, codes, _ = load_index()
    out = ROOT / "web" / "data"
    vectors = np.vstack([X, T])
    np.clip(np.round(vectors * SCALE), -127, 127).astype(np.int8).tofile(out / "search_index.bin")
    meta = {"model": JS_MODEL, "dim": int(X.shape[1]), "count": int(len(codes)), "scale": SCALE,
            "profileWeight": PROFILE_WEIGHT, "minScore": MIN_SCORE, "codes": codes.tolist()}
    (out / "search_index.json").write_text(json.dumps(meta), encoding="utf-8")
    size = (out / "search_index.bin").stat().st_size / 1e6
    print(f"Exported {len(codes)} jobs x {X.shape[1]} dims ({size:.2f} MB) to {out}")


if __name__ == "__main__":
    calibrate() if sys.argv[1:] == ["calibrate"] else export()
