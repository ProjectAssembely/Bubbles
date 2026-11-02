"""Project 1, step 1: turn each occupation's text into one embedding vector.

Transformer models only read the first few hundred tokens, and an occupation's full
task list is often longer. So we embed each sentence on its own (the title + description,
and every task statement) and average them into one vector per occupation.

Run from the project root:
    .venv\\Scripts\\python src\\embed.py [model_name]

Outputs (in data/processed/):
    embeddings_<model>.npy        (n_occupations x dim), L2-normalized
    embeddings_<model>_codes.csv  occupation code for each row
"""
import sys
import time

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from load_onet import PROCESSED, load_occupations, load_table, CODE

DEFAULT_MODEL = "all-mpnet-base-v2"  # best general-purpose semantic similarity; use all-MiniLM-L6-v2 for speed


def build_sentences() -> pd.DataFrame:
    """One row per sentence to embed, tagged with its occupation code."""
    occ = load_occupations()
    tasks = load_table("Task Statements")[[CODE, "Task"]].rename(columns={CODE: "code", "Task": "text"})
    # Only keep occupations that have tasks (drops the 'All Other' catch-alls)
    occ = occ[occ["code"].isin(tasks["code"])]
    header = pd.DataFrame({"code": occ["code"], "text": occ["title"] + ". " + occ["description"]})
    return pd.concat([header, tasks], ignore_index=True)


def embed(model_name: str = DEFAULT_MODEL) -> tuple[np.ndarray, pd.Series]:
    sentences = build_sentences()
    print(f"Embedding {len(sentences):,} sentences from {sentences['code'].nunique()} occupations "
          f"with {model_name}...")
    # Cache sentence-level vectors so later steps never need to re-run the slow part
    cache = PROCESSED / f"sentence_embeddings_{model_name.split('/')[-1]}.npy"
    if cache.exists() and len(np.load(cache, mmap_mode="r")) == len(sentences):
        vecs = np.load(cache)
        print(f"Loaded cached sentence vectors from {cache.name}")
    else:
        model = SentenceTransformer(model_name)
        t0 = time.time()
        vecs = model.encode(sentences["text"].tolist(), batch_size=64, show_progress_bar=True,
                            normalize_embeddings=True)
        print(f"Done in {time.time() - t0:.0f}s")
        np.save(cache, vecs)

    # Average sentence vectors per occupation, then re-normalize to unit length
    df = pd.DataFrame(vecs)
    df["code"] = sentences["code"].values
    occ_vecs = df.groupby("code", sort=True).mean()
    X = occ_vecs.to_numpy(dtype=np.float32)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    return X, occ_vecs.index.to_series(name="code")


def main() -> None:
    model_name = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    X, codes = embed(model_name)
    tag = model_name.split("/")[-1]
    np.save(PROCESSED / f"embeddings_{tag}.npy", X)
    codes.to_csv(PROCESSED / f"embeddings_{tag}_codes.csv", index=False)
    print(f"Saved {X.shape} to {PROCESSED / f'embeddings_{tag}.npy'}")


if __name__ == "__main__":
    main()
