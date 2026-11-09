"""Semantic job search: find the jobs closest in meaning to any free-text query.

The query is embedded with the same model as the jobs (see embed.py), so "I like animals"
finds animal trainers and caretakers even though no title contains those words.

Each job is scored against two vectors:
  - its full profile (title + description + tasks, from embed.py): captures what the work involves
  - its title alone: short queries like "computers" or "space" often read like titles
The blend (60/40) ranked results best in testing, e.g. "fixing cars" -> automotive technicians
rather than rail car repairers.

If even the best match is below MIN_SCORE, the query is out of scope and nothing is returned.
"""
import os

import numpy as np
import pandas as pd

from load_onet import PROCESSED
from text_clustering import load_embeddings

MODEL = "all-mpnet-base-v2"
PROFILE_WEIGHT = 0.6  # share of the score from the full job profile; the rest comes from the title
# Calibrated with calibrate(): real queries ("pizza", "money", "I like animals") score >= 0.33,
# nonsense ("asdfghjkl", "my favorite color is blue", "2 + 2") scores <= 0.26.
MIN_SCORE = 0.28
TOP_K = 10


class JobSearch:
    def __init__(self, model_name: str = MODEL):
        # The model is already downloaded by embed.py; offline mode avoids a slow network check
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from sentence_transformers import SentenceTransformer

        X, codes = load_embeddings(model_name)
        self.X = X
        self.codes = codes.to_numpy()
        h = pd.read_csv(PROCESSED / "hierarchy.csv", index_col=0)
        self.info = h.reindex(self.codes)[["title", "group_name", "sub_label"]]
        self.model = SentenceTransformer(model_name)
        self.T = self._title_embeddings(model_name)

    def _title_embeddings(self, model_name: str) -> np.ndarray:
        """Title vectors, cached next to the job embeddings (rebuilt if the job list changes)."""
        cache = PROCESSED / f"title_embeddings_{model_name}.npz"
        if cache.exists():
            data = np.load(cache, allow_pickle=False)
            if np.array_equal(data["codes"], self.codes.astype(str)):
                return data["vectors"]
        T = self.model.encode(self.info["title"].tolist(), normalize_embeddings=True, batch_size=64)
        np.savez(cache, codes=self.codes.astype(str), vectors=T)
        return T

    def scores(self, query: str) -> np.ndarray:
        q = self.model.encode([query], normalize_embeddings=True)[0]
        return PROFILE_WEIGHT * (self.X @ q) + (1 - PROFILE_WEIGHT) * (self.T @ q)

    def search(self, query: str, k: int = TOP_K, min_score: float = MIN_SCORE) -> list[dict]:
        query = query.strip()
        if not query:
            return []
        s = self.scores(query)
        top = np.argsort(s)[::-1][:k]
        return [
            {
                "code": self.codes[i],
                "title": self.info.iloc[i]["title"],
                "group": self.info.iloc[i]["group_name"],
                "cluster": self.info.iloc[i]["sub_label"],
                "score": round(float(s[i]), 3),
            }
            for i in top
            if s[i] >= min_score
        ]


def calibrate(engine: JobSearch) -> None:
    """Print the best match score for sensible and nonsense queries, to choose MIN_SCORE."""
    queries = [
        "I like animals", "fixing cars", "helping sick people", "working outdoors", "math and numbers",
        "cooking", "computers", "I want to draw", "teaching kids", "law", "money", "pizza",
        "asdfghjkl", "qwerty zxcv", "the", "banana", "my favorite color is blue", "lorem ipsum dolor sit amet",
        "2 + 2", "hello", "purple elephants dancing", "quantum chromodynamics",
    ]
    for q in queries:
        s = engine.scores(q)
        i = int(s.argmax())
        print(f"{s[i]:.3f}  {q:30} -> {engine.info.iloc[i]['title']}")


if __name__ == "__main__":
    calibrate(JobSearch())
