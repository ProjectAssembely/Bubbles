"""Project 1, steps 2-3: reduce embeddings with UMAP, then cluster with HDBSCAN and K-Means.

Used by notebooks/02_text_clustering.ipynb.
"""
import numpy as np
import pandas as pd
import hdbscan
import umap
from sklearn.cluster import KMeans

from evaluation import SEED, score  # noqa: F401  (score is re-exported for the notebooks)
from load_onet import PROCESSED


def load_embeddings(tag: str = "all-mpnet-base-v2") -> tuple[np.ndarray, pd.Series]:
    X = np.load(PROCESSED / f"embeddings_{tag}.npy")
    codes = pd.read_csv(PROCESSED / f"embeddings_{tag}_codes.csv")["code"]
    return X, codes


def reduce(X: np.ndarray, n_components: int, n_neighbors: int = 15) -> np.ndarray:
    """UMAP with cosine distance. min_dist=0 packs similar points tightly, which helps clustering."""
    reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=0.0,
                        metric="cosine", random_state=SEED)
    return reducer.fit_transform(X)


def run_hdbscan(Xr: np.ndarray, min_cluster_size: int, min_samples: int | None = None) -> hdbscan.HDBSCAN:
    model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples,
                            cluster_selection_method="eom", prediction_data=True)
    return model.fit(Xr)


def run_kmeans(Xr: np.ndarray, k: int) -> KMeans:
    return KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=SEED).fit(Xr)


def sweep_hdbscan(Xr: np.ndarray, sizes, min_samples_list=(None,), reference=None) -> pd.DataFrame:
    rows = []
    for mcs in sizes:
        for ms in min_samples_list:
            labels = run_hdbscan(Xr, mcs, ms).labels_
            rows.append({"min_cluster_size": mcs, "min_samples": ms or mcs, **score(Xr, labels, reference)})
    return pd.DataFrame(rows)


def sweep_kmeans(Xr: np.ndarray, ks, reference=None) -> pd.DataFrame:
    rows = []
    for k in ks:
        km = run_kmeans(Xr, k)
        rows.append({"k": k, "inertia": km.inertia_, **score(Xr, km.labels_, reference)})
    return pd.DataFrame(rows)


def cluster_summary(labels: np.ndarray, titles: pd.Series, n_examples: int = 5) -> pd.DataFrame:
    """Size and a few example job titles for each cluster."""
    df = pd.DataFrame({"cluster": labels, "title": titles.values})
    g = df.groupby("cluster")["title"]
    return pd.DataFrame({
        "size": g.size(),
        "examples": g.apply(lambda t: "; ".join(t.head(n_examples))),
    })
