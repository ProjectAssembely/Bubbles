"""Shared settings, cluster-quality scoring and automatic choice of K, used by both projects."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score,
                             silhouette_score)

SEED = 42  # random seed for UMAP, PCA and K-Means so results are reproducible

# Higher is better for these; lower is better for Davies-Bouldin
HIGHER_IS_BETTER = {"silhouette": True, "davies_bouldin": False, "calinski_harabasz": True}


def k_metrics(X, ks) -> pd.DataFrame:
    """For K-Means++ at each K: inertia, silhouette (higher = better), Davies-Bouldin
    (lower = better) and Calinski-Harabasz (higher = better)."""
    X = np.asarray(X)
    rows = []
    for k in ks:
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=SEED).fit(X)
        l = km.labels_
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": silhouette_score(X, l),
                     "davies_bouldin": davies_bouldin_score(X, l), "calinski_harabasz": calinski_harabasz_score(X, l)})
    return pd.DataFrame(rows).set_index("k")


def kneedle(ks, inertia) -> int:
    """Elbow point: the K farthest from the straight line joining the two ends of the
    normalized inertia curve (Satopaa et al., 2011)."""
    x, y = np.asarray(list(ks), float), np.asarray(inertia, float)
    x, y = (x - x.min()) / (x.max() - x.min()), (y - y.min()) / (y.max() - y.min())
    dist = np.abs((y[-1] - y[0]) * x - (x[-1] - x[0]) * y + x[-1] * y[0] - y[-1] * x[0])
    return int(np.asarray(list(ks))[dist.argmax()])


def consensus_k(metrics: pd.DataFrame, k_min: int, k_max: int,
                use=("silhouette", "davies_bouldin", "calinski_harabasz")) -> tuple[int, pd.Series]:
    """The K with the best average rank across the chosen metrics within [k_min, k_max].
    Returns the chosen K and every K's average rank."""
    m = metrics.loc[k_min:k_max]
    rank = sum(m[name].rank(ascending=not HIGHER_IS_BETTER[name]) for name in use) / len(use)
    return int(rank.idxmin()), rank.rename("average_rank")


def score(Xr: np.ndarray, labels: np.ndarray, reference: dict[str, pd.Series] | None = None) -> dict:
    """Quality metrics for one labeling. Noise points (-1) are left out of the silhouette.

    reference: optional existing groupings to compare against with the Adjusted Rand Index,
    e.g. {"SOC": job family per job, "RIASEC": top interest per job}.
    """
    mask = labels != -1
    n_clusters = len(set(labels[mask]))
    out = {
        "n_clusters": n_clusters,
        "noise_pct": round(100 * (1 - mask.mean()), 1),
        "silhouette": round(silhouette_score(Xr[mask], labels[mask]), 3) if n_clusters > 1 else np.nan,
    }
    for name, ref in (reference or {}).items():
        out[f"ARI_vs_{name}"] = round(adjusted_rand_score(ref.values[mask], labels[mask]), 3)
    return out
