"""Project 2: cluster occupations by their numeric O*NET attribute scores.

Used by notebooks/03_numeric_clustering.ipynb.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score,
                             silhouette_samples, silhouette_score)
from sklearn.preprocessing import StandardScaler

from evaluation import SEED, score
from load_onet import PROCESSED


def load_features(scale: str = "IM") -> pd.DataFrame:
    """Occupations x attributes matrix built by load_onet.py ('IM' = importance, 'LV' = level)."""
    return pd.read_csv(PROCESSED / f"features_{scale}.csv", index_col="code")


def standardize(F: pd.DataFrame) -> pd.DataFrame:
    """Z-score each attribute so ones with larger spread don't dominate the distances."""
    return pd.DataFrame(StandardScaler().fit_transform(F), index=F.index, columns=F.columns)


def run_kmeans(Z, k: int) -> KMeans:
    return KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=SEED).fit(Z)


def run_agglomerative(Z, k: int) -> np.ndarray:
    return AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(Z)


def sweep(Z, ks, reference=None) -> pd.DataFrame:
    """K-Means++ and Ward agglomerative for each K, with inertia for the elbow plot."""
    Z = np.asarray(Z)
    rows = []
    for k in ks:
        km = run_kmeans(Z, k)
        rows.append({"method": "kmeans", "k": k, "inertia": km.inertia_, **score(Z, km.labels_, reference)})
        rows.append({"method": "agglomerative", "k": k, "inertia": np.nan,
                     **score(Z, run_agglomerative(Z, k), reference)})
    return pd.DataFrame(rows)


def profile(Z: pd.DataFrame, labels: np.ndarray, n: int = 5) -> pd.Series:
    """Top-n attributes that most set each cluster apart (highest average z-score)."""
    out = {}
    for c in sorted(set(labels)):
        centroid = Z[labels == c].mean().sort_values(ascending=False)
        out[c] = ", ".join(name.split(":", 1)[1] for name in centroid.index[:n])
    return pd.Series(out, name="defining_attributes")


# Automatic choice of K -------------------------------------------------------------------
# Every internal metric's global optimum on this data is K=2 (physical vs. non-physical work),
# which is too coarse for exploring careers. So K is chosen within a useful range: at least 4
# broad groups, and no more than 12 so the top-level choices stay easy to scan.
K_MIN, K_MAX = 4, 12


def k_metrics(Z, ks) -> pd.DataFrame:
    """Inertia, silhouette (higher = better), Davies-Bouldin (lower = better) and
    Calinski-Harabasz (higher = better) for K-Means at each K."""
    Z = np.asarray(Z)
    rows = []
    for k in ks:
        km = run_kmeans(Z, k)
        l = km.labels_
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": silhouette_score(Z, l),
                     "davies_bouldin": davies_bouldin_score(Z, l), "calinski_harabasz": calinski_harabasz_score(Z, l)})
    return pd.DataFrame(rows).set_index("k")


def kneedle(ks, inertia) -> int:
    """Elbow point: the K farthest from the straight line joining the two ends of the
    normalized inertia curve (Satopaa et al., 2011)."""
    x, y = np.asarray(list(ks), float), np.asarray(inertia, float)
    x, y = (x - x.min()) / (x.max() - x.min()), (y - y.min()) / (y.max() - y.min())
    dist = np.abs((y[-1] - y[0]) * x - (x[-1] - x[0]) * y + x[-1] * y[0] - y[-1] * x[0])
    return int(np.asarray(list(ks))[dist.argmax()])


def gap_statistic(Z, ks, inertias, n_refs: int = 10) -> pd.DataFrame:
    """Gap statistic (Tibshirani et al., 2001): compares log(inertia) with what K-Means gets on
    random data with no clusters (uniform in the data's principal-axis bounding box)."""
    Z = np.asarray(Z)
    rng = np.random.default_rng(SEED)
    Zc = Z - Z.mean(axis=0)
    _, _, Vt = np.linalg.svd(Zc, full_matrices=False)
    P = Zc @ Vt.T
    lo, hi = P.min(axis=0), P.max(axis=0)
    rows = []
    for k, w in zip(ks, inertias):
        ref = np.log([KMeans(k, n_init=3, random_state=b).fit(rng.uniform(lo, hi, P.shape) @ Vt).inertia_
                      for b in range(n_refs)])
        rows.append({"k": k, "gap": ref.mean() - np.log(w), "gap_se": ref.std() * np.sqrt(1 + 1 / n_refs)})
    return pd.DataFrame(rows).set_index("k")


def gap_choice(gap: pd.DataFrame) -> int:
    """Tibshirani's rule: the smallest K with gap(K) >= gap(K+1) - se(K+1)."""
    ks = gap.index.to_list()
    for k, k1 in zip(ks, ks[1:]):
        if gap.loc[k, "gap"] >= gap.loc[k1, "gap"] - gap.loc[k1, "gap_se"]:
            return k
    return ks[-1]


def stability(Z, k: int, n_boot: int = 10) -> float:
    """Mean ARI between K-Means solutions fitted on bootstrap resamples (1 = always the same groups)."""
    Z = np.asarray(Z)
    rng = np.random.default_rng(SEED)
    labs = [run_kmeans(Z[rng.choice(len(Z), len(Z))], k).predict(Z) for _ in range(n_boot)]
    return float(np.mean([adjusted_rand_score(labs[i], labs[j])
                          for i in range(n_boot) for j in range(i + 1, n_boot)]))


def choose_k(metrics: pd.DataFrame, k_min: int = K_MIN, k_max: int = K_MAX) -> tuple[int, pd.Series]:
    """Consensus K: the best average rank across silhouette, Davies-Bouldin and Calinski-Harabasz
    within [k_min, k_max]. Returns the chosen K and every K's average rank."""
    m = metrics.loc[k_min:k_max]
    rank = (m["silhouette"].rank(ascending=False) + m["davies_bouldin"].rank()
            + m["calinski_harabasz"].rank(ascending=False)) / 3
    return int(rank.idxmin()), rank.rename("average_rank")


def misfits(Z, labels: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """Jobs whose silhouette is below the threshold: closer to another cluster than their own.
    These are the 'niche' roles the README asks to keep separate."""
    return silhouette_samples(np.asarray(Z), labels) < threshold
