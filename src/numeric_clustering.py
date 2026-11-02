"""Project 2: cluster occupations by their numeric O*NET attribute scores.

Used by notebooks/03_numeric_clustering.ipynb.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_samples
from sklearn.preprocessing import StandardScaler

from load_onet import PROCESSED
from text_clustering import SEED, score


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


def misfits(Z, labels: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """Jobs whose silhouette is below the threshold: closer to another cluster than their own.
    These are the 'niche' roles the README asks to keep separate."""
    return silhouette_samples(np.asarray(Z), labels) < threshold
