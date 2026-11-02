"""Top-level bubble groups from text embeddings and numeric attributes combined.

Numeric-only groups (Phase 3) describe *how* work is done but split fields apart
(e.g. nurses and nursing assistants landed in different groups). Text-only groups keep
fields together but match career interests less well. Mixing both gives groups that
keep fields together AND line up best with RIASEC interests.

Each source is reduced with PCA and scaled to the same overall size, then weighted.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

from numeric_clustering import load_features, standardize
from text_clustering import SEED, load_embeddings

TEXT_WEIGHT = 0.7   # share of the distance that comes from the text embeddings
N_GROUPS = 6
TEXT_DIMS, NUMERIC_DIMS = 50, 40  # PCA sizes (numeric 40 dims keeps ~90% of variance)


def _block(M: np.ndarray, n: int) -> np.ndarray:
    """PCA-reduce, then scale so each row's average squared length is 1."""
    P = PCA(n_components=n, random_state=SEED).fit_transform(M)
    return P / np.sqrt((P ** 2).sum(axis=1).mean())


def hybrid_matrix(text_weight: float = TEXT_WEIGHT) -> pd.DataFrame:
    """Combined feature matrix for jobs that have both text and numeric data."""
    X, codes = load_embeddings()
    E = pd.DataFrame(X, index=codes)
    Z = standardize(load_features("IM"))
    common = Z.index.intersection(E.index)
    H = np.hstack([np.sqrt(text_weight) * _block(E.loc[common].values, TEXT_DIMS),
                   np.sqrt(1 - text_weight) * _block(Z.loc[common].values, NUMERIC_DIMS)])
    return pd.DataFrame(H, index=common)


def hybrid_groups(text_weight: float = TEXT_WEIGHT, k: int = N_GROUPS) -> pd.Series:
    """Group id for every job with text. Jobs without numeric scores get the majority
    group of their 5 nearest neighbors in text-embedding space."""
    H = hybrid_matrix(text_weight)
    labels = pd.Series(KMeans(n_clusters=k, n_init=10, random_state=SEED).fit_predict(H),
                       index=H.index, name="group")
    X, codes = load_embeddings()
    E = pd.DataFrame(X, index=codes)
    missing = E.index.difference(labels.index)
    if len(missing):
        knn = KNeighborsClassifier(n_neighbors=5, metric="cosine").fit(E.loc[labels.index], labels)
        labels = pd.concat([labels, pd.Series(knn.predict(E.loc[missing]), index=missing, name="group")])
    return labels.loc[E.index]
