"""Shared settings and cluster-quality scoring used by both projects."""
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, silhouette_score

SEED = 42  # random seed for UMAP, PCA and K-Means so results are reproducible


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
