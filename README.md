# Bubbles — Project 1: Clustering by task text (NLP)

> This branch contains **only Project 1**. The full project (Project 2, the named top-level groups and the
> bubble UI) lives on `main`; Project 2 alone is on `project-2-numeric-clustering`.

Bubbles uses unsupervised machine learning to cluster O*NET occupations by semantic similarity. Project 1 groups
jobs by **what their task statements say**.

## The plan (from the project brief)

- **Extract and embed:** vectorize the text into dense, fixed-length vectors using a transformer model optimized
  for semantic text similarity.
- **Dimension reduction:** high-dimensional embeddings can degrade distance metrics (the curse of dimensionality).
  Apply UMAP or PCA to reduce vectors while preserving local and global structure.
- **Density-based clustering:** use HDBSCAN and K-Means and compare.

## What's here

| File | What it does |
|---|---|
| `src/load_onet.py` | Reads the O*NET 31.0 text files and builds one text block per occupation (title + description + tasks) |
| `src/embed.py` | Embeds every task statement with `all-mpnet-base-v2` and averages them into one 768-number vector per job (avoids the model cutting off long task lists); caches results |
| `src/text_clustering.py` | UMAP reduction, HDBSCAN and K-Means runs and parameter sweeps, cluster summaries |
| `src/evaluation.py` | Shared seed and scoring (silhouette, noise %, agreement with O*NET job families and RIASEC interests) and automatic choice of K (silhouette, Davies-Bouldin, Calinski-Harabasz, Kneedle elbow) |
| `notebooks/02_text_clustering.ipynb` | The full analysis, with plots |

## Quick start

Requires Python 3.12.

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

1. Download the O*NET 31.0 database (text format) from https://www.onetcenter.org/database.html and unzip it to
   `data/raw/onet_31_0/`.
2. Run:

```powershell
.venv\Scripts\python src\load_onet.py   # build tables in data/processed/
.venv\Scripts\python src\embed.py       # sentence embeddings (~10 min on CPU, cached after)
.venv\Scripts\python src\embed.py all-MiniLM-L6-v2   # optional: smaller model for the comparison in section 1c
.venv\Scripts\jupyter nbconvert --to notebook --execute --inplace notebooks\02_text_clustering.ipynb
```

## Findings

- **Embedding:** 923 occupations (those with task statements) → 768-dimensional vectors.
- **Dimension reduction (raw vs. PCA vs. UMAP, all scored in the original space):**
  - In raw 768d, the farthest job is only ~2.5× farther than the nearest: the curse of dimensionality.
    After UMAP the contrast is ~200×.
  - HDBSCAN labels **38–66% of jobs as noise** on raw or PCA vectors at any setting, versus **~9% after UMAP**.
  - PCA preserves neighborhoods best (trustworthiness up to 0.999) but doesn't create the density contrast HDBSCAN
    needs. UMAP is the right choice here.
- **Embedding model (mpnet vs. MiniLM):** the 5× smaller `all-MiniLM-L6-v2` performs almost the same as
  `all-mpnet-base-v2`: neighbor purity 0.342 vs 0.348, both give 31 HDBSCAN clusters, agreement with O*NET families
  0.361 vs 0.363, and the two models' clusterings agree at ARI 0.87. mpnet is kept for the analysis.
- **Choosing K for K-Means automatically:** over K = 2–60, silhouette and Davies-Bouldin both show a coarse peak at
  K=4 and a fine one around K=44; the K with their best average rank is **K=44**. Calinski-Harabasz rises toward
  the end of any range on UMAP output, so it isn't used.
- **HDBSCAN vs. K-Means (on UMAP 10d):**

  | | HDBSCAN | K-Means (K=44) |
  |---|---|---|
  | Clusters | 31, found by the algorithm | 44, chosen automatically |
  | Noise | 8.6% flagged as niche | none: every job forced into a cluster |
  | Silhouette | **0.60** | 0.56 |
  | Agreement with each other | ARI 0.67 | |

- Clusters capture a job's **field** (legal, culinary, IT, construction trades). HDBSCAN's outlier scores flag
  unusual groups, e.g. the six funeral-related jobs form a tight group too small to be its own cluster.
- The notebook also explains **how HDBSCAN works** (core distance, mutual reachability, condensed tree,
  stability-based selection) and shows the condensed tree, membership probabilities and outlier scores.

## Data credit

This project uses the O*NET 31.0 Database by the U.S. Department of Labor, Employment and Training Administration
(USDOL/ETA), under the [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) license. O*NET® is a trademark of USDOL/ETA.
