# Bubbles
This project is a tool that uses unsupervised machine learning to automatically cluster thousands of O*NET occupations by semantic similarity, then presents them as a zoomable, nested bubble interface so someone can start broad from a category like "I want to help people" and drill down.


For Project 1: Clustering text-based task statement (NLP approach)

Extract and embed: Vectorize the text data into dense, fixed-length vectors using a transformer model optimized for semantic text similarity
Dimension Reduction: High-dimensional embeddings can degrade distance metrics (the curse of dimensionality). Apply UMAP (Uniform Manifold Approximation and Projection) or PCA to reduce vectors to a manageable scale while preserving local and global structures.
Density-Based Clustering: Use HDBSCAN(research it) and K-Means and compare. 

For Project 2: Clustering by Numeric Attributes (Data Science Approach)

Pivot and Matrix Construction: Create a feature matrix where each row represents an O*NET-SOC code and each column represents a specific attribute score (e.g., Level of Active Listening, Importance of Mathematics)
Feature Standardization: Scale the numeric data using StandardScaler to handle any variance differences across distinct descriptor types
Partitioning Optimization: Since the numeric feature space is dense and continuous, K-Means++ or Agglomerative Hierarchical Clustering works remarkably well. Use the Silhouette Method or Elbow Method (Inertia) to automatically determine the mathematical optimal number of clusters (K)

O*NET data contains highly niche roles. Rely on density algorithms like HDBSCAN or set a strict silhouette distance threshold to capture unclassifiable job elements as independent vectors. Use TF-IDF on the text fields within each finalized cluster group to automatically pull out the top 5 distinguishing words. This provides an instant label or name for your newly generated job cluster.

---

## Quick start

Requires Python 3.12 (3.14 isn't supported by all the libraries yet).

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

**1. Get the data.** Download the O*NET 31.0 database (text format) from
https://www.onetcenter.org/database.html and unzip it to `data/raw/onet_31_0/`.

**2. Run the pipeline** (from the project root):

```powershell
.venv\Scripts\python src\load_onet.py        # build tables in data/processed/
.venv\Scripts\python src\embed.py            # sentence embeddings (~10 min on CPU, cached after)
.venv\Scripts\jupyter nbconvert --to notebook --execute --inplace notebooks\02_text_clustering.ipynb
.venv\Scripts\jupyter nbconvert --to notebook --execute --inplace notebooks\03_numeric_clustering.ipynb
.venv\Scripts\python src\build_hierarchy.py  # writes web/data/hierarchy.json
```

**3. Open the bubble UI:**

```powershell
cd web
..\.venv\Scripts\python -m http.server 8000
```

Then visit http://localhost:8000. Link straight to a job with `#Job Title`, e.g. `http://localhost:8000/#Registered Nurses`.

## How it works

| Step | Code | What it does |
|---|---|---|
| Load | `src/load_onet.py` | Text corpus, attribute matrices (161 attributes), RIASEC interests |
| Project 1 | `src/embed.py`, `notebooks/02_text_clustering.ipynb` | Embed each task with `all-mpnet-base-v2`, average per job → UMAP → HDBSCAN vs. K-Means |
| Project 2 | `notebooks/03_numeric_clustering.ipynb` | StandardScaler → K-Means++ vs. Ward agglomerative, elbow + silhouette |
| Labels | `src/labeling.py`, `notebooks/04_cluster_labels.ipynb` | TF-IDF top terms per cluster; plain-language names for top groups |
| Top groups | `src/top_groups.py`, `notebooks/06_top_groups.ipynb` | 9 named groups (Care, Guide, Protect, Build, Invent, Discover, Create, Connect, Organize) assigned by similarity to example tasks |
| Hierarchy | `src/build_hierarchy.py`, `notebooks/05_hierarchy.ipynb` | 9 top groups → HDBSCAN subclusters → jobs |
| UI | `web/` | D3 zoomable circle packing |

## Findings

- **Text clustering (Project 1):** HDBSCAN (silhouette 0.60, 31 clusters) beat K-Means (0.54) and flagged ~9% of jobs
  as niche. Clusters capture the *field* (legal, culinary, IT, trades).
- **Numeric clustering (Project 2):** the attribute space is a continuum, not separate islands (silhouette ≤ 0.24 for any K),
  so the "mathematically optimal" K=2 is too coarse. K=6 gives nameable groups that capture *how* work is done
  (physical, social, analytical) and match RIASEC interests better than text clusters.
- **Combined:** mixing both (70% text, 30% numeric) kept fields together (80% vs. 64% for numeric only) and matched
  career interests better (ARI 0.35 vs. 0.24). See `notebooks/05_hierarchy.ipynb`.
- **Named top level:** the bubbles now start from 9 fixed groups. Each is described by example tasks; jobs are assigned
  by embedding similarity, then refined with a seeded K-Means that stays anchored to the descriptions. This matches
  career interests best of all versions (ARI 0.38) and passes all hand-checked placements. See `notebooks/06_top_groups.ipynb`.
- **Labels:** plain TF-IDF on merged cluster text let one jargon-heavy job dominate a cluster's label. Using the share of
  jobs that mention a term as the TF fixed this.

## Data credit

This project uses the O*NET 31.0 Database by the U.S. Department of Labor, Employment and Training Administration
(USDOL/ETA), under the [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) license. O*NET® is a trademark of USDOL/ETA.
The groupings were generated by this project and are not part of O*NET.
