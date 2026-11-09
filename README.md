# Bubbles — Project 2: Clustering by numeric attributes (data science)

> This branch contains **only Project 2**. The full project (Project 1, the named top-level groups and the
> bubble UI) lives on `main`; Project 1 alone is on `project-1-text-clustering`.

Bubbles uses unsupervised machine learning to cluster O*NET occupations. Project 2 groups jobs by their **numeric
attribute scores**: how important each ability, skill, knowledge area and work activity is to the job.

## The plan (from the project brief)

- **Pivot and matrix construction:** a feature matrix where each row is an O*NET-SOC code and each column a specific
  attribute score (e.g. Level of Active Listening, Importance of Mathematics).
- **Feature standardization:** scale the data with StandardScaler to handle variance differences across descriptor types.
- **Partitioning optimization:** K-Means++ or Agglomerative Hierarchical Clustering, using the Silhouette or Elbow
  (Inertia) method to determine the optimal number of clusters (K).
- **Niche roles:** set a silhouette threshold to flag jobs that don't fit any cluster.

## What's here

| File | What it does |
|---|---|
| `src/load_onet.py` | Reads the O*NET 31.0 text files and pivots Abilities, Knowledge, Work Activities, Essential Skills and Transferable Skills into occupation × attribute matrices (importance `IM` and level `LV`) |
| `src/numeric_clustering.py` | StandardScaler, K-Means++ and Ward agglomerative sweeps, defining attributes per cluster, silhouette-based niche detection |
| `src/evaluation.py` | Shared seed and scoring: silhouette, agreement with O*NET job families and RIASEC interests |
| `notebooks/03_numeric_clustering.ipynb` | The full analysis, with plots |

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
.venv\Scripts\jupyter nbconvert --to notebook --execute --inplace notebooks\03_numeric_clustering.ipynb
```

## Findings

- **Matrix:** 910 occupations × 161 attributes (52 abilities, 41 work activities, 33 knowledge areas,
  25 transferable + 10 essential skills). Importance and level scores correlate at 0.96, so importance alone is used.
- **The space is a continuum.** The first principal component alone explains 34% of the variance (roughly physical ↔
  cognitive work), and silhouette scores stay **at or below 0.24 for every K**. Jobs shift gradually rather than
  forming separate islands.
- **Choosing K:** the silhouette method's optimum is K=2 (physical vs. non-physical), which is too coarse to use, and
  the elbow is smooth. **K=6** was chosen for interpretability:

  | Group | Defining attributes |
  |---|---|
  | Tech & analysis | Programming, Mathematics, Technology Design |
  | Mechanical & repair | Equipment Maintenance, Repairing |
  | Teaching & social | Instructing, Speaking, Learning Strategies |
  | Healthcare & caregiving | Assisting and Caring for Others, Medicine |
  | Physical labor | Stamina, Strength, Handling Objects |
  | Business & admin | Administrative, Customer Service, Sales |

- **K-Means++ vs. Ward agglomerative:** K-Means scores higher on silhouette at almost every K; Ward gives a full tree
  (dendrogram), useful for building a hierarchy.
- Numeric groups describe **how** work is done, so they match RIASEC career interests (ARI ≈ 0.26) better than
  text clusters do.
- **Niche roles:** 38 jobs (4%) have a negative silhouette (closer to another cluster than their own) and are flagged.

## Data credit

This project uses the O*NET 31.0 Database by the U.S. Department of Labor, Employment and Training Administration
(USDOL/ETA), under the [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) license. O*NET® is a trademark of USDOL/ETA.
