"""Phase 5: combine the clustering results into one nested hierarchy for the bubble UI.

    Level 1  9 named groups (Care, Guide, Protect, Build, Invent, Discover, Create, Connect,
             Organize), assigned by similarity to example tasks (top_groups.py)
    Level 2  text subclusters found inside each group (embeddings -> UMAP -> HDBSCAN)
    Level 3  large subclusters (> MAX_LEAF_CLUSTER jobs) are split once more with K-Means
    Leaves   individual occupations

Jobs HDBSCAN can't place go into a "Unique roles" bubble in their group, flagged as niche.

Run from the project root (after load_onet.py, embed.py and the Phase 3 notebook):
    .venv\\Scripts\\python src\\build_hierarchy.py

Outputs:
    web/data/hierarchy.json        nested tree for D3
    data/processed/hierarchy.csv   one row per occupation with its group / subcluster path
"""
import json
import math

import hdbscan
import numpy as np
import pandas as pd
import umap
from sklearn.cluster import KMeans

from labeling import load_job_text, top_terms
from load_onet import CODE, PROCESSED, ROOT, load_table
from text_clustering import SEED, load_embeddings
from top_groups import GROUPS, top_groups

MIN_CLUSTER_SIZE, MIN_SAMPLES = 5, 2   # HDBSCAN settings inside each group
MAX_LEAF_CLUSTER = 30                  # split subclusters bigger than this
TARGET_LEAF_SIZE = 15                  # aim for ~15 jobs per split
MIN_SPLIT_SIZE = 3                     # merge smaller splits into their nearest neighbor
N_TASKS_SHOWN = 5
NICHE = -1


def subcluster(Xg: np.ndarray) -> np.ndarray:
    """Level 2: HDBSCAN on a group's own UMAP projection. Returns labels, -1 = niche."""
    U = umap.UMAP(n_components=5, n_neighbors=10, min_dist=0.0, metric="cosine",
                  random_state=SEED).fit_transform(Xg)
    return hdbscan.HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE, min_samples=MIN_SAMPLES).fit(U).labels_


def split_large(Xc: np.ndarray) -> np.ndarray:
    """Level 3: split an oversized subcluster with K-Means on the (unit-length) embeddings."""
    k = max(2, math.ceil(len(Xc) / TARGET_LEAF_SIZE))
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(Xc)
    labels, centers = km.labels_.copy(), km.cluster_centers_
    # Fold tiny splits (1-2 jobs) into the closest remaining split
    while True:
        sizes = np.bincount(labels, minlength=len(centers))
        small = [c for c in range(len(centers)) if 0 < sizes[c] < MIN_SPLIT_SIZE]
        big = [c for c in range(len(centers)) if sizes[c] >= MIN_SPLIT_SIZE]
        if not small or not big:
            break
        for i in np.where(labels == small[0])[0]:
            labels[i] = big[int(np.argmax(centers[big] @ Xc[i]))]
    return np.unique(labels, return_inverse=True)[1]  # renumber 0..n-1


def build_paths(E: pd.DataFrame, groups: pd.Series) -> pd.DataFrame:
    """One row per job: group, sub (level 2 id), split (level 3 id or -1)."""
    rows = []
    for g in sorted(groups.unique()):
        idx = groups.index[groups == g]
        sub = subcluster(E.loc[idx].values)
        df = pd.DataFrame({"group": g, "sub": sub, "split": -1}, index=idx)
        for s in sorted(set(sub) - {NICHE}):
            members = df.index[df["sub"] == s]
            if len(members) > MAX_LEAF_CLUSTER:
                split = split_large(E.loc[members].values)
                if split.max() > 0:  # skip if everything merged back into one split
                    df.loc[members, "split"] = split
        rows.append(df)
    return pd.concat(rows)


def label_clusters(paths: pd.DataFrame, text: pd.Series) -> dict:
    """TF-IDF labels for every level 2 and level 3 cluster, compared against their siblings."""
    labels = {}
    for g, gdf in paths.groupby("group"):
        for s, row in top_terms(text, gdf["sub"]).iterrows():
            labels[(g, s)] = row
        for s, sdf in gdf[gdf["split"] >= 0].groupby("sub"):
            for k, row in top_terms(text, sdf["split"], exclude=()).iterrows():
                labels[(g, s, k)] = row
    return labels


def job_details() -> pd.DataFrame:
    occ = pd.read_csv(PROCESSED / "occupations.csv", index_col="code")
    interests = pd.read_csv(PROCESSED / "interests.csv", index_col="code")["top_interest"]
    tasks = load_table("Task Statements")
    tasks = tasks.sort_values("Task Type", key=lambda s: s.ne("Core"))  # core tasks first
    top_tasks = tasks.groupby(CODE)["Task"].agg(lambda t: list(t.head(N_TASKS_SHOWN)))
    return occ.join(interests).join(top_tasks.rename("tasks"))


def leaf(code: str, d: pd.Series, niche: bool) -> dict:
    return {
        "name": d["title"], "code": code, "value": 1, "niche": niche,
        "description": d["description"],
        "tasks": d["tasks"] if isinstance(d["tasks"], list) else [],
        "job_zone": None if pd.isna(d["job_zone"]) else int(d["job_zone"]),
        "interest": None if pd.isna(d["top_interest"]) else d["top_interest"],
        "url": f"https://www.onetonline.org/link/summary/{code}",
    }


def cluster_node(row: pd.Series, children: list) -> dict:
    return {"name": row["label"], "keywords": list(row["top_terms"]), "children": children}


def build_tree(paths: pd.DataFrame, labels: dict, details: pd.DataFrame) -> dict:
    root = {"name": "All occupations", "children": []}
    for group in GROUPS:  # keep the groups in their defined order
        g = group["name"]
        gdf = paths[paths["group"] == g]
        if gdf.empty:
            continue
        group_children = []
        for s, sdf in gdf.groupby("sub"):
            jobs = lambda df, niche=False: [leaf(c, details.loc[c], niche) for c in df.index]
            if s == NICHE:
                group_children.append({"name": "Unique roles", "niche": True,
                                       "keywords": ["hard to group with others"], "children": jobs(sdf, True)})
            elif (sdf["split"] >= 0).any():
                splits = [cluster_node(labels[(g, s, k)], jobs(kdf)) for k, kdf in sdf.groupby("split")]
                group_children.append(cluster_node(labels[(g, s)], splits))
            else:
                group_children.append(cluster_node(labels[(g, s)], jobs(sdf)))
        root["children"].append({"name": g, "tagline": group["tagline"], "children": group_children})
    return root


def main() -> None:
    X, codes = load_embeddings()
    E = pd.DataFrame(X, index=codes)
    text = load_job_text()

    groups = top_groups().loc[E.index]
    paths = build_paths(E, groups)
    labels = label_clusters(paths, text)
    details = job_details()

    tree = build_tree(paths, labels, details)
    out = ROOT / "web" / "data" / "hierarchy.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(tree, ensure_ascii=False), encoding="utf-8")

    flat = paths.join(details["title"])
    flat["group_name"] = flat["group"]
    flat["sub_label"] = [labels[(g, s)]["label"] if s != NICHE else "Unique roles"
                         for g, s in zip(flat["group"], flat["sub"])]
    flat.to_csv(PROCESSED / "hierarchy.csv")

    n_sub = paths[paths["sub"] != NICHE].groupby("group")["sub"].nunique()
    n_split = paths[paths["split"] >= 0].groupby(["group", "sub"]).size()
    print(f"Jobs: {len(paths)}  |  niche: {(paths['sub'] == NICHE).sum()}")
    print(f"Level 2 clusters: {n_sub.sum()}  |  of which split again: {len(n_split)}")
    print(flat.groupby("group_name").agg(jobs=("title", "size"), subclusters=("sub", "nunique")).to_string())
    print(f"Saved {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
