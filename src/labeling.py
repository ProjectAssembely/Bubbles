"""Phase 4: name clusters automatically with TF-IDF.

Each cluster is treated as one "document". A term scores high when it's common inside the
cluster (TF) but appears in few other clusters (IDF), so the top terms describe what sets it apart.

TF here is the share of the cluster's jobs that use the term, not a raw word count. With raw
counts, one job with a long, jargon-heavy task list (e.g. RFID engineers) could take over the
label of a 75-job cluster.
"""
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer

from load_onet import PROCESSED

# Words common in almost every O*NET task statement that say nothing about the job
DOMAIN_STOP_WORDS = {
    "perform", "performs", "performing", "duties", "duty", "including", "related", "use", "using", "used",
    "ensure", "ensuring", "work", "workers", "worker", "working", "provide", "providing", "according",
    "activities", "activity", "specified", "determine", "prepare", "maintain", "maintaining",
    "procedures", "review", "information", "required", "requirements", "assist", "assisting", "conduct",
    "develop", "developing", "direct", "coordinate", "monitor", "appropriate", "necessary", "various",
    "new", "existing", "make", "based", "needs", "needed", "areas", "area", "methods", "techniques",
    "tasks", "task", "occupations", "occupation", "etc", "such", "include", "includes", "jobs",
    "abreast", "developments", "current", "proper", "large", "holding", "explain", "incoming", "levels",
    "applications", "operational", "effects",
}
# Note: TF-IDF's max_df also drops any term that appears in more than half of the clusters.
STOP_WORDS = list(ENGLISH_STOP_WORDS | DOMAIN_STOP_WORDS)


def load_job_text() -> pd.Series:
    """Title + description + tasks for each occupation, indexed by code (from load_onet.py)."""
    return pd.read_csv(PROCESSED / "corpus.csv", index_col="code")["text"]


def _stem(term: str) -> str:
    """Very light normalization so 'patient' and 'patients' count as the same term."""
    return " ".join(re.sub(r"(ies|es|s)$", "", w) for w in term.split())


def tfidf_scores(job_text: pd.Series, labels: pd.Series, min_jobs: int = 2) -> pd.DataFrame:
    """Clusters x terms TF-IDF matrix.

    TF  = share of the cluster's jobs whose text contains the term
    IDF = log(1 + n_clusters / n_clusters_using_term), where a cluster "uses" a term
          if at least 10% of its jobs do
    """
    vec = CountVectorizer(stop_words=STOP_WORDS, ngram_range=(1, 2), binary=True, min_df=min_jobs,
                          token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b")
    X = vec.fit_transform(job_text.loc[labels.index].values)  # jobs x terms, 0/1
    clusters = np.sort(labels.unique())
    tf = np.vstack([np.asarray(X[(labels == c).values].mean(axis=0)).ravel() for c in clusters])
    df = (tf >= 0.10).sum(axis=0)
    idf = np.log(1 + len(clusters) / np.maximum(df, 1))
    return pd.DataFrame(tf * idf, index=clusters, columns=vec.get_feature_names_out())


def top_terms(job_text: pd.Series, labels: pd.Series, n: int = 5, exclude=(-1,)) -> pd.DataFrame:
    """Top-n distinguishing terms per cluster.

    job_text: text per occupation (index = code)
    labels:   cluster id per occupation (index = code); ids in `exclude` (noise) are skipped
    """
    labels = labels[~labels.isin(exclude)]
    S = tfidf_scores(job_text, labels)
    vocab = S.columns.to_numpy()

    rows = []
    for cluster in S.index:
        scores = S.loc[cluster].to_numpy()
        chosen, seen = [], set()
        for j in scores.argsort()[::-1]:
            term = vocab[j]
            # Words reduced to a short root so 'sew' / 'sewing' / 'sewing machines' count as overlapping
            words = {_stem(w)[:5] for w in term.split()}
            # Skip near-duplicates: any term sharing a word with one already chosen
            if words & seen:
                continue
            chosen.append(term)
            seen.update(words)
            if len(chosen) == n:
                break
        rows.append({"cluster": cluster, "size": int((labels == cluster).sum()), "top_terms": chosen,
                     "label": " · ".join(t.title() for t in chosen[:3])})
    return pd.DataFrame(rows).set_index("cluster")
