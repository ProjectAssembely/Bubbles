"""Load the O*NET 31.0 text database and build the inputs for both projects.

Run from the project root:
    .venv\\Scripts\\python src\\load_onet.py

Outputs (in data/processed/):
    occupations.csv      one row per occupation: code, title, description, job zone
    corpus.csv           Project 1 input: one text block per occupation (title + description + tasks)
    features_IM.csv      Project 2 input: occupations x attribute importance scores
    features_LV.csv      Project 2 input: occupations x attribute level scores
    interests.csv        RIASEC interest scores + top interest per occupation
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "onet_31_0"
PROCESSED = ROOT / "data" / "processed"

CODE = "O*NET-SOC Code"

# Numeric rating files used for Project 2. Each has IM (importance) and LV (level) scales.
# In O*NET 31.0 the old "Skills" file is split into Essential + Transferable Skills.
RATING_FILES = {
    "ability": "Abilities",
    "knowledge": "Knowledge",
    "activity": "Work Activities",
    "skill_essential": "Essential Skills",
    "skill_transferable": "Transferable Skills",
}

RIASEC = ["Realistic", "Investigative", "Artistic", "Social", "Enterprising", "Conventional"]


def load_table(name: str) -> pd.DataFrame:
    """Read one O*NET tab-separated file by name, e.g. 'Task Statements'."""
    return pd.read_csv(RAW / f"{name}.txt", sep="\t", dtype={CODE: str})


def load_occupations() -> pd.DataFrame:
    occ = load_table("Occupation Data")
    zones = load_table("Job Zones")[[CODE, "Job Zone"]]
    occ = occ.merge(zones, on=CODE, how="left")
    return occ.rename(columns={CODE: "code", "Title": "title", "Description": "description",
                               "Job Zone": "job_zone"})


def build_corpus(occ: pd.DataFrame, core_only: bool = False) -> pd.DataFrame:
    """Project 1: one text block per occupation = title + description + all task statements."""
    tasks = load_table("Task Statements")
    if core_only:
        tasks = tasks[tasks["Task Type"] == "Core"]
    task_text = (tasks.groupby(CODE)["Task"]
                 .agg(lambda t: " ".join(t))
                 .rename("tasks"))
    task_count = tasks.groupby(CODE).size().rename("n_tasks")

    df = occ.set_index("code").join(task_text).join(task_count)
    df["n_tasks"] = df["n_tasks"].fillna(0).astype(int)
    df["tasks"] = df["tasks"].fillna("")
    df["text"] = df["title"] + ". " + df["description"] + " " + df["tasks"]
    return df.reset_index()[["code", "title", "n_tasks", "text"]]


def build_feature_matrix(scale: str = "IM") -> pd.DataFrame:
    """Project 2: pivot all rating files into one occupations x attributes matrix."""
    frames = []
    for prefix, name in RATING_FILES.items():
        t = load_table(name)
        t = t[t["Scale ID"] == scale]
        wide = t.pivot_table(index=CODE, columns="Element Name", values="Data Value")
        wide.columns = [f"{prefix}:{c}" for c in wide.columns]
        frames.append(wide)
    # Inner join: keep only occupations that have ratings in every file
    matrix = pd.concat(frames, axis=1, join="inner")
    matrix.index.name = "code"
    return matrix


def load_interests() -> pd.DataFrame:
    """RIASEC interest scores (scale OI, 1-7) plus each occupation's top interest."""
    t = load_table("Career Interest Types")
    t = t[t["Scale ID"] == "OI"]
    wide = t.pivot_table(index=CODE, columns="Element Name", values="Data Value")[RIASEC]
    wide["top_interest"] = wide.idxmax(axis=1)
    wide.index.name = "code"
    return wide


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)

    occ = load_occupations()
    corpus = build_corpus(occ)
    feats_im = build_feature_matrix("IM")
    feats_lv = build_feature_matrix("LV")
    interests = load_interests()

    occ.to_csv(PROCESSED / "occupations.csv", index=False)
    corpus.to_csv(PROCESSED / "corpus.csv", index=False)
    feats_im.to_csv(PROCESSED / "features_IM.csv")
    feats_lv.to_csv(PROCESSED / "features_LV.csv")
    interests.to_csv(PROCESSED / "interests.csv")

    with_tasks = corpus[corpus["n_tasks"] > 0]
    print(f"Occupations:                 {len(occ)}")
    print(f"  with task statements:      {len(with_tasks)}  "
          f"(median {int(with_tasks['n_tasks'].median())} tasks each)")
    print(f"  with numeric ratings:      {len(feats_im)}  ({feats_im.shape[1]} attributes)")
    print(f"  with interest scores:      {len(interests)}")
    print(f"  missing values in features: {int(feats_im.isna().sum().sum())}")
    print("\nAttributes per group:")
    print(pd.Series([c.split(':')[0] for c in feats_im.columns]).value_counts().to_string())
    print("\nOccupations by top interest:")
    print(interests["top_interest"].value_counts().to_string())
    print(f"\nSaved to {PROCESSED}")


if __name__ == "__main__":
    main()
