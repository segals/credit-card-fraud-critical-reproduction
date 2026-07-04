"""Fetch and cache the ULB credit-card-fraud dataset (Kaggle mlg-ulb / OpenML id 1597).

Using OpenML keeps the project reproducible without a Kaggle account or API key.
The raw CSV is cached under data/ so the notebook can load it offline afterwards.
"""
from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CSV_PATH = DATA_DIR / "creditcard.csv"
# OpenML id 42175 = full ULB dataset with all 31 columns: Time, V1..V28, Amount, Class.
# (The shorter "creditcard" mirror, id 1597, silently drops the Time column, which we need
#  for the assignment's temporal analysis, so we deliberately use 42175 instead.)
OPENML_ID = 42175


def load_creditcard(force_download: bool = False) -> pd.DataFrame:
    """Return the fraud dataframe, downloading from OpenML on first use."""
    if CSV_PATH.exists() and not force_download:
        return pd.read_csv(CSV_PATH)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    bunch = fetch_openml(data_id=OPENML_ID, as_frame=True, parser="auto")
    frame = bunch.frame.copy()
    # OpenML labels the target column "Class"; normalise dtype to int {0,1}.
    frame["Class"] = frame["Class"].astype(int)
    frame.to_csv(CSV_PATH, index=False)
    return frame


if __name__ == "__main__":
    df = load_creditcard()
    print("shape:", df.shape)
    print("columns:", list(df.columns))
    print("class counts:\n", df["Class"].value_counts())
    print("fraud prevalence: {:.4%}".format(df["Class"].mean()))
