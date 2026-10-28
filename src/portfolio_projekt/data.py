"""Loading functions for the toxic comment datasets."""

from pathlib import Path

import pandas as pd

# Project root, independent of where a notebook or script is started from
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

# severe_toxicity is not used (only 13 majority cases in the full dataset)
LABELS = ["toxicity", "obscene", "threat", "insult", "identity_attack", "sexual_explicit"]
THRESHOLD = 0.5


def load_english() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return the enriched training data and the official validation and test splits."""
    df_train = pd.read_parquet(DATA_DIR / "processed" / "civil_comments_train_enriched.parquet")
    df_val = pd.read_parquet(DATA_DIR / "raw" / "civil_comments" / "validation")
    df_test = pd.read_parquet(DATA_DIR / "raw" / "civil_comments" / "test")

    for df in (df_train, df_val, df_test):
        df["lang"] = "en"
        df["source"] = "civil_comments"

    return df_train, df_val, df_test


def load_german() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return the German training, validation and test splits."""
    df = pd.read_parquet(DATA_DIR / "processed" / "german_unified.parquet")
    splits = []
    for name in ("train", "validation", "test"):
        part = df[df["split"] == name].drop(columns="split").reset_index(drop=True)
        splits.append(part)
    return tuple(splits)