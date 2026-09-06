"""Load the ULB credit-card dataset and make a leakage-free split.

The dataset is not stored in Git. Download it once from Kaggle
(``mlg-ulb/creditcardfraud``) and place ``creditcard.csv`` in the repo root,
or pass an explicit path.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

EXPECTED_COLUMNS = 31  # Time, V1..V28, Amount, Class


def load_creditcard(path: str | Path) -> pd.DataFrame:
    """Read the dataset CSV and check its shape.

    Raises
    ------
    FileNotFoundError
        If the file is missing (with a hint on where to get it).
    ValueError
        If the column layout is not the expected ULB layout.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download the ULB dataset from Kaggle "
            "(mlg-ulb/creditcardfraud) and place creditcard.csv there."
        )
    df = pd.read_csv(path)
    if df.shape[1] != EXPECTED_COLUMNS or "Class" not in df.columns:
        raise ValueError(
            f"Unexpected columns: got {df.shape[1]} columns, expected {EXPECTED_COLUMNS} "
            "including a 'Class' target."
        )
    return df


def split_xy(
    df: pd.DataFrame, target: str = "Class"
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate the feature matrix from the target vector."""
    return df.drop(columns=[target]), df[target]


def stratified_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    seed: int = 42,
    target: str = "Class",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split into train/test while preserving the class ratio in both parts.

    The notebook used ``train_test_split(..., random_state=1)`` with no
    ``stratify``; with 492 positives that risks an uneven fraud count between
    the two parts. ``stratify=y`` removes that risk.
    """
    X, y = split_xy(df, target)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    return X_train, X_test, y_train, y_test
