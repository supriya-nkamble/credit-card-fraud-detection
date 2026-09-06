"""Tests for loading and splitting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ccfraud.data import load_creditcard, stratified_split


def _synthetic_frame(n: int = 2000, pos: int = 20, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cols = {"Time": rng.uniform(0, 1e5, n)}
    for i in range(1, 29):
        cols[f"V{i}"] = rng.normal(size=n)
    cols["Amount"] = rng.uniform(0, 500, n)
    y = np.zeros(n, dtype=int)
    y[rng.choice(n, size=pos, replace=False)] = 1
    cols["Class"] = y
    return pd.DataFrame(cols)


def test_stratified_split_preserves_class_ratio():
    df = _synthetic_frame(n=4000, pos=40)
    X_tr, X_te, y_tr, y_te = stratified_split(df, test_size=0.25, seed=42)
    assert len(X_tr) + len(X_te) == len(df)
    base = df["Class"].mean()
    assert y_tr.mean() == pytest.approx(base, rel=0.15)
    assert y_te.mean() == pytest.approx(base, rel=0.15)
    assert y_te.sum() >= 1  # the rare class must appear in the test part


def test_stratified_split_is_deterministic():
    df = _synthetic_frame()
    a = stratified_split(df, seed=7)
    b = stratified_split(df, seed=7)
    assert list(a[3].index) == list(b[3].index)


def test_load_creditcard_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_creditcard(tmp_path / "nope.csv")


def test_load_creditcard_wrong_shape(tmp_path):
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(bad, index=False)
    with pytest.raises(ValueError):
        load_creditcard(bad)
