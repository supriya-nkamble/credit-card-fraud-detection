"""Tests for the stacked ensemble."""

from __future__ import annotations

import numpy as np
import pytest
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression

from ccfraud.ensemble import build_stack
from tests._synthetic import make_creditcard_frame


def test_build_stack_needs_two_members():
    with pytest.raises(ValueError):
        build_stack([("lgbm", LGBMClassifier(), "none")])


def test_stack_fits_and_predicts_proba():
    df = make_creditcard_frame(n=1500, pos=90, seed=2)
    X = df.drop(columns="Class").to_numpy()
    y = df["Class"].to_numpy()

    stack = build_stack(
        [
            ("lgbm", LGBMClassifier(n_estimators=60, verbosity=-1), "smote"),
            ("logreg", LogisticRegression(max_iter=500), "none"),
        ],
        seed=0,
        cv_folds=3,
    )
    stack.fit(X, y)
    proba = stack.predict_proba(X)
    assert proba.shape == (len(y), 2)
    assert np.all((proba >= 0) & (proba <= 1))
    # on data with real signal the stack should beat chance
    from sklearn.metrics import average_precision_score

    assert average_precision_score(y, proba[:, 1]) > y.mean() * 2
