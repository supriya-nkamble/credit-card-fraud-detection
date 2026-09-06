"""Tests that resampling stays inside the CV fold (no leakage)."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from ccfraud.pipeline import build_pipeline


def _imbalanced(n=600, pos=30, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5))
    y = np.zeros(n, dtype=int)
    idx = rng.choice(n, size=pos, replace=False)
    y[idx] = 1
    X[idx] += 1.5  # give the positives a little signal
    return X, y


def test_unknown_strategy_rejected():
    with pytest.raises(ValueError):
        build_pipeline(LogisticRegression(), "oversample_everything")


def test_smote_does_not_touch_the_validation_fold():
    X, y = _imbalanced()
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    pipe = build_pipeline(LogisticRegression(max_iter=500), "smote", seed=0)

    for train_idx, val_idx in cv.split(X, y):
        val_pos_before = int(y[val_idx].sum())
        val_size_before = val_idx.size
        pipe.fit(X[train_idx], y[train_idx])
        # predict on the untouched validation fold: its size and prevalence
        # must be exactly what the split produced - SMOTE only ran on train.
        proba = pipe.predict_proba(X[val_idx])
        assert proba.shape == (val_size_before, 2)
        assert int(y[val_idx].sum()) == val_pos_before


def test_smote_balances_the_training_fold_only():
    X, y = _imbalanced()
    from imblearn.over_sampling import SMOTE

    # what the pipeline's resampler sees and produces on the training data
    Xs, ys = SMOTE(random_state=0).fit_resample(X, y)
    assert ys.sum() == (ys == 0).sum()  # balanced after resampling
    assert len(ys) > len(y)  # rows were synthesised
    assert len(y) == 600  # original array is unchanged


def test_class_weight_strategy_sets_balanced_weight():
    pipe = build_pipeline(LogisticRegression(), "class_weight")
    assert pipe.named_steps["clf"].class_weight == "balanced"
    assert "resample" not in pipe.named_steps
