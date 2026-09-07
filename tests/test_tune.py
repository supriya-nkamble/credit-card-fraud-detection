"""Tests for the Optuna search (tiny trial count, synthetic data)."""

from __future__ import annotations

from dataclasses import replace

import pytest

from ccfraud.config import DEFAULT_CONFIG
from ccfraud.data import split_xy
from ccfraud.tune import TUNABLE, make_estimator, tune
from tests._synthetic import make_creditcard_frame


@pytest.mark.parametrize("name", TUNABLE)
def test_tune_returns_valid_result(name):
    df = make_creditcard_frame(n=1200, pos=70, seed=1)
    X, y = split_xy(df)
    config = replace(DEFAULT_CONFIG, cv_folds=3)

    res = tune(
        name,
        X.to_numpy(),
        y.to_numpy(),
        config=config,
        n_trials=3,
        strategies=["none", "smote"],
    )

    assert res.estimator == name
    assert res.strategy in {"none", "smote"}
    assert isinstance(res.params, dict) and res.params
    assert 0.0 <= res.pr_auc_mean <= 1.0
    assert res.n_trials == 3
    # the returned params must actually build an estimator
    est = make_estimator(name, res.params, seed=0)
    assert hasattr(est, "fit")


def test_tune_rejects_unknown_estimator():
    df = make_creditcard_frame(n=400, pos=20)
    X, y = split_xy(df)
    with pytest.raises(ValueError):
        tune("catboost", X.to_numpy(), y.to_numpy(), n_trials=1)
