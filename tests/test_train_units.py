"""Unit tests for the small functions train.run() composes.

Each of these used to be inline code inside one 165-line run(); now that they
are named functions, they can be tested in isolation instead of only through
the end-to-end smoke test.
"""

from __future__ import annotations

from dataclasses import replace

from sklearn.model_selection import StratifiedKFold

from ccfraud.config import DEFAULT_CONFIG
from ccfraud.data import split_xy
from ccfraud.train import _fit_and_evaluate, _fit_ensemble, _pick_spec, _threshold_from_oof
from tests._synthetic import make_creditcard_frame


def _train_arrays(n=1500, pos=90, seed=1):
    df = make_creditcard_frame(n=n, pos=pos, seed=seed)
    X, y = split_xy(df)
    return X.to_numpy(), y.to_numpy()


def test_pick_spec_default_returns_ranking_and_members():
    X, y = _train_arrays()
    config = replace(DEFAULT_CONFIG, cv_folds=3)

    spec, meta, members = _pick_spec(X, y, config, fast=True, tune=False, n_trials=1)

    assert spec.name in {"logreg", "lightgbm"}
    assert meta["tuned"] is None
    assert len(meta["cv_ranking"]) == 4  # 2 candidates x 2 strategies in fast mode
    assert all(m.name == "lightgbm" for m in members)  # only lightgbm is a GBM here


def test_pick_spec_tune_returns_tuned_metadata():
    X, y = _train_arrays(n=1000, pos=60, seed=2)
    config = replace(DEFAULT_CONFIG, cv_folds=3)

    spec, meta, members = _pick_spec(X, y, config, fast=True, tune=True, n_trials=2)

    assert spec.name == "lightgbm (tuned)"  # fast mode tunes lightgbm only
    assert "lightgbm" in meta["tuned"]
    assert meta["cv_ranking"] == []
    assert len(members) == 1


def test_threshold_from_oof_is_a_valid_probability():
    X, y = _train_arrays(n=1200, pos=70, seed=3)
    config = replace(DEFAULT_CONFIG, cv_folds=3)
    spec, _, _ = _pick_spec(X, y, config, fast=True, tune=False, n_trials=1)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=config.seed)

    threshold, precision = _threshold_from_oof(spec, X, y, cv, config)

    assert 0.0 <= threshold <= 1.0
    assert 0.0 <= precision <= 1.0


def test_fit_and_evaluate_returns_reports_and_a_working_model():
    X, y = _train_arrays(n=1200, pos=70, seed=4)
    config = replace(DEFAULT_CONFIG, cv_folds=3)
    spec, _, _ = _pick_spec(X, y, config, fast=True, tune=False, n_trials=1)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=config.seed)

    plain_rep, calibrated_rep, brier, model = _fit_and_evaluate(
        spec, X, y, X, y, threshold=0.5, cost=config.cost_matrix, cv=cv
    )

    for rep in (plain_rep, calibrated_rep):
        assert 0.0 <= rep.pr_auc <= 1.0
    assert 0.0 <= brier <= 1.0
    assert model.predict_proba(X).shape == (len(y), 2)


def test_fit_ensemble_skips_with_fewer_than_two_members():
    X, y = _train_arrays(n=600, pos=30, seed=5)
    config = replace(DEFAULT_CONFIG, cv_folds=3)
    _, _, members = _pick_spec(X, y, config, fast=True, tune=True, n_trials=1)  # 1 member

    report, model = _fit_ensemble(members, X, y, X, y, 0.5, config.cost_matrix, config)

    assert report is None
    assert model is None


def test_fit_ensemble_stacks_two_members():
    X, y = _train_arrays(n=1500, pos=90, seed=6)
    config = replace(DEFAULT_CONFIG, cv_folds=3)
    _, _, members = _pick_spec(
        X, y, config, fast=False, tune=False, n_trials=1
    )  # default sweep -> lightgbm + xgboost members

    report, model = _fit_ensemble(members, X, y, X, y, 0.5, config.cost_matrix, config)

    assert report is not None
    assert 0.0 <= report.pr_auc <= 1.0
    assert model.predict_proba(X).shape == (len(y), 2)
