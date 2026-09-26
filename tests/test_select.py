"""Tests for the model-selection sweep."""

from __future__ import annotations

from dataclasses import replace

from ccfraud.config import DEFAULT_CONFIG
from ccfraud.data import split_xy
from ccfraud.select import CANDIDATES, _compatible, select_model, to_spec
from tests._synthetic import make_creditcard_frame


def test_select_model_returns_sorted_ranking():
    df = make_creditcard_frame(n=1500, pos=90)
    X, y = split_xy(df)
    config = replace(DEFAULT_CONFIG, cv_folds=3)

    best, ranking = select_model(
        X.to_numpy(),
        y.to_numpy(),
        config=config,
        names=["logreg", "lightgbm"],
        strategies=["none", "smote"],
    )

    assert len(ranking) == 4  # 2 estimators x 2 strategies
    means = [r.pr_auc_mean for r in ranking]
    assert all(a >= b for a, b in zip(means, means[1:], strict=False))
    assert best is ranking[0]
    assert 0.0 <= best.pr_auc_mean <= 1.0
    assert len(best.fold_scores) == 3


def test_select_model_beats_prevalence():
    df = make_creditcard_frame(n=2000, pos=120, seed=3)
    X, y = split_xy(df)
    config = replace(DEFAULT_CONFIG, cv_folds=3)

    best, _ = select_model(
        X.to_numpy(), y.to_numpy(), config=config, names=["lightgbm"], strategies=["none"]
    )
    prevalence = y.mean()
    assert best.pr_auc_mean > 3 * prevalence


def test_compatible_is_true_for_every_real_candidate_under_class_weight():
    # all four registered candidates expose class_weight or scale_pos_weight
    for name in CANDIDATES:
        assert _compatible(name, "class_weight") is True


def test_compatible_ignores_strategy_other_than_class_weight():
    for name in CANDIDATES:
        assert _compatible(name, "smote") is True
        assert _compatible(name, "none") is True


def test_to_spec_builds_a_working_pipeline():
    df = make_creditcard_frame(n=800, pos=40, seed=5)
    X, y = split_xy(df)
    config = replace(DEFAULT_CONFIG, cv_folds=3)

    best, _ = select_model(
        X.to_numpy(), y.to_numpy(), config=config, names=["logreg"], strategies=["none"]
    )
    spec = to_spec(best, seed=config.seed)
    assert spec.name == "logreg"

    pipe = spec.build()
    pipe.fit(X.to_numpy(), y.to_numpy())
    proba = pipe.predict_proba(X.to_numpy())
    assert proba.shape == (len(y), 2)
    # build() must return a fresh estimator each call, not a shared fitted one
    assert spec.build() is not pipe
