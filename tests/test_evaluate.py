"""Regression tests for the evaluation module.

These pin the bugs the original notebook shipped with:
* ROC-AUC computed from hard labels instead of probabilities
* no baseline model
* decision threshold left at 0.5
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from ccfraud.config import CostMatrix
from ccfraud.evaluate import (
    dummy_baseline,
    evaluate,
    expected_cost,
    pick_threshold,
)


def test_expected_cost_matches_hand_count():
    # y=[0,0,1,1], proba=[.1,.9,.2,.8], threshold .5 -> preds [0,1,0,1]
    # tn=1, fp=1, fn=1, tp=1
    cm = CostMatrix(false_negative=200, false_positive=5, true_positive=5, true_negative=0)
    cost = expected_cost([0, 0, 1, 1], [0.1, 0.9, 0.2, 0.8], 0.5, cm)
    assert cost == pytest.approx(1 * 5 + 1 * 200 + 1 * 5)


def test_pick_threshold_respects_min_precision():
    rng = np.random.default_rng(0)
    y = np.array([0] * 200 + [1] * 20)
    # Positives score higher on average but the classes overlap.
    proba = np.concatenate([rng.uniform(0.0, 0.6, 200), rng.uniform(0.4, 1.0, 20)])
    thr = pick_threshold(y, proba, min_precision=0.8)
    pred = (proba >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    precision = tp / (tp + fp)
    assert precision >= 0.8


def test_pick_threshold_falls_back_when_unreachable():
    y = np.array([0] * 50 + [1] * 5)
    proba = np.full(55, 0.5)  # no separation at all
    thr = pick_threshold(y, proba, min_precision=0.99)
    assert np.isfinite(thr)


def test_roc_auc_uses_probabilities_not_labels():
    # A model whose ranking is perfect but whose 0.5-thresholded labels are
    # all wrong must still score ROC-AUC = 1.0. That is only true if evaluate()
    # feeds probabilities (not hard labels) to roc_auc_score.
    class RankingModel:
        classes_ = np.array([0, 1])

        def predict_proba(self, X):
            p = np.asarray(X, dtype=float).ravel()
            return np.column_stack([1 - p, p])

    y = np.array([0, 0, 1, 1])
    X = np.array([0.10, 0.20, 0.30, 0.40])  # positives rank above negatives...
    rep = evaluate(RankingModel(), X, y, threshold=0.5)  # ...but none exceed 0.5
    assert rep.roc_auc == pytest.approx(1.0)
    assert rep.recall == 0.0  # confirms the 0.5 threshold really does miss them


def test_dummy_baseline_has_zero_fraud_recall():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(400, 4))
    y = np.array([0] * 390 + [1] * 10)
    rep = dummy_baseline(X, y, X, y, strategy="most_frequent")
    assert rep.recall == 0.0
    assert rep.pr_auc == pytest.approx(10 / 400, abs=1e-6)


def test_evaluate_on_real_classifier_is_finite():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(300, 5))
    y = (X[:, 0] + rng.normal(scale=0.5, size=300) > 1.0).astype(int)
    model = LogisticRegression().fit(X, y)
    rep = evaluate(model, X, y, threshold=0.5)
    for value in (rep.pr_auc, rep.roc_auc, rep.precision, rep.recall, rep.f1):
        assert 0.0 <= value <= 1.0
