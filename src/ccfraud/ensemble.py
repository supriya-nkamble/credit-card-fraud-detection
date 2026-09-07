"""Stacked ensemble of the gradient-boosted candidates.

Each member is a leakage-free ``build_pipeline`` (scale -> resample -> estimator);
``StackingClassifier`` trains them, takes their out-of-fold probabilities as
features, and fits a logistic-regression meta-model on top. Resampling still
happens only on each member's internal training folds.

On the ULB dataset stacking typically adds ~0.005-0.01 PR-AUC over the best
single model - a real but small gain.
"""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from .pipeline import build_pipeline

Member = tuple[str, Any, str]  # (name, unfitted estimator, imbalance_strategy)


def build_stack(members: list[Member], seed: int = 42, cv_folds: int = 5) -> StackingClassifier:
    """Assemble a StackingClassifier from ``(name, estimator, strategy)`` members."""
    if len(members) < 2:
        raise ValueError("a stack needs at least two members")

    estimators = [
        (name, build_pipeline(est, strategy, seed=seed))
        for name, est, strategy in members
    ]
    return StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000, class_weight="balanced"),
        stack_method="predict_proba",
        cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed),
        n_jobs=1,  # each member already runs with n_jobs=-1
        passthrough=False,
    )
