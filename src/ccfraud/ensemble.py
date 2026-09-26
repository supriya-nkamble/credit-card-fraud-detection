"""Stacked ensemble of the gradient-boosted candidates.

Each member is a ``ModelSpec`` (see :mod:`ccfraud.spec`); ``spec.build()``
already returns a leakage-free ``scale -> resample -> estimator`` pipeline, so
this module only wires members into a ``StackingClassifier`` - it does not
build pipelines itself. ``StackingClassifier`` trains the members, takes their
out-of-fold probabilities as features, and fits a logistic-regression
meta-model on top. Resampling still happens only on each member's internal
training folds.

On the ULB dataset stacking has not beaten the best single member (see the
README) - it is here as a documented, measured experiment, not a claim.
"""

from __future__ import annotations

from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from .spec import ModelSpec


def build_stack(members: list[ModelSpec], seed: int = 42, cv_folds: int = 5) -> StackingClassifier:
    """Assemble a StackingClassifier from two or more ``ModelSpec`` members."""
    if len(members) < 2:
        raise ValueError("a stack needs at least two members")

    estimators = [(m.name, m.build()) for m in members]
    return StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000, class_weight="balanced"),
        stack_method="predict_proba",
        cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed),
        n_jobs=1,  # each member already runs with n_jobs=-1
        passthrough=False,
    )
