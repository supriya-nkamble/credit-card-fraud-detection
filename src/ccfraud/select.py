"""Stratified K-fold model selection scored on PR-AUC.

For every ``(estimator, imbalance_strategy)`` pair we run a stratified K-fold
loop and average the fold PR-AUC. Scaling and resampling happen inside each
fold (see :mod:`ccfraud.pipeline`), so the score is an honest estimate of
held-out performance rather than a leaked one.

The gradient-boosted trees are trained to a real size, not the 20 rounds the
notebook used.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

from .config import Config
from .pipeline import IMBALANCE_STRATEGIES, build_pipeline


def _make_xgb() -> Any:
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="aucpr",
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
    )


def _make_lgbm() -> Any:
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        n_estimators=600,
        num_leaves=64,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        n_jobs=-1,
        random_state=42,
        verbosity=-1,
    )


# name -> factory. Factories are used so each fold gets a fresh, unfitted model.
CANDIDATES: dict[str, Callable[[], Any]] = {
    "logreg": lambda: LogisticRegression(max_iter=1000, n_jobs=-1),
    "random_forest": lambda: RandomForestClassifier(
        n_estimators=300, n_jobs=-1, random_state=42
    ),
    "xgboost": _make_xgb,
    "lightgbm": _make_lgbm,
}


@dataclass
class FoldResult:
    name: str
    strategy: str
    pr_auc_mean: float
    pr_auc_std: float
    fold_scores: list[float] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "strategy": self.strategy,
            "pr_auc_mean": self.pr_auc_mean,
            "pr_auc_std": self.pr_auc_std,
            "fold_scores": self.fold_scores,
        }


def _score_pair(
    name: str,
    strategy: str,
    X: np.ndarray,
    y: np.ndarray,
    cv: StratifiedKFold,
    seed: int,
) -> FoldResult:
    scores: list[float] = []
    for train_idx, val_idx in cv.split(X, y):
        pipe = build_pipeline(CANDIDATES[name](), strategy, seed=seed)
        pipe.fit(X[train_idx], y[train_idx])
        proba = pipe.predict_proba(X[val_idx])[:, 1]
        scores.append(float(average_precision_score(y[val_idx], proba)))
    return FoldResult(
        name=name,
        strategy=strategy,
        pr_auc_mean=float(np.mean(scores)),
        pr_auc_std=float(np.std(scores)),
        fold_scores=scores,
    )


def _compatible(name: str, strategy: str) -> bool:
    if strategy != "class_weight":
        return True
    # logreg / random_forest expose class_weight; xgboost exposes
    # scale_pos_weight; lightgbm exposes class_weight -> all four are fine.
    return True


def select_model(
    X: ArrayLike,
    y: ArrayLike,
    config: Config | None = None,
    names: list[str] | None = None,
    strategies: list[str] | None = None,
) -> tuple[FoldResult, list[FoldResult]]:
    """Rank every candidate pair by mean CV PR-AUC.

    Returns ``(best, ranking)`` where ``ranking`` is sorted best-first.
    """
    config = config or Config()
    names = names or list(CANDIDATES)
    strategies = strategies or list(IMBALANCE_STRATEGIES)

    X = np.asarray(X)
    y = np.asarray(y).astype(int)
    cv = StratifiedKFold(n_splits=config.cv_folds, shuffle=True, random_state=config.seed)

    results: list[FoldResult] = []
    for name in names:
        for strategy in strategies:
            if not _compatible(name, strategy):
                continue
            results.append(_score_pair(name, strategy, X, y, cv, config.seed))

    ranking = sorted(results, key=lambda r: r.pr_auc_mean, reverse=True)
    return ranking[0], ranking
