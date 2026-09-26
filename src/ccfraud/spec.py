"""A model choice, decoupled from how it was chosen.

``select_model`` (cross-validated ranking) and ``tune`` (Optuna search) pick a
model in two different ways, but everything downstream - the threshold search,
calibration, ensembling, reporting - only needs to know its name, its
imbalance strategy, its CV score, and how to build a fresh copy of it. Both
paths convert their result to a ``ModelSpec`` (via each module's ``to_spec``),
so ``train.run()`` never has to know which path produced it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class ModelSpec:
    name: str
    strategy: str
    cv_pr_auc_mean: float
    cv_pr_auc_std: float
    build: Callable[[], Any]  # -> a fresh, unfitted pipeline

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "strategy": self.strategy,
            "cv_pr_auc_mean": self.cv_pr_auc_mean,
            "cv_pr_auc_std": self.cv_pr_auc_std,
        }
