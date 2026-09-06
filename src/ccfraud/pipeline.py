"""Build a leakage-free scale -> resample -> classify pipeline.

The notebook scaled the full training matrix and then ran SMOTE on it once,
before any cross-validation. Both steps therefore saw data that later became
validation folds. Wrapping them in an ``imblearn`` pipeline moves scaling and
resampling *inside* each fold: ``fit_resample`` is called on the training part
only, and the validation part keeps the real 0.17 % fraud rate.
"""

from __future__ import annotations

from typing import Any

from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.preprocessing import RobustScaler

IMBALANCE_STRATEGIES = ("none", "smote", "smotetomek", "class_weight")


def _supports_class_weight(estimator: Any) -> bool:
    params = estimator.get_params()
    return "class_weight" in params or "scale_pos_weight" in params


def build_pipeline(
    estimator: Any,
    imbalance_strategy: str = "none",
    seed: int = 42,
) -> Pipeline:
    """Return an ``imblearn`` Pipeline for one (estimator, strategy) pair.

    Parameters
    ----------
    estimator:
        Any scikit-learn compatible classifier exposing ``predict_proba``.
    imbalance_strategy:
        ``"none"``        - scale then classify, no rebalancing
        ``"smote"``       - SMOTE oversampling on the training fold
        ``"smotetomek"``  - SMOTE + Tomek-link cleaning on the training fold
        ``"class_weight"``- no resampling; ask the estimator to reweight classes
    """
    if imbalance_strategy not in IMBALANCE_STRATEGIES:
        raise ValueError(
            f"unknown imbalance_strategy {imbalance_strategy!r}; "
            f"choose from {IMBALANCE_STRATEGIES}"
        )

    est = clone(estimator)
    steps: list[tuple[str, Any]] = [("scaler", RobustScaler())]

    if imbalance_strategy == "smote":
        steps.append(("resample", SMOTE(random_state=seed)))
    elif imbalance_strategy == "smotetomek":
        steps.append(("resample", SMOTETomek(random_state=seed)))
    elif imbalance_strategy == "class_weight":
        params = est.get_params()
        if "class_weight" in params:
            est.set_params(class_weight="balanced")
        elif "scale_pos_weight" in params:
            # xgboost: recommended value is n_negative / n_positive; the
            # selection step passes the actual ratio via set_params, this is
            # only a safe non-1 default.
            est.set_params(scale_pos_weight=10.0)
        elif not _supports_class_weight(est):
            raise ValueError(
                f"{type(est).__name__} has no class_weight/scale_pos_weight; "
                "use a resampling strategy instead"
            )

    steps.append(("clf", est))
    return Pipeline(steps)
