"""Optuna hyperparameter search for the gradient-boosted candidates.

The objective is the same mean CV PR-AUC the selection step uses, computed with
the leakage-free pipeline (scaling + resampling inside each fold). The imbalance
strategy is searched jointly with the model hyperparameters.

This is a genuine but small lever: on the ULB dataset the honest PR-AUC ceiling
is ~0.88, so expect single-percent-point gains, not a jump to 0.95+.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

from .config import Config
from .pipeline import build_pipeline

TUNABLE = ("lightgbm", "xgboost")


@dataclass
class TuneResult:
    estimator: str
    strategy: str
    params: dict[str, Any]
    pr_auc_mean: float
    pr_auc_std: float
    n_trials: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "estimator": self.estimator,
            "strategy": self.strategy,
            "params": self.params,
            "pr_auc_mean": self.pr_auc_mean,
            "pr_auc_std": self.pr_auc_std,
            "n_trials": self.n_trials,
        }


def _lightgbm_space(trial: Any) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1200, step=100),
        "num_leaves": trial.suggest_int("num_leaves", 16, 256, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }


def _xgboost_space(trial: Any) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000, step=100),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "gamma": trial.suggest_float("gamma", 1e-3, 5.0, log=True),
    }


_SPACES: dict[str, Callable[[Any], dict[str, Any]]] = {
    "lightgbm": _lightgbm_space,
    "xgboost": _xgboost_space,
}


def make_estimator(name: str, params: dict[str, Any], seed: int = 42) -> Any:
    """Instantiate a fresh estimator of ``name`` with ``params``."""
    if name == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(n_jobs=-1, random_state=seed, verbosity=-1, **params)
    if name == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_jobs=-1,
            random_state=seed,
            tree_method="hist",
            eval_metric="aucpr",
            **params,
        )
    raise ValueError(f"unknown estimator {name!r}; tunable: {TUNABLE}")


def tune(
    estimator_name: str,
    X: ArrayLike,
    y: ArrayLike,
    config: Config | None = None,
    n_trials: int = 30,
    strategies: list[str] | None = None,
) -> TuneResult:
    """Run an Optuna TPE search; return the best params, strategy, and CV PR-AUC."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    config = config or Config()
    strategies = strategies or list(config.imbalance_strategies)
    space = _SPACES.get(estimator_name)
    if space is None:
        raise ValueError(f"unknown estimator {estimator_name!r}; tunable: {TUNABLE}")

    X_arr = np.asarray(X)
    y_arr = np.asarray(y).astype(int)
    cv = StratifiedKFold(n_splits=config.cv_folds, shuffle=True, random_state=config.seed)

    def objective(trial: Any) -> float:
        params = space(trial)
        strategy = trial.suggest_categorical("imbalance_strategy", strategies)
        scores: list[float] = []
        for tr_idx, va_idx in cv.split(X_arr, y_arr):
            est = make_estimator(estimator_name, params, config.seed)
            pipe = build_pipeline(est, strategy, seed=config.seed)
            pipe.fit(X_arr[tr_idx], y_arr[tr_idx])
            proba = pipe.predict_proba(X_arr[va_idx])[:, 1]
            ap = float(average_precision_score(y_arr[va_idx], proba))
            scores.append(0.0 if np.isnan(ap) else ap)
        trial.set_user_attr("pr_auc_std", float(np.std(scores)))
        return float(np.mean(scores))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=config.seed),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_trial
    params = {k: v for k, v in best.params.items() if k != "imbalance_strategy"}
    return TuneResult(
        estimator=estimator_name,
        strategy=str(best.params["imbalance_strategy"]),
        params=params,
        pr_auc_mean=float(best.value if best.value is not None else 0.0),
        pr_auc_std=float(best.user_attrs.get("pr_auc_std", 0.0)),
        n_trials=n_trials,
    )
