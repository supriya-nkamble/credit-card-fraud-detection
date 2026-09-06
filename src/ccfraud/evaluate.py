"""Evaluation helpers for a highly imbalanced binary problem.

Key differences from the original notebook:

* ROC-AUC and PR-AUC are computed from **predicted probabilities**, never from
  hard 0/1 labels. ``roc_auc_score(y_true, y_pred)`` on labels is not a real AUC.
* PR-AUC (average precision) is the headline metric, because at 0.17 % positives
  ROC-AUC and accuracy are both close to meaningless.
* The decision threshold is chosen from the precision-recall curve to satisfy a
  minimum precision, not left at the default 0.5.
* Every report carries a money cost, so two models with a similar F1 can still
  be ranked.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)

from .config import CostMatrix


def _positive_scores(model: Any, X: ArrayLike) -> np.ndarray:
    """Return P(class = 1) for each row, whatever scoring API the model exposes."""
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X))[:, 1]
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(X))
        # Map an unbounded margin to (0, 1) so a 0.5 threshold still makes sense.
        return 1.0 / (1.0 + np.exp(-scores))
    raise TypeError(f"{type(model).__name__} has neither predict_proba nor decision_function")


@dataclass(frozen=True)
class Report:
    """Metrics for one model at one threshold."""

    pr_auc: float
    roc_auc: float
    precision: float
    recall: float
    f1: float
    threshold: float
    expected_cost: float
    support_pos: int
    confusion: list[list[int]]  # [[tn, fp], [fn, tp]]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def expected_cost(
    y_true: ArrayLike,
    proba: ArrayLike,
    threshold: float,
    cost_matrix: CostMatrix,
) -> float:
    """Total money cost of the confusion matrix implied by ``threshold``."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float(
        tn * cost_matrix.true_negative
        + fp * cost_matrix.false_positive
        + fn * cost_matrix.false_negative
        + tp * cost_matrix.true_positive
    )


def pick_threshold(
    y_true: ArrayLike,
    proba: ArrayLike,
    min_precision: float,
) -> float:
    """Lowest threshold whose precision is >= ``min_precision``.

    "Lowest" maximises recall among the points that clear the precision bar.
    If no threshold reaches ``min_precision`` (the target is simply not
    achievable on this data), fall back to the threshold with the best F1 so
    the model still predicts something useful.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    # precision_recall_curve returns len(thresholds) == len(precision) - 1;
    # align by dropping the final point (recall == 0, precision == 1 by fiat).
    precision = precision[:-1]
    recall = recall[:-1]
    if thresholds.size == 0:
        return 0.5

    ok = np.where(precision >= min_precision)[0]
    if ok.size:
        return float(thresholds[ok[0]])

    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )
    return float(thresholds[int(np.argmax(f1))])


def evaluate(
    model: Any,
    X: ArrayLike,
    y: ArrayLike,
    threshold: float = 0.5,
    cost_matrix: CostMatrix | None = None,
) -> Report:
    """Score a fitted model on ``(X, y)`` at a given decision threshold."""
    cost_matrix = cost_matrix or CostMatrix()
    y = np.asarray(y).astype(int)
    proba = _positive_scores(model, X)
    y_pred = (proba >= threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y, y_pred, labels=[1], average="binary", zero_division=0
    )
    cm = confusion_matrix(y, y_pred, labels=[0, 1])
    return Report(
        pr_auc=float(average_precision_score(y, proba)),
        roc_auc=float(roc_auc_score(y, proba)),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        threshold=float(threshold),
        expected_cost=expected_cost(y, proba, threshold, cost_matrix),
        support_pos=int(y.sum()),
        confusion=cm.astype(int).tolist(),
    )


def dummy_baseline(
    X_train: ArrayLike,
    y_train: ArrayLike,
    X_test: ArrayLike,
    y_test: ArrayLike,
    strategy: str = "most_frequent",
    seed: int = 42,
) -> Report:
    """Trivial reference model.

    ``most_frequent`` always predicts "genuine": ~99.83 % accuracy, 0 % fraud
    recall, PR-AUC ~= the positive prevalence. It exists to kill "accuracy" as
    a headline number.
    """
    dummy = DummyClassifier(strategy=strategy, random_state=seed)
    dummy.fit(X_train, y_train)
    return evaluate(dummy, X_test, y_test, threshold=0.5)
