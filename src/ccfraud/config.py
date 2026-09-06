"""Frozen run configuration for the fraud-detection pipeline.

Every tunable lives here so the logic modules stay free of hardcoded values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# Repository root = two levels up from this file (src/ccfraud/config.py).
REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class CostMatrix:
    """Money cost of each prediction outcome, in currency units.

    A missed fraud (false negative) is far more expensive than a
    false alarm (false positive). The defaults below are illustrative,
    not measured; override them with numbers from the fraud-ops team.
    """

    false_negative: float = 200.0  # chargeback + investigation for a missed fraud
    false_positive: float = 5.0  # manual review of a blocked genuine transaction
    true_positive: float = 5.0  # review cost of a correctly flagged fraud
    true_negative: float = 0.0


@dataclass(frozen=True)
class Config:
    """End-to-end configuration for training and evaluation."""

    data_path: Path = REPO_ROOT / "creditcard.csv"
    output_dir: Path = REPO_ROOT / "output"

    target: str = "Class"
    positive_label: int = 1

    seed: int = 42
    test_size: float = 0.2
    cv_folds: int = 5

    # Model *selection* runs on a stratified subsample this large (all positives
    # kept). The PR-AUC ranking is stable well below the full 227k rows, and the
    # winner is still trained, calibrated, and evaluated on the full data.
    select_sample: int = 80_000

    # Lowest precision the fraud class must reach at the chosen threshold.
    min_precision: float = 0.90

    cost_matrix: CostMatrix = field(default_factory=CostMatrix)

    # Imbalance strategies tried during model selection. "smotetomek" is
    # supported by the pipeline but left out of the default sweep: on 200k+ rows
    # its Tomek-link cleaning is very slow and rarely beats plain SMOTE.
    imbalance_strategies: tuple[str, ...] = ("none", "smote", "class_weight")


DEFAULT_CONFIG = Config()
