"""Synthetic data shaped like the ULB credit-card file, for offline tests."""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_creditcard_frame(n: int = 3000, pos: int = 45, seed: int = 0) -> pd.DataFrame:
    """A frame with the ULB layout (Time, V1..V28, Amount, Class).

    A few ``V`` columns carry real signal for the positive class so a
    classifier can learn something in the smoke tests.
    """
    rng = np.random.default_rng(seed)
    cols: dict[str, np.ndarray] = {"Time": rng.uniform(0, 1.7e5, n)}
    for i in range(1, 29):
        cols[f"V{i}"] = rng.normal(size=n)
    cols["Amount"] = rng.gamma(2.0, 40.0, n)

    y = np.zeros(n, dtype=int)
    pos_idx = rng.choice(n, size=pos, replace=False)
    y[pos_idx] = 1

    frame = pd.DataFrame(cols)
    for col in ("V1", "V3", "V10", "V14", "V17"):
        frame.loc[pos_idx, col] += rng.normal(2.5, 0.5, size=pos)
    frame["Class"] = y
    return frame
