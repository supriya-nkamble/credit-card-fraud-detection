"""Shared fixtures."""

from __future__ import annotations

import pytest

from tests._synthetic import make_creditcard_frame


@pytest.fixture
def creditcard_csv(tmp_path):
    path = tmp_path / "creditcard.csv"
    make_creditcard_frame().to_csv(path, index=False)
    return path
