"""End-to-end smoke test: the whole pipeline runs on synthetic data."""

from __future__ import annotations

import json
from dataclasses import replace

from ccfraud.config import DEFAULT_CONFIG
from ccfraud.train import run


def test_run_writes_metrics_json(creditcard_csv, tmp_path):
    out_dir = tmp_path / "output"
    config = replace(
        DEFAULT_CONFIG,
        data_path=creditcard_csv,
        output_dir=out_dir,
        cv_folds=3,
        test_size=0.25,
    )

    result = run(config, fast=True)

    metrics_path = out_dir / "metrics.json"
    assert metrics_path.exists()
    saved = json.loads(metrics_path.read_text())

    assert saved["selected"]["name"] in {"logreg", "lightgbm"}
    for key in ("baseline_most_frequent", "baseline_logreg", "selected_calibrated"):
        assert key in saved["reports"]

    # run() returns reports already serialised to plain dicts
    baseline = result["reports"]["baseline_most_frequent"]
    selected = result["reports"]["selected_calibrated"]
    assert baseline["recall"] == 0.0
    assert selected["pr_auc"] > baseline["pr_auc"]  # the model beats "always genuine"
    assert 0.0 <= result["threshold"] <= 1.0
