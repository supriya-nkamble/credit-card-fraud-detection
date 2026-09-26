"""Train and evaluate the fraud detector end to end.

Steps
-----
1. Load the dataset and make a stratified train/test split.
2. Score a most-frequent baseline and a logistic-regression baseline.
3. Pick the model (``_pick_spec``):
   * default    - select the best (estimator, imbalance strategy) by CV PR-AUC
   * ``--tune`` - Optuna search over LightGBM and XGBoost, keep the better
4. Choose the decision threshold from out-of-fold probabilities so the fraud
   precision meets ``config.min_precision`` (``_threshold_from_oof``).
5. Fit the plain model and an isotonic-calibrated copy (``_fit_and_evaluate``).
6. Optionally (``--ensemble``) stack the gradient-boosted models
   (``_fit_ensemble``).
7. Evaluate on the held-out test set and write ``output/metrics.json`` + curves.

Selection and tuning both return a :class:`ccfraud.spec.ModelSpec`, so steps
4-6 do not need to know which one produced the chosen model.

Usage
-----
    ccfraud-train --help
    python -m ccfraud.train                    # default selection
    python -m ccfraud.train --tune --ensemble  # search + stack (slow)
    python scripts/train.py --fast             # tiny smoke run
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from ccfraud.config import DEFAULT_CONFIG, Config
from ccfraud.data import load_creditcard, stratified_split
from ccfraud.evaluate import Report, dummy_baseline, evaluate, pick_threshold
from ccfraud.pipeline import build_pipeline
from ccfraud.spec import ModelSpec


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--data-path", type=Path, default=DEFAULT_CONFIG.data_path)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_CONFIG.output_dir)
    p.add_argument("--test-size", type=float, default=DEFAULT_CONFIG.test_size)
    p.add_argument("--seed", type=int, default=DEFAULT_CONFIG.seed)
    p.add_argument("--cv-folds", type=int, default=DEFAULT_CONFIG.cv_folds)
    p.add_argument("--min-precision", type=float, default=DEFAULT_CONFIG.min_precision)
    p.add_argument("--tune", action="store_true", help="Optuna search over LightGBM + XGBoost")
    p.add_argument("--n-trials", type=int, default=30, help="Optuna trials per estimator")
    p.add_argument("--ensemble", action="store_true", help="also fit a stacked ensemble")
    p.add_argument(
        "--fast",
        action="store_true",
        help="small candidate grid + subsampled negatives; for CI / smoke tests only",
    )
    return p.parse_args(argv)


def _subsample_majority(X, y, keep_negatives: int, seed: int):
    rng = np.random.default_rng(seed)
    neg = np.where(y == 0)[0]
    pos = np.where(y == 1)[0]
    take = min(keep_negatives, neg.size)
    keep = np.concatenate([pos, rng.choice(neg, size=take, replace=False)])
    keep.sort()
    return X[keep], y[keep]


def _pick_spec(
    X_sel: np.ndarray,
    y_sel: np.ndarray,
    config: Config,
    fast: bool,
    tune: bool,
    n_trials: int,
) -> tuple[ModelSpec, dict[str, Any], list[ModelSpec]]:
    """Choose a model. Returns (chosen spec, metrics.json metadata, ensemble members)."""
    meta: dict[str, Any]
    if tune:
        from ccfraud.tune import to_spec as tune_to_spec
        from ccfraud.tune import tune as run_tune

        tune_names = ["lightgbm"] if fast else ["lightgbm", "xgboost"]
        tuned = {
            n: run_tune(n, X_sel, y_sel, config=config, n_trials=n_trials) for n in tune_names
        }
        specs = {n: tune_to_spec(t, config.seed) for n, t in tuned.items()}
        best_name = max(tuned, key=lambda n: tuned[n].pr_auc_mean)
        meta = {"tuned": {n: t.as_dict() for n, t in tuned.items()}, "cv_ranking": []}
        return specs[best_name], meta, list(specs.values())

    from ccfraud.select import select_model
    from ccfraud.select import to_spec as select_to_spec

    select_names: list[str] | None = ["logreg", "lightgbm"] if fast else None
    strategies: list[str] | None = ["none", "smote"] if fast else None
    best, ranking = select_model(
        X_sel, y_sel, config=config, names=select_names, strategies=strategies
    )
    gbm_members = [
        select_to_spec(r, config.seed) for r in ranking if r.name in ("lightgbm", "xgboost")
    ]
    meta = {"tuned": None, "cv_ranking": [r.as_dict() for r in ranking]}
    return select_to_spec(best, config.seed), meta, gbm_members[:2]


def _threshold_from_oof(
    spec: ModelSpec, X_train: np.ndarray, y_train: np.ndarray, cv: StratifiedKFold, config: Config
) -> tuple[float, float]:
    """Pick the operating threshold from out-of-fold probabilities on the training set.

    Out-of-fold predictions use ~390 positives (vs. a single hold-out slice's
    far fewer), so the PR curve behind the threshold is stable.
    """
    oof_proba = cross_val_predict(
        spec.build(), X_train, y_train, cv=cv, method="predict_proba"
    )[:, 1]
    threshold = pick_threshold(y_train, oof_proba, config.min_precision)
    at_threshold = oof_proba >= threshold
    precision = float((at_threshold & (y_train == 1)).sum() / max(at_threshold.sum(), 1))
    return threshold, precision


def _fit_and_evaluate(
    spec: ModelSpec,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float,
    cost,
    cv: StratifiedKFold,
) -> tuple[Report, Report, float, Any]:
    """Fit the plain model (the operating point) and an isotonic-calibrated copy."""
    plain = spec.build()
    plain.fit(X_train, y_train)
    plain_report = evaluate(plain, X_test, y_test, threshold=threshold, cost_matrix=cost)

    calibrated = CalibratedClassifierCV(spec.build(), method="isotonic", cv=cv)
    calibrated.fit(X_train, y_train)
    calibrated_report = evaluate(calibrated, X_test, y_test, threshold=threshold, cost_matrix=cost)
    brier = float(brier_score_loss(y_test, calibrated.predict_proba(X_test)[:, 1]))
    return plain_report, calibrated_report, brier, calibrated


def _fit_ensemble(
    members: list[ModelSpec],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float,
    cost,
    config: Config,
) -> tuple[Report | None, Any]:
    """Stack ``members`` if there are at least two; otherwise skip."""
    if len(members) < 2:
        return None, None
    from ccfraud.ensemble import build_stack

    stack = build_stack(members, seed=config.seed, cv_folds=config.cv_folds)
    stack.fit(X_train, y_train)
    return evaluate(stack, X_test, y_test, threshold=threshold, cost_matrix=cost), stack


def run(
    config: Config,
    fast: bool,
    tune: bool = False,
    n_trials: int = 30,
    ensemble: bool = False,
) -> dict:
    df = load_creditcard(config.data_path)
    X_train_df, X_test_df, y_train_s, y_test_s = stratified_split(
        df, test_size=config.test_size, seed=config.seed, target=config.target
    )
    X_train, X_test = X_train_df.to_numpy(), X_test_df.to_numpy()
    y_train, y_test = y_train_s.to_numpy().astype(int), y_test_s.to_numpy().astype(int)
    cost = config.cost_matrix

    reports: dict[str, Report] = {
        "baseline_most_frequent": dummy_baseline(X_train, y_train, X_test, y_test, seed=config.seed)
    }
    lr = build_pipeline(
        LogisticRegression(max_iter=1000, class_weight="balanced"), "none", seed=config.seed
    )
    lr.fit(X_train, y_train)
    reports["baseline_logreg"] = evaluate(lr, X_test, y_test, threshold=0.5, cost_matrix=cost)

    keep_neg = 15_000 if fast else max(config.select_sample - int(y_train.sum()), 1)
    X_sel, y_sel = _subsample_majority(X_train, y_train, keep_neg, config.seed)
    spec, meta, ensemble_members = _pick_spec(X_sel, y_sel, config, fast, tune, n_trials)
    cv = StratifiedKFold(n_splits=config.cv_folds, shuffle=True, random_state=config.seed)
    threshold, oof_precision = _threshold_from_oof(spec, X_train, y_train, cv, config)
    reports["selected"], reports["selected_calibrated"], brier, plot_model = _fit_and_evaluate(
        spec, X_train, y_train, X_test, y_test, threshold, cost, cv
    )

    ensemble_meta = None
    if ensemble:
        ens_report, stack_model = _fit_ensemble(
            ensemble_members, X_train, y_train, X_test, y_test, threshold, cost, config
        )
        if ens_report is not None:
            reports["ensemble"] = ens_report
            plot_model = stack_model
            ensemble_meta = {"members": [m.name for m in ensemble_members]}

    out = _assemble_metrics(
        config,
        run_flags={"fast": fast, "tune": tune, "n_trials": n_trials, "ensemble": ensemble},
        y_train=y_train,
        y_test=y_test,
        spec=spec,
        meta=meta,
        threshold=threshold,
        oof_precision=oof_precision,
        brier=brier,
        ensemble_meta=ensemble_meta,
        reports=reports,
    )
    _write_outputs(out, plot_model, X_test, y_test, config.output_dir)
    return out


def _assemble_metrics(
    config: Config,
    run_flags: dict[str, Any],
    y_train: np.ndarray,
    y_test: np.ndarray,
    spec: ModelSpec,
    meta: dict[str, Any],
    threshold: float,
    oof_precision: float,
    brier: float,
    ensemble_meta: dict[str, Any] | None,
    reports: dict[str, Report],
) -> dict[str, Any]:
    """Shape the ``metrics.json`` payload from already-computed results."""
    return {
        "config": {
            "seed": config.seed,
            "test_size": config.test_size,
            "cv_folds": config.cv_folds,
            "min_precision": config.min_precision,
            "fast": run_flags["fast"],
            "tune": run_flags["tune"],
            "n_trials": run_flags["n_trials"] if run_flags["tune"] else None,
            "ensemble": run_flags["ensemble"],
        },
        "prevalence": {
            "train_positive_rate": float(y_train.mean()),
            "test_positive_rate": float(y_test.mean()),
            "test_positive_count": int(y_test.sum()),
        },
        "selected": spec.as_dict(),
        "tuned": meta["tuned"],
        "ensemble": ensemble_meta,
        "threshold": threshold,
        "min_precision_target": config.min_precision,
        "min_precision_met": oof_precision >= config.min_precision,
        "oof_precision_at_threshold": oof_precision,
        "calibrated_brier_score": brier,
        "cv_ranking": meta["cv_ranking"],
        "reports": {k: v.as_dict() for k, v in reports.items()},
    }


def _write_outputs(out: dict, model, X_test, y_test, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(out, indent=2))

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay

        proba = model.predict_proba(X_test)[:, 1]
        fig, ax = plt.subplots(figsize=(5, 4))
        PrecisionRecallDisplay.from_predictions(y_test, proba, ax=ax)
        ax.set_title("Precision-Recall (held-out test)")
        fig.tight_layout()
        fig.savefig(output_dir / "pr_curve.png", dpi=120)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(5, 4))
        RocCurveDisplay.from_predictions(y_test, proba, ax=ax)
        ax.set_title("ROC (held-out test)")
        fig.tight_layout()
        fig.savefig(output_dir / "roc_curve.png", dpi=120)
        plt.close(fig)
    except Exception as exc:  # noqa: BLE001 - plotting must never fail the run
        print(f"[warn] could not render curves: {exc}")


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    config = replace(
        DEFAULT_CONFIG,
        data_path=args.data_path,
        output_dir=args.output_dir,
        test_size=args.test_size,
        seed=args.seed,
        cv_folds=args.cv_folds,
        min_precision=args.min_precision,
    )
    out = run(
        config,
        fast=args.fast,
        tune=args.tune,
        n_trials=args.n_trials,
        ensemble=args.ensemble,
    )

    sel = out["selected"]
    print(
        f"\nselected: {sel['name']} + {sel['strategy']}  "
        f"CV PR-AUC {sel['cv_pr_auc_mean']:.3f} +/- {sel['cv_pr_auc_std']:.3f}"
    )
    met = "met" if out["min_precision_met"] else "NOT met"
    print(
        f"threshold: {out['threshold']:.4f}  "
        f"(min-precision {out['min_precision_target']:.2f} {met}; "
        f"out-of-fold precision here {out['oof_precision_at_threshold']:.3f})"
    )
    for key in ("selected", "ensemble"):
        rep = out["reports"].get(key)
        if rep is None:
            continue
        print(
            f"{key:>9}  test PR-AUC {rep['pr_auc']:.3f}  ROC-AUC {rep['roc_auc']:.3f}  "
            f"P {rep['precision']:.3f}  R {rep['recall']:.3f}  F1 {rep['f1']:.3f}  "
            f"cost {rep['expected_cost']:.0f}"
        )
    print(f"calibrated Brier score {out['calibrated_brier_score']:.5f}")
    print(f"wrote {args.output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
