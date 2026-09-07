"""Train and evaluate the fraud detector end to end.

Steps
-----
1. Load the dataset and make a stratified train/test split.
2. Score a most-frequent baseline and a logistic-regression baseline.
3. Pick the model:
   * default        - select the best (estimator, imbalance strategy) by CV PR-AUC
   * ``--tune``     - Optuna search over LightGBM and XGBoost, keep the better
4. Choose the decision threshold from out-of-fold probabilities so the fraud
   precision meets ``config.min_precision``.
5. Calibrate the winner's probabilities (isotonic, internal CV).
6. Optionally (``--ensemble``) stack the gradient-boosted models.
7. Evaluate on the held-out test set and write ``output/metrics.json`` + curves.

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

import numpy as np

from ccfraud.config import DEFAULT_CONFIG, Config
from ccfraud.data import load_creditcard, stratified_split
from ccfraud.evaluate import Report, dummy_baseline, evaluate, pick_threshold


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


def run(
    config: Config,
    fast: bool,
    tune: bool = False,
    n_trials: int = 30,
    ensemble: bool = False,
) -> dict:
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import brier_score_loss
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    from ccfraud.pipeline import build_pipeline
    from ccfraud.select import CANDIDATES, select_model

    df = load_creditcard(config.data_path)
    X_train_df, X_test_df, y_train_s, y_test_s = stratified_split(
        df, test_size=config.test_size, seed=config.seed, target=config.target
    )
    X_train = X_train_df.to_numpy()
    X_test = X_test_df.to_numpy()
    y_train = y_train_s.to_numpy().astype(int)
    y_test = y_test_s.to_numpy().astype(int)

    cost = config.cost_matrix
    reports: dict[str, Report] = {}
    reports["baseline_most_frequent"] = dummy_baseline(
        X_train, y_train, X_test, y_test, seed=config.seed
    )

    lr = build_pipeline(
        LogisticRegression(max_iter=1000, class_weight="balanced"), "none", seed=config.seed
    )
    lr.fit(X_train, y_train)
    reports["baseline_logreg"] = evaluate(lr, X_test, y_test, threshold=0.5, cost_matrix=cost)

    # selection subsample (winner is refit on the full training set)
    if fast:
        keep_neg = 15_000
        names: list[str] | None = ["logreg", "lightgbm"]
        strategies: list[str] | None = ["none", "smote"]
    else:
        keep_neg = max(config.select_sample - int(y_train.sum()), 1)
        names = None
        strategies = None
    X_sel, y_sel = _subsample_majority(X_train, y_train, keep_neg, config.seed)

    cv = StratifiedKFold(n_splits=config.cv_folds, shuffle=True, random_state=config.seed)
    tuned_section: dict | None = None
    ensemble_members: list[tuple[str, dict, str]] = []

    # --- choose the single model ---
    if tune:
        from ccfraud.tune import make_estimator
        from ccfraud.tune import tune as run_tune

        tune_names = ["lightgbm"] if fast else ["lightgbm", "xgboost"]
        tuned = {n: run_tune(n, X_sel, y_sel, config=config, n_trials=n_trials) for n in tune_names}
        best_tr = max(tuned.values(), key=lambda t: t.pr_auc_mean)
        tuned_section = {n: t.as_dict() for n, t in tuned.items()}
        selected_meta = {
            "name": f"{best_tr.estimator} (tuned)",
            "strategy": best_tr.strategy,
            "cv_pr_auc_mean": best_tr.pr_auc_mean,
            "cv_pr_auc_std": best_tr.pr_auc_std,
        }

        def make_base():
            est = make_estimator(best_tr.estimator, best_tr.params, config.seed)
            return build_pipeline(est, best_tr.strategy, seed=config.seed)

        ensemble_members = [(t.estimator, t.params, t.strategy) for t in tuned.values()]
        ranking_dump: list[dict] = []
    else:
        best, ranking = select_model(
            X_sel, y_sel, config=config, names=names, strategies=strategies
        )
        selected_meta = {
            "name": best.name,
            "strategy": best.strategy,
            "cv_pr_auc_mean": best.pr_auc_mean,
            "cv_pr_auc_std": best.pr_auc_std,
        }

        def make_base():
            return build_pipeline(CANDIDATES[best.name](), best.strategy, seed=config.seed)

        gbm = [r for r in ranking if r.name in ("lightgbm", "xgboost")][:2]
        ensemble_members = [(r.name, {}, r.strategy) for r in gbm]
        ranking_dump = [r.as_dict() for r in ranking]

    # --- operating threshold from out-of-fold probabilities on the training set ---
    oof_proba = cross_val_predict(
        make_base(), X_train, y_train, cv=cv, method="predict_proba"
    )[:, 1]
    threshold = pick_threshold(y_train, oof_proba, config.min_precision)
    oof_precision_at_threshold = float(
        ((oof_proba >= threshold) & (y_train == 1)).sum()
        / max((oof_proba >= threshold).sum(), 1)
    )

    # --- refit on the full training set: uncalibrated (operating point) + calibrated ---
    base_pipe = make_base()
    base_pipe.fit(X_train, y_train)
    reports["selected"] = evaluate(base_pipe, X_test, y_test, threshold=threshold, cost_matrix=cost)

    final = CalibratedClassifierCV(make_base(), method="isotonic", cv=cv)
    final.fit(X_train, y_train)
    reports["selected_calibrated"] = evaluate(
        final, X_test, y_test, threshold=threshold, cost_matrix=cost
    )
    brier = float(brier_score_loss(y_test, final.predict_proba(X_test)[:, 1]))
    plot_model = final

    # --- optional stacked ensemble ---
    ensemble_section: dict | None = None
    if ensemble and len(ensemble_members) >= 2:
        from ccfraud.ensemble import build_stack
        from ccfraud.tune import make_estimator

        def _member_est(name: str, params: dict):
            return make_estimator(name, params, config.seed) if params else CANDIDATES[name]()

        members = [
            (name, _member_est(name, params), strat)
            for name, params, strat in ensemble_members
        ]
        stack = build_stack(members, seed=config.seed, cv_folds=config.cv_folds)
        stack.fit(X_train, y_train)
        stack_rep = evaluate(stack, X_test, y_test, threshold=threshold, cost_matrix=cost)
        reports["ensemble"] = stack_rep
        ensemble_section = {
            "members": [f"{n}+{s}" for n, _p, s in ensemble_members],
            "note": "PR-AUC / ROC-AUC are threshold-free; P/R/F1 are at the shared threshold",
        }
        plot_model = stack

    out = {
        "config": {
            "seed": config.seed,
            "test_size": config.test_size,
            "cv_folds": config.cv_folds,
            "min_precision": config.min_precision,
            "fast": fast,
            "tune": tune,
            "n_trials": n_trials if tune else None,
            "ensemble": ensemble,
        },
        "prevalence": {
            "train_positive_rate": float(y_train.mean()),
            "test_positive_rate": float(y_test.mean()),
            "test_positive_count": int(y_test.sum()),
        },
        "selected": selected_meta,
        "tuned": tuned_section,
        "ensemble": ensemble_section,
        "threshold": threshold,
        "min_precision_target": config.min_precision,
        "min_precision_met": oof_precision_at_threshold >= config.min_precision,
        "oof_precision_at_threshold": oof_precision_at_threshold,
        "calibrated_brier_score": brier,
        "cv_ranking": ranking_dump,
        "reports": {k: v.as_dict() for k, v in reports.items()},
    }
    _write_outputs(out, plot_model, X_test, y_test, config.output_dir)
    return out


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
