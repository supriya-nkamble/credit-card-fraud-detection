"""Train and evaluate the fraud detector end to end.

Steps
-----
1. Load the dataset and make a stratified train/test split.
2. Score a most-frequent baseline and a logistic-regression baseline.
3. Select the best (estimator, imbalance strategy) by mean CV PR-AUC.
4. Choose the decision threshold from out-of-fold probabilities on the
   training set so the fraud precision meets ``config.min_precision``.
5. Calibrate the winner's probabilities (isotonic, internal CV).
6. Evaluate on the held-out test set and write ``output/metrics.json``
   plus precision-recall and ROC curve PNGs.

Usage
-----
    ccfraud-train --help             # installed console script
    python -m ccfraud.train          # full run
    python scripts/train.py --fast   # small grid, subsampled; for a smoke test
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


def _base_pipeline(best, seed: int):
    """Rebuild the fresh, unfitted pipeline for the winning (name, strategy)."""
    from ccfraud.pipeline import build_pipeline
    from ccfraud.select import CANDIDATES

    return build_pipeline(CANDIDATES[best.name](), best.strategy, seed=seed)


def run(config: Config, fast: bool) -> dict:
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, train_test_split

    from ccfraud.pipeline import build_pipeline
    from ccfraud.select import select_model

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

    # --- model selection ---
    if fast:
        X_sel, y_sel = _subsample_majority(X_train, y_train, 15_000, config.seed)
        names: list[str] | None = ["logreg", "lightgbm"]
        strategies: list[str] | None = ["none", "smote"]
    else:
        X_sel, y_sel = X_train, y_train
        names = None
        strategies = None

    best, ranking = select_model(
        X_sel, y_sel, config=config, names=names, strategies=strategies
    )
    cv = StratifiedKFold(n_splits=config.cv_folds, shuffle=True, random_state=config.seed)

    # --- pick the threshold on calibrated probabilities the model has NOT seen ---
    # Split the training set: fit+calibrate on one part, choose the operating
    # threshold on the other. A threshold picked on uncalibrated scores would not
    # transfer to the calibrated model.
    X_fit, X_thr, y_fit, y_thr = train_test_split(
        X_train, y_train, test_size=0.25, random_state=config.seed, stratify=y_train
    )
    cal_for_thr = CalibratedClassifierCV(
        _base_pipeline(best, config.seed), method="isotonic", cv=cv
    )
    cal_for_thr.fit(X_fit, y_fit)
    thr_proba = cal_for_thr.predict_proba(X_thr)[:, 1]
    threshold = pick_threshold(y_thr, thr_proba, config.min_precision)

    # --- refit the calibrated winner on the full training set, score on test ---
    final = CalibratedClassifierCV(
        _base_pipeline(best, config.seed), method="isotonic", cv=cv
    )
    final.fit(X_train, y_train)
    reports["selected_calibrated"] = evaluate(
        final, X_test, y_test, threshold=threshold, cost_matrix=cost
    )

    base_pipe = _base_pipeline(best, config.seed)
    base_pipe.fit(X_train, y_train)
    reports["selected_uncalibrated"] = evaluate(
        base_pipe, X_test, y_test, threshold=0.5, cost_matrix=cost
    )

    out = {
        "config": {
            "seed": config.seed,
            "test_size": config.test_size,
            "cv_folds": config.cv_folds,
            "min_precision": config.min_precision,
            "fast": fast,
        },
        "prevalence": {
            "train_positive_rate": float(y_train.mean()),
            "test_positive_rate": float(y_test.mean()),
            "test_positive_count": int(y_test.sum()),
        },
        "selected": {
            "name": best.name,
            "strategy": best.strategy,
            "cv_pr_auc_mean": best.pr_auc_mean,
            "cv_pr_auc_std": best.pr_auc_std,
        },
        "threshold": threshold,
        "cv_ranking": [r.as_dict() for r in ranking],
        "reports": {k: v.as_dict() for k, v in reports.items()},
    }
    _write_outputs(out, final, X_test, y_test, config.output_dir)
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
    out = run(config, fast=args.fast)

    sel = out["selected"]
    rep = out["reports"]["selected_calibrated"]
    print(
        f"\nselected: {sel['name']} + {sel['strategy']}  "
        f"CV PR-AUC {sel['cv_pr_auc_mean']:.3f} +/- {sel['cv_pr_auc_std']:.3f}"
    )
    print(f"threshold: {out['threshold']:.4f}")
    print(
        f"held-out test  PR-AUC {rep['pr_auc']:.3f}  ROC-AUC {rep['roc_auc']:.3f}  "
        f"precision {rep['precision']:.3f}  recall {rep['recall']:.3f}  F1 {rep['f1']:.3f}"
    )
    print(f"wrote {args.output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
